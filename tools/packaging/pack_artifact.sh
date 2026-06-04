#!/bin/bash
set -euo pipefail

# RecStore CI packer: collects executables or .so libraries and their runtime deps.
# Usage:
#   tools/packaging/pack_artifact.sh <output-tar.gz> <artifact1> [artifact2 ...]
# Examples:
#   tools/packaging/pack_artifact.sh build/packed-bin.tar.gz build/bin/grpc_ps_server
#   tools/packaging/pack_artifact.sh build/packed-lib.tar.gz build/lib/lib_recstore_ops.so

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output-tar.gz> <artifact1> [artifact2 ...]" >&2
  exit 2
fi

OUTPUT_TAR="$1"
shift

WORK_DIR="$(mktemp -d)"
PACKAGE_ROOT="${WORK_DIR}/package"
BIN_DIR="${PACKAGE_ROOT}/bin"
LIB_DIR="${PACKAGE_ROOT}/lib"
DEPS_DIR="${PACKAGE_ROOT}/deps/lib"
MANIFEST="${PACKAGE_ROOT}/manifest.txt"

mkdir -p "${BIN_DIR}" "${LIB_DIR}" "${DEPS_DIR}"
echo "RecStore Pack Manifest" > "${MANIFEST}"
date >> "${MANIFEST}"

copy_unique() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"
  if [[ -f "${dest_dir}/${base}" ]]; then
    return 0
  fi
  if [[ -L "$src" ]]; then
    local target
    target="$(readlink -f "$src")"
    if [[ -f "$target" ]]; then
      local tbase
      tbase="$(basename "$target")"
      if [[ ! -f "${dest_dir}/${tbase}" ]]; then
        cp -a "$target" "${dest_dir}/"
      fi
      ln -sf "$tbase" "${dest_dir}/${base}"
    else
      cp -L "$src" "${dest_dir}/"
    fi
  else
    cp -a "$src" "${dest_dir}/"
  fi
}

should_exclude_dependency() {
  local dep="$1"
  local base
  base="$(basename "$dep")"

  case "$base" in
    linux-vdso.so*|ld-linux*.so*|libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|libresolv.so*|libutil.so*|libgcc_s.so*|libstdc++.so*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

parse_ldd_and_copy_deps() {
  local target="$1"
  local deps_output

  echo "\n[lddtree] ${target}" >> "${MANIFEST}"
  if command -v lddtree >/dev/null 2>&1; then
    deps_output="$(lddtree -l "$target")"
  else
    echo "lddtree not available; falling back to ldd parsing" >> "${MANIFEST}"
    deps_output="$(ldd "$target" | awk '{ for(i=1;i<=NF;i++){ if ($i ~ /^\//) print $i; } }')"
  fi

  printf '%s\n' "$deps_output" | tee -a "${MANIFEST}" >/dev/null
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    [[ -f "$dep" ]] || continue
    if should_exclude_dependency "$dep"; then
      echo "[excluded-host-lib] $dep" >> "${MANIFEST}"
      continue
    fi
    copy_unique "$dep" "${DEPS_DIR}"
  done <<EOF
$deps_output
EOF
}

resolve_artifact_path() {
  local path="$1"
  if [[ -f "$path" ]]; then
    echo "$path"
    return 0
  fi

  if [[ "$path" == */lib_recstore_ops.so ]]; then
    local alt="${path%lib_recstore_ops.so}_recstore_ops.so"
    if [[ -f "$alt" ]]; then
      echo "$alt"
      return 0
    fi
  fi

  return 1
}

detect_type() {
  local path="$1"
  local info
  info="$(file -Lb "$path" || true)"
  if [[ "$info" == *"ELF"* && "$info" == *"executable"* ]]; then
    echo "exe"
  elif [[ "$path" == *.so* || ( "$info" == *"ELF"* && "$info" == *"shared object"* ) ]]; then
    echo "so"
  else
    echo "unknown"
  fi
}

for artifact in "$@"; do
  artifact_path="$(resolve_artifact_path "$artifact" || true)"
  if [[ -z "${artifact_path:-}" ]]; then
    echo "Artifact not found: $artifact" >&2
    exit 1
  fi

  type="$(detect_type "$artifact_path")"
  case "$type" in
    exe)
      copy_unique "$artifact_path" "${BIN_DIR}"
      parse_ldd_and_copy_deps "$artifact_path"
      ;;
    so)
      copy_unique "$artifact_path" "${LIB_DIR}"
      parse_ldd_and_copy_deps "$artifact_path"
      ;;
    *)
      echo "Skipping unknown type: $artifact_path" >&2
      ;;
  esac
done

inject_rpath() {
  local target="$1"
  local origin_rpath="$2"
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --set-rpath "$origin_rpath" "$target" || true
  else
    echo "patchelf not available; skip rpath injection for $target" >&2
  fi
}

if command -v file >/dev/null 2>&1; then
  for f in "${BIN_DIR}"/*; do
    [[ -f "$f" ]] || continue
    if file -Lb "$f" | grep -q "ELF"; then
      inject_rpath "$f" "\$ORIGIN/../deps/lib:\$ORIGIN/../lib"
    fi
  done
  for f in "${LIB_DIR}"/*; do
    [[ -f "$f" ]] || continue
    if file -Lb "$f" | grep -q "ELF"; then
      inject_rpath "$f" "\$ORIGIN/../deps/lib:\$ORIGIN"
    fi
  done
else
  echo "file command not available; cannot detect ELF for rpath injection" >&2
fi

cat > "${PACKAGE_ROOT}/README.txt" <<EOF
This package contains selected RecStore binaries and/or shared libraries
and their runtime dependencies as resolved by ldd on the build host.

Structure:
- bin/: executables
- lib/: shared libraries (.so)
- deps/lib/: copied shared library dependencies from ldd
- manifest.txt: ldd outputs and timestamp

Additionally, when available, dependencies are resolved via lddtree (pax-utils)
and recorded here. For each packaged target, the lddtree listing is appended
to manifest.txt for diagnostic purposes.

Note: Host-provided runtime libraries such as glibc and the system loader are intentionally excluded.
EOF

tar -C "${WORK_DIR}" --numeric-owner --owner=0 --group=0 -czf "${OUTPUT_TAR}" package

echo "Packed artifacts to ${OUTPUT_TAR}"
echo "Contents:" && tar -tzf "${OUTPUT_TAR}" | sed 's/^/  /'

rm -rf "${WORK_DIR}"
