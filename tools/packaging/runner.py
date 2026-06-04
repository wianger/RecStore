#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

def run_cmd(cmd, **kwargs):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)

def ensure_env(pkg_root: str):
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = f"{pkg_root}/deps/lib:{pkg_root}/lib"
    if ld:
        new_ld = f"{new_ld}:{ld}"
    os.environ["LD_LIBRARY_PATH"] = new_ld
    print(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")

def preflight(bin_path: str, inject_rpath: bool):
    print(f"Preflight ldd: {bin_path}")
    res = run_cmd(["ldd", bin_path])
    print(res.stdout)
    need_inject = ("not found" in res.stdout) and inject_rpath
    if need_inject:
        patchelf = shutil.which("patchelf")
        if patchelf:
            print("Injecting rpath via patchelf")
            run_cmd([patchelf, "--set-rpath", "$ORIGIN/../deps/lib:$ORIGIN/../lib", bin_path])
            pr = run_cmd([patchelf, "--print-rpath", bin_path])
            print("New rpath:", pr.stdout)
            print("Reflight ldd after rpath:")
            print(run_cmd(["ldd", bin_path]).stdout)
        else:
            print("patchelf not available; skipping rpath injection")

    return res.stdout

def main():
    ap = argparse.ArgumentParser(description="Generic runner for packed RecStore artifacts")
    ap.add_argument("package_root", help="Root directory of extracted package")
    ap.add_argument("binary_rel_path", help="Relative path to binary within package root")
    ap.add_argument("--ready-pattern", default="listening on", help="Log readiness pattern")
    ap.add_argument("--timeout", type=int, default=180, help="Timeout seconds for readiness")
    ap.add_argument("--log", default=None, help="Log file path (default: <pkg_root>/../logs/run.log)")
    ap.add_argument("--inject-rpath", action="store_true", help="Attempt rpath injection if deps unresolved")
    ap.add_argument("--keep-alive", action="store_true", help="Leave process running after readiness (caller stops it)")
    args = ap.parse_args()

    pkg_root = os.path.abspath(args.package_root)
    bin_path = os.path.join(pkg_root, args.binary_rel_path)
    if not os.path.exists(bin_path) or not os.access(bin_path, os.X_OK):
        print(f"Binary not found or not executable: {bin_path}", file=sys.stderr)
        sys.exit(1)

    default_log_dir = os.path.abspath(os.path.join(pkg_root, "..", "logs"))
    os.makedirs(default_log_dir, exist_ok=True)
    log_file = args.log or os.path.join(default_log_dir, "run.log")
    log_parent = os.path.abspath(os.path.dirname(log_file))
    os.makedirs(log_parent, exist_ok=True)

    ensure_env(pkg_root)
    deps_dir = os.path.join(pkg_root, "deps", "lib")
    if os.path.isdir(deps_dir):
        try:
            print("Listing deps/lib:")
            for n in sorted(os.listdir(deps_dir)):
                print("  ", n)
        except Exception:
            pass

    ldd_out = preflight(bin_path, inject_rpath=args.inject_rpath)

    not_found = []
    for line in ldd_out.splitlines():
        if "=> not found" in line:
            name = line.split()[0]
            not_found.append(name)
    if not_found:
        deps_dir = os.path.join(pkg_root, "deps", "lib")
        if os.path.isdir(deps_dir):
            for name in not_found:
                prefix = name.split('.so')[0]
                try:
                    candidates = [f for f in os.listdir(deps_dir) if f.startswith(prefix) and f.endswith(tuple([".so", ".so.1", ".so.2", ".so.25", ".so.25.2"]))]
                    if candidates:
                        target = sorted(candidates, key=len, reverse=True)[0]
                        link_path = os.path.join(deps_dir, name)
                        target_rel = target
                        if not os.path.exists(link_path):
                            os.symlink(target_rel, link_path)
                            print(f"Created symlink {name} -> {target_rel}")
                except Exception:
                    pass
        # Re-run ldd to show updated resolution
        print("Reflight ldd after symlink fixes:")
        print(run_cmd(["ldd", bin_path]).stdout)

    print(f"Starting {bin_path} ... (log: {log_file})")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([bin_path], stdout=lf, stderr=subprocess.STDOUT)

    start_ts = time.time()
    pattern = re.compile(args.ready_pattern)

    try:
        while True:
            # check readiness by scanning log file
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                    content = lf.read()
                    if pattern.search(content):
                        print(f"Process ready (matched pattern: {args.ready_pattern})")
                        break
            except FileNotFoundError:
                pass

            if proc.poll() is not None:
                print("Process exited prematurely. Logs:")
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                        print(lf.read())
                except Exception:
                    pass
                sys.exit(1)

            if time.time() - start_ts >= args.timeout:
                print(f"Timeout ({args.timeout}s) waiting for readiness. Logs:")
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                        print(lf.read())
                except Exception:
                    pass
                sys.exit(1)

            time.sleep(1)
    finally:
        if not args.keep_alive:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    if args.keep_alive:
        print("Process left running (keep-alive). Caller is responsible for stopping it.")
    else:
        print("Run completed successfully.")

if __name__ == "__main__":
    main()
