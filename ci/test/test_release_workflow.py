from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pytorch_framework_build_is_not_gated_by_cuda() -> None:
    cmake = (REPO_ROOT / "src/framework/CMakeLists.txt").read_text()

    assert "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)" in cmake
    assert "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)" in cmake
    assert "_recstore_libtorch_requires_cuda" in cmake

    gpu_index = cmake.index("add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)")
    pytorch_index = cmake.index("add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)")
    first_endif_after_gpu = cmake.find("endif()", gpu_index)

    assert first_endif_after_gpu != -1
    assert pytorch_index > first_endif_after_gpu


def test_release_workflow_builds_cpu_libtorch_ops_package() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "LIBTORCH_VARIANT=cpu" in workflow
    assert "cmake -DENABLE_CUDA=OFF" in workflow
    assert "bash ci/pack/pack_artifact.sh" in workflow
    assert "build/lib/lib_recstore_ops.so" in workflow
