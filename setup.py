import os
import re
import glob
import shutil
import pathlib
import subprocess
import setuptools
from setuptools import setup


ROOT_DIR = pathlib.Path(__file__).resolve().parent


def cuda_version() -> tuple[int, ...]:
    candidates = []
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        candidates.append(pathlib.Path(cuda_home) / "bin" / "nvcc")
    nvcc_in_path = shutil.which("nvcc")
    if nvcc_in_path:
        candidates.append(pathlib.Path(nvcc_in_path))
    candidates.append(pathlib.Path("/usr/local/cuda/bin/nvcc"))

    nvcc_bin = next((p for p in candidates if p.is_file()), None)
    if nvcc_bin is None:
        raise FileNotFoundError("Could not locate NVCC. Ensure CUDA is installed and accessible.")

    result = subprocess.run([nvcc_bin, "-V"], capture_output=True, check=True, text=True)
    match = re.search(r"release\s*([\d.]+)", result.stdout)
    if not match:
        raise ValueError("Could not parse CUDA version from NVCC output.")
    return tuple(map(int, match.group(1).split(".")))


def build_extensions() -> list[setuptools.Extension]:
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        return []

    debug_mode = os.getenv("DEBUG", "0") == "1"
    opt_flag = "-O0" if debug_mode else "-O3"

    cxx_flags = [
        opt_flag,
        "-fvisibility=hidden",
        "-fdiagnostics-color=always",
        "-std=c++17",
    ]
    nvcc_flags = [
        opt_flag,
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "-include", "/usr/include/semaphore.h",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++17",
        "-diag-suppress=3189",
    ]
    link_args = ["-lcuda", "-lnccl", "-lcublasLt"]

    if debug_mode:
        print("Compiling in debug mode")
        cxx_flags.append("-g")
        nvcc_flags.append("-g")
        link_args.extend(["-O0", "-g"])

    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version < (12, 6):
            raise RuntimeError(f"miniTransformer requires CUDA 12.6 or newer. Got {version}")
        cuda_archs = os.getenv("HYBRID_CUDA_ARCHS", "100a;103a")
        for arch in cuda_archs.split(";"):
            nvcc_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

    csrc_dir = ROOT_DIR / "miniTransformer" / "csrc"
    cutlass_dir = ROOT_DIR / "third-party" / "cutlass"
    sources = (
        glob.glob(str(csrc_dir / "**/*.cpp"), recursive=True)
        + glob.glob(str(csrc_dir / "**/*.cu"), recursive=True)
    )
    include_dirs = [
        str(csrc_dir),
        str(cutlass_dir / "include"),
        str(cutlass_dir / "include" / "cute"),
        str(cutlass_dir / "include" / "cutlass"),
        str(cutlass_dir / "tools" / "util" / "include"),
    ]

    return [CUDAExtension(
        name="fp4_gemm",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        extra_link_args=link_args,
    )]


def get_cmdclass() -> dict:
    try:
        from torch.utils.cpp_extension import BuildExtension
        return {"build_ext": BuildExtension}
    except ImportError:
        return {}


setup(
    packages=setuptools.find_packages(exclude=["tests*", "doc*"]),
    ext_modules=build_extensions(),
    cmdclass=get_cmdclass(),
)
