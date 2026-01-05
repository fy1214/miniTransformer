import os
import re
import glob
import shutil
import pathlib
import subprocess
import setuptools
from typing import Tuple
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def cuda_version() -> Tuple[int, ...]:
    # Try to locate NVCC
    nvcc_bin = None

    # Check in CUDA_HOME environment variable
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        nvcc_bin = pathlib.Path(cuda_home) / "bin" / "nvcc"

    # Check in PATH if not found in CUDA_HOME
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            nvcc_bin = pathlib.Path(nvcc_path)

    # Check in default directory
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_bin = pathlib.Path("/usr/local/cuda/bin/nvcc")

    if not nvcc_bin.is_file():
        raise FileNotFoundError("Could not locate NVCC. Please ensure CUDA is installed and accessible.")

    # Run nvcc to get the version
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run NVCC: {e}")

    # Parse the version string
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        raise ValueError("Could not parse CUDA version from NVCC output.")

    version = tuple(map(int, match.group(1).split(".")))
    return version


def setup_extensions() -> setuptools.Extension:
    debug_mode = os.getenv("DEBUG", "0") == "1"

    # compiler flags
    cxx_flags = [
        "-O3" if not debug_mode else "-O0",
        "-fvisibility=hidden",
        "-fdiagnostics-color=always",
        "-std=c++17",
    ]
    nvcc_flags = [
        "-O3" if not debug_mode else "-O0",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++17",
        "-diag-suppress=3189",  # suppress warning from libtorch headers
    ]
    extra_link_args = ["-lcuda", ]

    # debug mode flags
    if debug_mode:
        print("Compiling in debug mode")
        cxx_flags.append("-g")
        nvcc_flags.append("-g")
        extra_link_args.extend(["-O0", "-g"])

    # version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version < (12, 6):
            raise RuntimeError(f"HYBRID requires CUDA 12.6 or newer. Got {version}")
        # add nvcc flags for specific architectures
        cuda_archs = os.getenv("HYBRID_CUDA_ARCHS", "89;90;90a;100a")
        for arch in cuda_archs.split(";"):
            nvcc_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

    # define sources and include directories
    root_dir = pathlib.Path(__file__).resolve().parent
    extensions_dir = root_dir / "csrc"
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
    sources += list(glob.glob(os.path.join(extensions_dir, "**/*.cu"), recursive=True))
    include_dirs = [
        str(extensions_dir),
        # cutlass include directories
        #str(root_dir / "third-party" / "cutlass" / "include"),
        #str(root_dir / "third-party" / "cutlass" / "include" / "cute"),
        #str(root_dir / "third-party" / "cutlass" / "include" / "cutlass"),
        #str(root_dir / "third-party" / "cutlass" / "tools" / "util" / "include"),
    ]

    # construct the extension
    ext_modules = CUDAExtension(
        name="fp4_gemm",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        extra_link_args=extra_link_args,
    )

    return [ext_modules]


def get_requirements() -> list[str]:
    return ['torch']

if __name__ == '__main__':
    setup(
        name='fp4_gemm',
        version='1.0.0',
        install_requires=get_requirements(),
        ext_modules=setup_extensions(),
        cmdclass={
            # "develop": PostDevelopCommand,
            "build_ext": BuildExtension,
        },  # 必须添加以支持混合编译
    )
