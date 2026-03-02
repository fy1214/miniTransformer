import os
import psutil
import torch


def print_memory_usage(msg=""):
    """打印当前进程的内存使用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # RSS: Resident Set Size - 实际物理内存
    # VMS: Virtual Memory Size - 虚拟内存
    print(f"\n=== 内存使用 [{msg}] ===")
    print(f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB")
    print(f"VMS: {mem_info.vms / 1024 ** 2:.2f} MB")
    print(f"内存百分比: {process.memory_percent():.2f}%")

    # 系统内存
    system_mem = psutil.virtual_memory()
    print(f"系统内存使用: {system_mem.percent}%")
    print(f"系统可用内存: {system_mem.available / 1024 ** 3:.2f} GB")


def print_gpu_memory(id, msg=""):
    """打印 GPU 内存使用"""
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    print(f"\n=== GPU 内存 [{msg}] ===")
    print(f"GPU {id}: {torch.cuda.get_device_name(id)}")

    # 已分配内存
    allocated = torch.cuda.memory_allocated(id) / 1024 ** 3
    # 缓存内存
    cached = torch.cuda.memory_reserved(id) / 1024 ** 3
    # 最大分配
    max_allocated = torch.cuda.max_memory_allocated(id) / 1024 ** 3
    # 最大缓存
    max_cached = torch.cuda.max_memory_reserved(id) / 1024 ** 3

    print(f"  已分配: {allocated:.2f} GB")
    print(f"  缓存: {cached:.2f} GB")
    print(f"  最大分配: {max_allocated:.2f} GB")
    print(f"  最大缓存: {max_cached:.2f} GB")