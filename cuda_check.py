import torch

def check_cuda_support():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Support: {torch.version.cuda}")

    n = torch.cuda.device_count()

    if n > 0:
        print(f"Using {n} GPU(s)")
        for i in range(n):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

    print(f"Using device: {device}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    return [device, n]

if __name__ == "__main__":
    check_cuda_support()