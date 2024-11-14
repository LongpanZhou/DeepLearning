import torch
from torch import nn

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

def select_gpu_with_most_memory(device, n):
    if device.type != "cuda":
        print("No GPUs are available.")
        return None

    if n == 0:
        print("No GPUs are available.")
        return None

    if n == 1:
        print("Only one GPU is available.")
        return None

    max_free_memory = 0
    selected_gpu = 0

    for i in range(n):
        memory_summary = torch.cuda.memory_summary(i)

        for line in memory_summary.splitlines():
            if "Free" in line:
                parts = line.split()
                if parts:
                    try:
                        free_memory_str = parts[-2]
                        free_memory = float(free_memory_str)
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            selected_gpu = i
                    except ValueError:
                        print(f"Could not parse free memory value from line: {line}")
                        continue

    print(f"Selected GPU {selected_gpu} with {max_free_memory} MB of free memory.")
    return selected_gpu

def check_parallel_availability(device, n, model, data_loader):
    if n == 1:
        return model

    sample_data, _ = next(iter(data_loader))
    sample_data = sample_data.to(device)

    try:
        with torch.no_grad():
            _ = model(sample_data)
        print("Single GPU memory is sufficient. Not using DataParallel.")
        return model
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Memory exceeded on single GPU, using DataParallel.")
            model = nn.DataParallel(model)
            model.to(device)
        else:
            raise e  # Raise other errors
    return model

if __name__ == "__main__":
    check_cuda_support()