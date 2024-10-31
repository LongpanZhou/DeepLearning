import torch

def check_cuda_support():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Support: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    return device

if __name__ == "__main__":
    check_cuda_support()