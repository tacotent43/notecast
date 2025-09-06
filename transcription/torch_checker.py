import torch

def check_torch() -> None:
    print("=== Checking PyTorch ===")
    print(f"Torch version: {torch.version}")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPU: {torch.cuda.device_count()}")
        print(f"Name of GPU: {torch.cuda.get_device_name(0)}")
    print("=== Check completed ===")