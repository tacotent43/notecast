from dataclasses import dataclass
import torch

@dataclass
class DeviceConfiguration:
    """
    Configurations for Whisper model on different devices.

    Attributes:
        device (str): Type of device. Possible options: "cuda", "cpu", "mps".
        
        model_name (str): Whisper models. Possible options:
            - "openai/whisper-tiny"
            - "openai/whisper-small"
            - "openai/whisper-medium"
            - "openai/whisper-large"
            - "openai/whisper-large-v2"
        
        batch_size (int): Chunks in one batch. Selected for VRAM.
        
        chunk_length_s (int): Length of one audio chunk in seconds. Smaller -> less VRAM.
        
        data_type (str): custom data type of model. Variants:
            - torch.float16 - for GPUs
            - torch.float32 - for CPU / weak GPU
            - torch.bfloat16 - for GPUs which has BF16 support
    """
    device: str = "cuda"
    model_name: str = "openai/whisper-large-v2"
    batch_size: int = 16
    chunk_length_s: int = 30
    data_type: str = "torch.float16"
    
    _dtype_map = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16
    }
    
    torch_dtype: torch.dtype = None
    
    def __post_init__(self):
        self.torch_dtype = self._dtype_map[self.data_type]