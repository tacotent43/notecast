from dataclasses import dataclass

import torch

@dataclass
class Configuration:
    # add new models
    device: str = "cuda"
    modelName: str = "openai/whisper-large-v2"
    chunkSize: int = 30
    batchSize: int = 16
    dataType: str = "torch.float16"

    _dtype_map = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
    }

    dType: torch.dtype = None

    def __post_init__(self):
        self.dType = self._dtype_map[self.dataType]
