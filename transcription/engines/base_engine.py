import torch
from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(
        self,
        modelName: str,
        language: str,
        dType: torch.dtype,
        device: str
    ):
        self.modelName = modelName
        self.device = device
        self.language = language
        self.dType = dType
    
    @abstractmethod
    def loadModel(self) -> None:
        pass

    @abstractmethod
    def unloadModel(self) -> None:
        pass

    @abstractmethod
    def transcribeBatch(
        self, 
        batch
    ) -> str:
        pass