# from logging import Logger
import time
import torch
import gc
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transcription.engines.base_engine import BaseEngine

class WhisperEngine(BaseEngine):
    def loadModel(self) -> None:
        self.processor = WhisperProcessor.from_pretrained(self.modelName)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.modelName,
            torch_dtype = self.dType # check twice
        ).to(self.device) # ??? recheck
    
    def unloadModel(self) -> None:
        self.model = None
        self.processor = None

        # TODO: MPS?
        if self.device == "cuda": 
            torch.cuda.empty_cache()
    
    def transcribeBatch(
        self,
        batch,
    ) -> str:
        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(
            batch, 
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        
        input_features = inputs.input_features.to(self.device).to(self.dType)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task="transcribe",
                temperature=0.0,
            )
        
        batchText = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True,
        )

        inputs = None
        input_features = None
        predicted_ids = None
        gc.collect()

        # maybe do here something with MPS?
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return " ".join(batchText)
        