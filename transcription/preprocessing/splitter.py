import torch
from typing import List

class Splitter:
    def __init__(
        self,
        chunkSize: int,
        batchSize: int,
    ) -> None:
        self.chunkSize = chunkSize * 16000 # 16 kHz after resampling
        self.batchSize = batchSize

    # maybe raise some exceptions here?
    def _split_to_chunks(
        self, 
        waveform: torch.Tensor,
    ) -> List:
        totalSamples = waveform.shape[0]
        chunksCount = (totalSamples + self.chunkSize - 1) // self.chunkSize

        chunks: List = []
        # tqdm or something here? 
        for chunkNum in range(chunksCount):
            start = chunkNum * self.chunkSize
            end = min((chunkNum + 1) * self.chunkSize, totalSamples)

            chunk = waveform[start : end].cpu().numpy().astype("float32")
            chunks.append(chunk)
        
        return chunks

    def _split_to_batches(
        self,
        chunks: List,
    ) -> List:
        batches: List = []

        for i in range(0, len(chunks), self.batchSize):
            batch = chunks[i : i + self.batchSize]
            batches.append(batch)
        
        return batches

    def split(
        self,
        waveform: torch.Tensor 
    ) -> List:
        chunks = self._split_to_chunks(waveform)
        batches = self._split_to_batches(chunks)
        return batches