from dataclasses import dataclass
import torchaudio
import torch

class Audio:
    waveform: torch.Tensor
    sr: int

    def load(self, filepath):
        """
        Loads audio from file's path
        """
        self.waveform, self.sr = torchaudio.load(
            filepath,
            backend="ffmpeg",
        )
        return self