from transcription.audio import Audio
import torchaudio

class AudioPreprocessor:
    TARGET_SAMPLING_RATE: int = 16000

    # for different models in future
    # def __init__(self, model):
    #     pass

    def _resample(
        self,
        audio: Audio
    ) -> None:
        if audio.sr != self.TARGET_SAMPLING_RATE:
            audio.waveform = torchaudio.functional.resample(
                audio.waveform, 
                audio.sr, 
                self.TARGET_SAMPLING_RATE
            )
    
    def _to_mono(
        self,
        audio: Audio
    ) -> None:
        if audio.waveform.shape[0] > 1:
            audio.waveform = audio.waveform.mean(dim=0, keepdim=True)
        audio.waveform = audio.waveform.squeeze(0)
    
    def prepare(
        self, 
        audio: Audio
    ):
        self._resample(audio)
        self._to_mono(audio)