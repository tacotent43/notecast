from transcription.audio import Audio
from transcription.preprocessing.audio_preprocessor import AudioPreprocessor
from transcription.preprocessing.splitter import Splitter
from transcription.engines.whisper import WhisperEngine
from transcription.configuration import Configuration

# maybe inherit from AudioTranscription and rename to something like WhisperTranscription?
class AudioTranscription:
    # add multimodel ability
    def __init__(
        self,
        filepath: str,
        config: Configuration,
        language,
        # logger
    ) -> None:
        self.filepath = filepath
        self.language = language
        # self.logger = logger

        self.audio = Audio()
        self.preprocessor = AudioPreprocessor()
        self.splitter = Splitter(
            chunkSize=config.chunkSize,
            batchSize=config.batchSize,
        )
        self.engine = WhisperEngine(
            modelName=config.modelName,
            language=self.language,
            dType=config.dType,
            device=config.device,
        )
    
    # maybe add something like temperature here?
    def transcribeAudio(self) -> str:
        transcription: list = []
        
        self.engine.loadModel()
        
        self.preprocessor.prepare(self.audio.load(self.filepath))

        batches = self.splitter.split(self.audio.waveform)

        for batch in batches:
            batchText: str = self.engine.transcribeBatch(batch)
            transcription.append(batchText)

        self.engine.unloadModel()
        
        return str(" ".join(transcription))
