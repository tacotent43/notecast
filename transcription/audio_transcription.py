from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from utils.logger import setup_logger
import time
import math
from tqdm import tqdm

logger = setup_logger("AudioTranscribe module")

class AudioTranscription:
    model_name = "openai/whisper-large-v2" 
    
    filepath: str
    waveform: torch.Tensor
    sampling_rate: int
    
    chunks: list = []
    chunk_size: int 
    device = "cuda"
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    
    language = "ru"
    
    all_transcription: list = []
    
    def __init__(
        self, 
        filepath: str,
        language = "ru",
        device = "cuda",
        model_name = "openai/whisper-large-v2"
    ) -> None:
        self.filepath = filepath
        self.language = language
        self.device = device
        self.model_name = model_name
        self.chunks: list = []
        try:
            logger.info("Loading model WhisperProcessor...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
            
            logger.info("Model loaded.")
            
            self.waveform, self.sampling_rate = torchaudio.load(filepath, format="mp3", backend="ffmpeg")
            logger.info(f"Successfully loaded file {filepath}.")
        
        except Exception as e:
            logger.error(f"Unable to load file {self.filepath}: {e}")
            raise
    
    def resample(self) -> None:
        self.waveform = torchaudio.functional.resample(self.waveform, self.sampling_rate, 16000)
    
    def to_mono(self):
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0, keepdim=True)
        self.waveform = self.waveform.squeeze(0)
    
    def split_to_chunks(self, chunk_length_s: int = 30) -> None:
        logger.info(f"Splitting audio on chunks...")
        
        self.chunk_size = chunk_length_s * 16000  # 16kHz after resampling
        total_samples = self.waveform.shape[0]
        chunks_count = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"File length - {total_samples / 16000:.1f} seconds, splitting on {chunks_count} chunks by {chunk_length_s} seconds per chunk.")

        self.chunks = []
        for idx in range(chunks_count):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, total_samples)
            chunk = self.waveform[start:end].cpu().numpy().astype("float32")
            self.chunks.append(chunk)
    
    def process_chunk(
        self, 
        chunk
    ) -> str:
        inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device).to(torch.float16)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task="transcribe",
                temperature=0.0
            )
        
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return text
    
    def process_all_chunks(self, batch_size: int = 16) -> None:
        start_time = time.time()
        try:
            self.all_transcription = []

            for i in tqdm(range(math.ceil(len(self.chunks) / batch_size))):
                # TODO: rewrite batching as a separate function
                batch = self.chunks[i*batch_size:(i+1)*batch_size]

                inputs = self.processor(
                    batch, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )

                input_features = inputs.input_features.to(self.device).to(torch.float16)

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        language=self.language,
                        task="transcribe",
                        temperature=0.0
                    )

                texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                self.all_transcription.extend(texts)

            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Errors occured while processing chunks: {e}")

    
    def transcribe_audio(self) -> str:
        self.resample()
        self.to_mono()
        self.split_to_chunks()
        self.process_all_chunks()
        return " ".join(self.all_transcription)