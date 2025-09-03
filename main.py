from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import logging
import time
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=== Checking PyTorch ===")
print(f"Torch version: {torch.version}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPU: {torch.cuda.device_count()}")
    print(f"Name of GPU: {torch.cuda.get_device_name(0)}")
print("=== Check completed ===")

class AudioTranscription:
    model_name = "openai/whisper-large-v2" 
    
    filepath: str
    waveform: torch.Tensor
    sampling_rate: int
    
    chunks: list = []
    # one chunk size in seconds
    chunk_size: int 
    device = "cuda"
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    # self.device here and all previous shit
    
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
    
    def process_all_chunks(self) -> None:
        start_time = time.time()
        
        try:
            for chunk_idx in tqdm(range(len(self.chunks))):
                self.all_transcription.append(self.process_chunk(self.chunks[chunk_idx]))
            
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

try:
    track = AudioTranscription("sample.mp3")
    print(track.transcribe_audio())
except Exception as e:
    logger.error(f"Execution error: {e}")