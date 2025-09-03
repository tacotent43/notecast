from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import logging
import time
import asyncio

FILENAME = "sample"
SAMPLING_FREQUENCY = 16000

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

device = "cuda"

logger.info("Loading model WhisperProcessor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
logger.info("Loading model WhisperForConditionalGeneration...")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)
logger.info("Model loaded")


def transcribe_long(path: str, language="ru", chunk_length_s: int = 30):
    logger.info(f"Starting transcription of long file: {path}")
    start_time = time.time()

    try:
        waveform, sr = torchaudio.load(path, format="mp3", backend="ffmpeg")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLING_FREQUENCY).squeeze()

        total_samples = waveform.shape[0]
        chunk_size = chunk_length_s * SAMPLING_FREQUENCY
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        logger.info(f"File length - {total_samples/SAMPLING_FREQUENCY:.1f} seconds, splitting on {num_chunks} chunks by {chunk_length_s} seconds")

        transcripts = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            chunk = waveform[start:end].cpu().numpy()

            inputs = processor(chunk, sampling_rate=SAMPLING_FREQUENCY, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(torch.float16)

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    language=language,
                    task="transcribe",
                    temperature=0.0
                )

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcripts.append(text)

            logger.info(f"Чанк {i+1}/{num_chunks} готов ({(end/SAMPLING_FREQUENCY):.1f} сек)")

        end_time = time.time()
        logger.info(f"Transcription completed - {end_time - start_time:.2f} seconds")

        return " ".join(transcripts)

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise


def split_into_chunks(
    filepath: str, 
    chunk_length_s: int = 30
) -> list:
    try:
        chunks = []
        waveform, sr = torchaudio.load(filepath, format="mp3", backend="ffmpeg")
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        logger.info(f"Started splitting file into chunks with length {chunk_length_s}")
        total_samples = waveform.shape[0]
        chunk_size = chunk_length_s
        
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        logger.info(f"File length - {total_samples/SAMPLING_FREQUENCY:.1f} seconds, разобьём на {num_chunks} чанков по {chunk_length_s} секунд")
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            chunk = waveform[start:end].cpu().numpy()
            
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error while splitting to chunks: {str(e)}")
        raise


chunks = split_into_chunks(f"{FILENAME}.mp3")

try:
    result = transcribe_long(f"{FILENAME}.mp3")
    print(result)
except Exception as e:
    logger.error(f"Execution error: {e}")