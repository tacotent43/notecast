from transcription.audio_transcription import AudioTranscription
from transcription.torch_checker import check_torch
from utils.logger import setup_logger

logger = setup_logger("main")

check_torch()

try:
    track = AudioTranscription("sample.mp3")
    print(track.transcribe_audio())
except Exception as e:
    logger.error(f"Execution error: {e}") 