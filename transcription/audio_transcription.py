import logging
import math
import time

import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transcription.device_configuration import DeviceConfiguration
from ui.ui_log_handler import UILogHandler


# TODO: implement transcription with shift
class AudioTranscription:
    model_name = "openai/whisper-large-v2"

    filepath: str
    waveform: torch.Tensor
    sampling_rate: int

    chunks: list = []
    batches: list = []
    chunk_size: int
    custom_chunk_length: int
    custom_batch_length: int

    device = "cuda"
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    logger: logging.Logger
    torch_dtype: torch.dtype

    language = "ru"

    all_transcription: list = []

    def __init__(
        self,
        filepath: str,
        device_configuration: DeviceConfiguration,
        logger: logging.Logger,
        language="ru",
    ) -> None:
        # TODO: add pretty docs here
        self.filepath = filepath
        self.language = language
        self.logger = logger

        # extracting configuration
        self.device = device_configuration.device
        self.model_name = device_configuration.model_name
        self.custom_chunk_length = device_configuration.chunk_length_s
        self.custom_batch_length = device_configuration.batch_size
        self.torch_dtype = device_configuration.torch_dtype

        self.chunks: list = []
        self.batches: list = []
        self.all_transcription: list = []
        try:
            logger.info("Loading model WhisperProcessor...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)

            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype
            ).to(self.device)

            logger.info("Model loaded.")

            self.waveform, self.sampling_rate = torchaudio.load(
                filepath, format="mp3", backend="ffmpeg"
            )
            logger.info(f"Successfully loaded file {filepath}.")

        except Exception as e:
            logger.error(f"Unable to load file {self.filepath}: {e}")
            raise

    def _resample(self) -> None:
        self.waveform = torchaudio.functional.resample(
            self.waveform, self.sampling_rate, 16000
        )

    def _to_mono(self):
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0, keepdim=True)
        self.waveform = self.waveform.squeeze(0)

    def _split_to_chunks(self, chunk_length_s: int = 30) -> None:
        self.logger.info(f"Splitting audio on chunks...")

        self.chunk_size = chunk_length_s * 16000  # 16kHz after resampling
        total_samples = self.waveform.shape[0]
        chunks_count = (total_samples + self.chunk_size - 1) // self.chunk_size

        self.logger.info(
            f"File length - {total_samples / 16000:.1f} seconds, splitting on {chunks_count} chunks by {chunk_length_s} seconds per chunk."
        )

        self.chunks = []
        for idx in tqdm(range(chunks_count)):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, total_samples)
            chunk = self.waveform[start:end].cpu().numpy().astype("float32")
            self.chunks.append(chunk)

    def _resplit_to_batches(self) -> None:
        self.logger.info(f"Splitting chunks into batches...")
        self.batches = []
        for i in range(0, len(self.chunks), self.custom_batch_length):
            batch = self.chunks[i : i + self.custom_batch_length]
            self.batches.append(batch)

    def _process_all_batches(self) -> None:
        start_time = time.time()
        try:
            self.all_transcription = []

            for idx in tqdm(range(len(self.batches))):
                inputs = self.processor(
                    self.batches[idx],
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )

                input_features = inputs.input_features.to(self.device).to(
                    self.torch_dtype
                )

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        language=self.language,
                        task="transcribe",
                        temperature=0.0,
                    )

                texts = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )
                self.all_transcription.extend(texts)
            end_time = time.time()
            self.logger.info(
                f"Transcription completed in {end_time - start_time:.2f} seconds"
            )

        except Exception as e:
            self.logger.error(f"Errors occured while processing chunks: {e}")

    def transcribe_audio(self) -> str:
        # TODO: maybe something else, not str?
        self._resample()
        self._to_mono()
        self._split_to_chunks()
        self._resplit_to_batches()
        self._process_all_batches()
        return " ".join(self.all_transcription)
