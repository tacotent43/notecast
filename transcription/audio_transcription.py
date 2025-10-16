import logging
import math
import time
import sys
import gc

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
    file_format: str

    chunks: list = []
    batches: list = []
    chunk_size: int
    custom_chunk_length: int
    custom_batch_length: int

    device = "cuda"
    processor: WhisperProcessor | None
    model: WhisperForConditionalGeneration | None
    logger: logging.Logger
    torch_dtype: torch.dtype

    language = "ru"

    all_transcription: list = []

    def __init__(
        self,
        filepath: str,
        device_configuration: DeviceConfiguration,
        logger: logging.Logger,
        language: str = "ru",
    ) -> None:
        # TODO: add pretty docs here
        self.filepath = filepath
        self.language = language
        self.logger = logger

        # setting file extension
        self.file_format = filepath.split(".")[-1]

        # extracting configuration
        self.device = device_configuration.device
        self.model_name = device_configuration.model_name
        self.custom_chunk_length = device_configuration.chunk_length_s
        self.custom_batch_length = device_configuration.batch_size
        self.torch_dtype = device_configuration.torch_dtype

        self.chunks: list = []
        self.batches: list = []
        self.all_transcription: list = []

    def _load_model(self) -> None:
        self.logger.info("Loading model WhisperProcessor...")
        try:
            start_time = time.time()

            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype
            ).to(self.device)

            end_time = time.time()
            self.logger.info(
                f"Model loaded successfully in {end_time - start_time:.2f} seconds."
            )
        except Exception as e:
            self.logger.error(f"Error while loading model: {e}")

    def _unload_model(self) -> None:
        self.logger.info("Unloading model...")
        self.model = None
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # TODO: maybe do something here for MPS
        self.logger.info("Model unloaded successfully.")

    def _load_file(self) -> None:
        self.logger.info(f"Loading file {self.filepath}")
        try:
            self.waveform, self.sampling_rate = torchaudio.load(
                self.filepath, format=self.file_format, backend="ffmpeg"
            )
            self.logger.info(f"Successfully loaded file {self.filepath}.")
        except Exception as e:
            self.logger.error(f"Unable to load file {self.filepath}: {e}")

    def _resample(self) -> None:
        self.waveform = torchaudio.functional.resample(
            self.waveform, self.sampling_rate, 16000
        )

    def _to_mono(self):
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0, keepdim=True)
        self.waveform = self.waveform.squeeze(0)

    def _split_to_chunks(self, shift: bool = False) -> None:
        self.logger.info(f"Splitting audio on chunks...")

        self.chunk_size = self.custom_chunk_length * 16000  # 16kHz after resampling
        total_samples = self.waveform.shape[0]
        chunks_count = (total_samples + self.chunk_size - 1) // self.chunk_size

        self.logger.info(
            f"File length - {total_samples / 16000:.1f} seconds, splitting on {chunks_count} chunks by {self.custom_chunk_length} seconds per chunk."
        )

        self.chunks = []
        for idx in tqdm(range(chunks_count)):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, total_samples)
            chunk = self.waveform[start:end].cpu().numpy().astype("float32")
            self.chunks.append(chunk)

    def _resplit_chunks_to_batches(self) -> None:
        self.logger.info(f"Splitting chunks into batches...")
        self.batches = []
        for i in range(0, len(self.chunks), self.custom_batch_length):
            batch = self.chunks[i : i + self.custom_batch_length]
            self.batches.append(batch)
        self.logger.info(f"Total: {len(self.batches)} batches, weight = {sys.getsizeof(self.batches)}")

    def _process_all_batches(self) -> None:
        start_time = time.time()
        try:
            assert self.processor is not None
            assert self.model is not None

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
                
                inputs = None
                input_features = None
                predicted_ids = None
                gc.collect()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            end_time = time.time()
            self.logger.info(
                f"Transcription completed in {end_time - start_time:.2f} seconds"
            )

        except Exception as e:
            self.logger.error(f"Errors occured while processing chunks: {e}")

    def transcribe_audio(self) -> str:
        self._load_model()
        self._load_file()
        self._resample()
        self._to_mono()
        self._split_to_chunks()
        self._resplit_chunks_to_batches()
        self._process_all_batches()
        self._unload_model()
        return " ".join(self.all_transcription)
