import logging
from dataclasses import dataclass

import torch


def check_torch(logger: logging.Logger) -> None:
    logger.info("=== Checking PyTorch ===")
    logger.info(f"Torch version: {torch.__version__}")

    # NVIDIA / AMD (CUDA API)
    if torch.cuda.is_available():
        backend = "CUDA"
        if torch.version.hip is not None:
            backend = "ROCm (AMD HIP)"

        logger.info(f"{backend} backend is available")
        logger.info(
            f"Compiled with: CUDA {torch.version.cuda}, ROCm {torch.version.hip}"
        )
        logger.info(f"Number of devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Apple Silicon (MPS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS backend is available (Apple Silicon)")
        logger.info(
            f"MPS version: {getattr(torch.backends.mps, '__version__', 'unknown')}"
        )
        logger.info("GPU: Apple Silicon (Metal)")

    # CPU only mode
    else:
        logger.info("Only CPU is available")

    logger.info("=== Check completed ===")
