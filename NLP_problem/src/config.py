"""
Centralized configuration for Wav2Vec2 CTC Icelandic ASR training.
All hardware-critical settings are defined here to prevent OOM on 8GB VRAM.

CRITICAL: DO NOT modify fp16, batch_size, or max_audio_duration without
understanding the VRAM implications. These are tuned for RTX 4060 8GB.
"""

from dataclasses import dataclass, field
from typing import Dict


# Hardcoded Icelandic vocabulary - DO NOT build from dataset (will hang)
VOCAB_DICT: Dict[str, int] = {
    "[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "á": 4, "b": 5, "d": 6, "ð": 7,
    "e": 8, "é": 9, "f": 10, "g": 11, "h": 12, "i": 13, "í": 14, "j": 15,
    "k": 16, "l": 17, "m": 18, "n": 19, "o": 20, "ó": 21, "p": 22, "r": 23,
    "s": 24, "t": 25, "u": 26, "ú": 27, "v": 28, "x": 29, "y": 30, "ý": 31,
    "þ": 32, "æ": 33, "ö": 34
}

VOCAB_SIZE = len(VOCAB_DICT)  # 35


@dataclass
class TrainingConfig:
    """
    Training configuration with hardware-safe defaults for 8GB VRAM.

    VRAM Budget (RTX 4060 8GB):
    - Model (FP16): ~0.6 GB
    - Optimizer states: ~1.2 GB
    - Gradients: ~0.6 GB
    - Activations (batch=2, 10s): ~2-3 GB with gradient checkpointing
    - CUDA overhead: ~0.5 GB
    - Total: ~6-7 GB (SAFE)
    """

    # Model
    base_model: str = "facebook/wav2vec2-large-xlsr-53"
    vocab_size: int = VOCAB_SIZE

    # Hardware constraints (8GB VRAM safe) - DO NOT MODIFY
    fp16: bool = True  # MANDATORY for 8GB VRAM
    batch_size: int = 2  # Max safe batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    gradient_checkpointing: bool = True  # Reduces activation memory

    # Data constraints
    max_audio_duration: float = 10.0  # CRITICAL: Longer audio = OOM
    sample_rate: int = 16000
    shuffle_buffer_size: int = 1000

    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 5000

    # CTC configuration
    ctc_loss_reduction: str = "mean"  # Use mean, not sum
    ctc_zero_infinity: bool = True  # Prevent NaN losses

    # Logging and checkpointing
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    eval_samples: int = 200

    # Output
    output_dir: str = "./output"

    # Reproducibility
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """Total samples per optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class BenchmarkConfig:
    """Configuration for the 100-step benchmark smoke test."""

    # Inherit hardware constraints from TrainingConfig
    fp16: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True

    # Benchmark-specific
    benchmark_steps: int = 100
    warmup_steps: int = 10  # Shorter warmup for benchmark

    # Data
    max_audio_duration: float = 10.0
    sample_rate: int = 16000


def get_training_config(**overrides) -> TrainingConfig:
    """
    Get training config with optional overrides.

    Safety: Prevents dangerous overrides for 8GB VRAM.
    """
    config = TrainingConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
            # Safety checks for critical parameters
            if key == "batch_size" and value > 2:
                raise ValueError(
                    f"batch_size={value} is unsafe for 8GB VRAM. Max is 2."
                )
            if key == "fp16" and value is False:
                raise ValueError(
                    "fp16=False will cause OOM on 8GB VRAM. Keep fp16=True."
                )
            if key == "max_audio_duration" and value > 15.0:
                raise ValueError(
                    f"max_audio_duration={value}s is risky. Max recommended is 10-15s."
                )
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return config


def get_benchmark_config(**overrides) -> BenchmarkConfig:
    """Get benchmark config with optional overrides."""
    config = BenchmarkConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
