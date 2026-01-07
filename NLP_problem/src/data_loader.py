"""
Streaming data loader for Icelandic ASR with Wav2Vec2 CTC.

CRITICAL CONSTRAINTS:
1. Uses streaming=True to handle ~1M samples without RAM overflow
2. Uses hardcoded vocabulary - NEVER iterate dataset to build vocab
3. Filters audio > 10s to prevent OOM during CTC training
4. Never uses len(dataset) - not available in streaming mode
"""

import re
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

import torch
import librosa
from datasets import load_dataset, IterableDataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from config import VOCAB_DICT, VOCAB_SIZE


# Characters allowed in transcriptions (excluding special tokens)
ALLOWED_CHARS = set(VOCAB_DICT.keys()) - {"[PAD]", "[UNK]"}
ALLOWED_CHARS_PATTERN = re.compile(f"[^{''.join(re.escape(c) for c in ALLOWED_CHARS)}]")


def create_tokenizer(vocab_dir: str = "./vocab") -> Wav2Vec2CTCTokenizer:
    """
    Create Wav2Vec2CTCTokenizer from hardcoded vocabulary.

    DO NOT attempt to build vocabulary from dataset - will hang on 1M samples.
    """
    vocab_path = Path(vocab_dir)
    vocab_path.mkdir(parents=True, exist_ok=True)
    vocab_file = vocab_path / "vocab.json"

    # Write hardcoded vocab to file
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(VOCAB_DICT, f, ensure_ascii=False, indent=2)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    return tokenizer


def create_feature_extractor(sample_rate: int = 16000) -> Wav2Vec2FeatureExtractor:
    """Create Wav2Vec2 feature extractor."""
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )


def create_processor(vocab_dir: str = "./vocab", sample_rate: int = 16000) -> Wav2Vec2Processor:
    """Create the full Wav2Vec2Processor (tokenizer + feature extractor)."""
    tokenizer = create_tokenizer(vocab_dir)
    feature_extractor = create_feature_extractor(sample_rate)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def normalize_text(text: str) -> str:
    """
    Normalize transcription text:
    - Lowercase
    - Replace spaces with word delimiter |
    - Remove characters not in vocabulary
    """
    text = text.lower().strip()
    text = text.replace(" ", "|")
    text = ALLOWED_CHARS_PATTERN.sub("", text)
    text = re.sub(r"\|+", "|", text)  # Remove consecutive delimiters
    text = text.strip("|")
    return text


# The dataset has splits by gender/age, not train/test
# We map "train" to most splits, "test" to "other"
TRAIN_SPLITS = [
    "female_lt_18_yrs",
    "female_18to49_yrs",
    "female_gt_49_yrs",
    "male_lt_18_yrs",
    "male_18to49_yrs",
    "male_gt_49_yrs",
]
TEST_SPLITS = ["other"]


def load_streaming_dataset(
    split: str = "train",
    trust_remote_code: bool = True
) -> IterableDataset:
    """
    Load Samromur dataset in STREAMING mode.

    CRITICAL: Always use streaming=True for this 1M sample dataset.
    Never attempt to load into memory or iterate for vocab building.

    The dataset splits are by gender/age:
    - train: female_*, male_* splits (most data)
    - test: "other" split
    """
    from datasets import interleave_datasets

    if split == "train":
        config_splits = TRAIN_SPLITS
    else:  # test
        config_splits = TEST_SPLITS

    datasets = []
    for config_split in config_splits:
        ds = load_dataset(
            "language-and-voice-lab/samromur_milljon",
            split=config_split,
            streaming=True,
            trust_remote_code=trust_remote_code
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    # Interleave all datasets for balanced sampling
    return interleave_datasets(datasets)


def preprocess_sample(
    sample: Dict[str, Any],
    processor: Wav2Vec2Processor,
    max_duration_sec: float = 10.0,
    target_sample_rate: int = 16000
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Preprocess a single sample:
    - Check duration (filter > max_duration_sec)
    - Resample audio to 16kHz
    - Normalize text
    - Extract features and encode labels

    Returns None if sample should be filtered out.
    """
    try:
        audio = sample["audio"]
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]

        # Filter by duration BEFORE resampling (faster)
        duration = len(audio_array) / sample_rate
        if duration > max_duration_sec:
            return None

        # Resample if needed
        if sample_rate != target_sample_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=target_sample_rate
            )

        # Get and normalize text
        text = sample.get("normalized_text") or sample.get("sentence", "")
        normalized_text = normalize_text(text)

        # Skip empty transcriptions
        if not normalized_text:
            return None

        # Extract audio features using feature extractor directly
        inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=False
        )

        # Encode text labels using tokenizer directly
        labels = processor.tokenizer(
            normalized_text,
            return_tensors="pt"
        ).input_ids

        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0) if inputs.attention_mask is not None else None,
            "labels": labels.squeeze(0),
        }

    except Exception as e:
        # Skip problematic samples silently
        return None


def collate_batch(
    batch: list,
    pad_token_id: int = VOCAB_DICT["[PAD]"]
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples with dynamic padding.

    - Pads input_values with 0.0
    - Pads labels with -100 (ignored by CTC loss)
    """
    # Pad input values
    input_values = [item["input_values"] for item in batch]
    max_input_len = max(v.shape[0] for v in input_values)

    padded_inputs = torch.zeros(len(batch), max_input_len)
    attention_mask = torch.zeros(len(batch), max_input_len, dtype=torch.long)

    for i, v in enumerate(input_values):
        padded_inputs[i, :v.shape[0]] = v
        attention_mask[i, :v.shape[0]] = 1

    # Pad labels with -100
    labels = [item["labels"] for item in batch]
    max_label_len = max(l.shape[0] for l in labels)

    padded_labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l

    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


class StreamingASRDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset for streaming HuggingFace dataset.

    Handles preprocessing on-the-fly and filters long audio samples.
    """

    def __init__(
        self,
        split: str,
        processor: Wav2Vec2Processor,
        max_duration_sec: float = 10.0,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        max_samples: Optional[int] = None  # Use .take(n) for testing
    ):
        self.split = split
        self.processor = processor
        self.max_duration_sec = max_duration_sec
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = load_streaming_dataset(self.split)

        # Shuffle for training
        if self.split == "train":
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=self.seed
            )

        # Limit samples if specified (useful for testing)
        if self.max_samples is not None:
            dataset = dataset.take(self.max_samples)

        for sample in dataset:
            processed = preprocess_sample(
                sample,
                self.processor,
                self.max_duration_sec
            )
            if processed is not None:
                yield processed


def get_data_collator():
    """Returns a data collator function for use with DataLoader."""
    def collator(batch):
        return collate_batch(batch)
    return collator


def create_dataloader(
    split: str,
    processor: Wav2Vec2Processor,
    batch_size: int = 2,
    max_duration_sec: float = 10.0,
    shuffle_buffer_size: int = 1000,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for streaming ASR data.

    Args:
        split: "train" or "test"
        processor: Wav2Vec2Processor instance
        batch_size: Batch size (keep at 2 for 8GB VRAM)
        max_duration_sec: Filter audio longer than this
        shuffle_buffer_size: Buffer size for shuffling
        seed: Random seed
        max_samples: Limit total samples (for testing/benchmark)

    Returns:
        DataLoader that streams data on-the-fly
    """
    dataset = StreamingASRDataset(
        split=split,
        processor=processor,
        max_duration_sec=max_duration_sec,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        max_samples=max_samples
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_data_collator(),
        num_workers=0  # Streaming doesn't support multi-worker well
    )


if __name__ == "__main__":
    # Quick test of data pipeline
    print("Creating processor...")
    processor = create_processor()
    print(f"Vocabulary size: {VOCAB_SIZE}")

    print("\nLoading streaming dataset (first 10 samples)...")
    dataloader = create_dataloader(
        split="train",
        processor=processor,
        batch_size=2,
        max_samples=10  # Limit for testing
    )

    print("\nProcessing batches...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  input_values shape: {batch['input_values'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        if i >= 2:
            break

    print("\nData pipeline test complete!")
