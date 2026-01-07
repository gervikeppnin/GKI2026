#!/usr/bin/env python3
"""
Compare our Wav2Vec2 baseline against Whisper Large Icelandic.
Uses the same test set for fair comparison.
"""

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from datasets import load_dataset, interleave_datasets
import evaluate
from tqdm import tqdm
import unicodedata

# Config
WAV2VEC_MODEL = "./output/best_model"
WAV2VEC_PROCESSOR = "./output/processor"
WHISPER_MODEL = "language-and-voice-lab/whisper-large-icelandic-62640-steps-967h"
TEST_SPLITS = ["other"]
NUM_SAMPLES = 500
MAX_DURATION = 10.0


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    # Remove punctuation except apostrophe
    text = ''.join(c for c in text if c.isalnum() or c.isspace() or c == "'")
    text = ' '.join(text.split())
    return text


def load_test_data():
    """Load test dataset."""
    print("Loading test dataset...")

    # Load the "other" split as test set (same as training eval)
    ds = load_dataset(
        "language-and-voice-lab/samromur_milljon",
        split="other",
        streaming=True,
        trust_remote_code=True
    )

    combined = ds

    # Filter and collect samples
    samples = []
    for sample in tqdm(combined, desc="Loading samples", total=NUM_SAMPLES * 2):
        audio = sample["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration <= MAX_DURATION and sample.get("normalized_text"):
            samples.append({
                "audio": audio["array"],
                "sampling_rate": audio["sampling_rate"],
                "text": sample["normalized_text"]
            })
            if len(samples) >= NUM_SAMPLES:
                break

    print(f"Loaded {len(samples)} test samples")
    return samples


def evaluate_wav2vec(samples, device):
    """Evaluate our Wav2Vec2 model."""
    print("\n" + "="*60)
    print("Evaluating Wav2Vec2 (our baseline)")
    print("="*60)

    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL).to(device)
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_PROCESSOR)
    model.eval()

    predictions = []
    references = []

    for sample in tqdm(samples, desc="Wav2Vec2"):
        audio = sample["audio"]

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0]
        pred_text = pred_text.replace("|", " ").strip()

        predictions.append(normalize_text(pred_text))
        references.append(normalize_text(sample["text"]))

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return predictions, references


def evaluate_whisper(samples, device):
    """Evaluate Whisper Large Icelandic."""
    print("\n" + "="*60)
    print("Evaluating Whisper Large Icelandic")
    print("="*60)

    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model.eval()

    # Force Icelandic
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="is", task="transcribe")

    predictions = []
    references = []

    for sample in tqdm(samples, desc="Whisper"):
        audio = sample["audio"]

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225
            )

        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(normalize_text(pred_text))
        references.append(normalize_text(sample["text"]))

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return predictions, references


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Load test data
    samples = load_test_data()

    # Evaluate Wav2Vec2
    wav2vec_preds, wav2vec_refs = evaluate_wav2vec(samples, device)
    wav2vec_wer = wer_metric.compute(predictions=wav2vec_preds, references=wav2vec_refs)
    wav2vec_cer = cer_metric.compute(predictions=wav2vec_preds, references=wav2vec_refs)

    # Evaluate Whisper
    whisper_preds, whisper_refs = evaluate_whisper(samples, device)
    whisper_wer = wer_metric.compute(predictions=whisper_preds, references=whisper_refs)
    whisper_cer = cer_metric.compute(predictions=whisper_preds, references=whisper_refs)

    # Results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Test samples: {len(samples)}")
    print()
    print(f"{'Model':<45} {'WER':>10} {'CER':>10}")
    print("-"*65)
    print(f"{'Wav2Vec2-XLSR-53 (our baseline)':<45} {wav2vec_wer*100:>9.2f}% {wav2vec_cer*100:>9.2f}%")
    print(f"{'Whisper Large Icelandic (967h)':<45} {whisper_wer*100:>9.2f}% {whisper_cer*100:>9.2f}%")
    print("-"*65)

    # Sample comparisons
    print("\nSample Comparisons:")
    print("-"*65)
    for i in range(min(5, len(samples))):
        print(f"\n[{i+1}] Reference: {whisper_refs[i]}")
        print(f"    Wav2Vec2:  {wav2vec_preds[i]}")
        print(f"    Whisper:   {whisper_preds[i]}")


if __name__ == "__main__":
    main()
