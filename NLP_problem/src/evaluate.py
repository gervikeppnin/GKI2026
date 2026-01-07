"""
Evaluation script for Wav2Vec2 CTC Icelandic ASR model.
Calculates Word Error Rate (WER) and Character Error Rate (CER).
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import VOCAB_DICT, VOCAB_SIZE
from data_loader import create_processor, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 CTC model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Path to processor (defaults to model_path/../processor)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save detailed predictions"
    )

    return parser.parse_args()


def decode_predictions(logits: torch.Tensor, processor: Wav2Vec2Processor) -> list:
    """Decode model logits to text using greedy decoding."""
    predicted_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # Replace word delimiter with space
    predictions = [pred.replace("|", " ").strip() for pred in predictions]
    return predictions


def decode_labels(labels: torch.Tensor, processor: Wav2Vec2Processor) -> list:
    """Decode label tensor to text."""
    decoded = []
    for label in labels:
        label_ids = [l.item() for l in label if l.item() != -100]
        text = processor.tokenizer.decode(label_ids)
        text = text.replace("|", " ").strip()
        decoded.append(text)
    return decoded


def calculate_metrics(predictions: list, references: list) -> dict:
    """Calculate WER and CER metrics."""
    # Filter empty pairs
    valid_pairs = [
        (pred, ref) for pred, ref in zip(predictions, references)
        if ref.strip()
    ]

    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0, "num_samples": 0}

    preds, refs = zip(*valid_pairs)

    word_error_rate = wer(list(refs), list(preds))
    char_error_rate = cer(list(refs), list(preds))

    return {
        "wer": word_error_rate,
        "cer": char_error_rate,
        "num_samples": len(valid_pairs)
    }


def evaluate(args):
    """Main evaluation function."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # Load processor
    if args.processor_path:
        processor_path = args.processor_path
    else:
        model_parent = Path(args.model_path).parent
        processor_path = model_parent / "processor"
        if not processor_path.exists():
            print("Processor not found, creating new one...")
            processor = create_processor()
        else:
            processor_path = str(processor_path)

    if isinstance(processor_path, (str, Path)):
        print(f"Loading processor from {processor_path}...")
        processor = Wav2Vec2Processor.from_pretrained(str(processor_path))

    print(f"\nEvaluating on {args.split} split...")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    # Create dataloader
    eval_dataloader = create_dataloader(
        split=args.split,
        processor=processor,
        batch_size=args.batch_size,
        max_samples=args.num_samples
    )

    # Run evaluation
    all_predictions = []
    all_references = []
    total_loss = 0.0
    num_batches = 0

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if args.fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            total_loss += outputs.loss.item()
            num_batches += 1

            predictions = decode_predictions(outputs.logits, processor)
            references = decode_labels(batch["labels"], processor)

            all_predictions.extend(predictions)
            all_references.extend(references)

            if len(all_predictions) >= args.num_samples:
                all_predictions = all_predictions[:args.num_samples]
                all_references = all_references[:args.num_samples]
                break

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_predictions, all_references)
    metrics["avg_loss"] = total_loss / max(num_batches, 1)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Average loss: {metrics['avg_loss']:.4f}")
    print(f"Word Error Rate (WER): {metrics['wer']:.2%}")
    print(f"Character Error Rate (CER): {metrics['cer']:.2%}")
    print("=" * 60)

    # Show samples
    print("\nSample predictions:")
    print("-" * 60)
    for i in range(min(10, len(all_predictions))):
        print(f"\n[{i+1}]")
        print(f"  REF: {all_references[i]}")
        print(f"  HYP: {all_predictions[i]}")

    # Save detailed results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Samples: {metrics['num_samples']}\n")
            f.write(f"WER: {metrics['wer']:.4f} ({metrics['wer']:.2%})\n")
            f.write(f"CER: {metrics['cer']:.4f} ({metrics['cer']:.2%})\n")
            f.write(f"Avg Loss: {metrics['avg_loss']:.4f}\n")
            f.write("=" * 60 + "\n\n")
            f.write("PREDICTIONS\n")
            f.write("-" * 60 + "\n")
            for i, (ref, pred) in enumerate(zip(all_references, all_predictions)):
                f.write(f"\n[{i+1}]\n")
                f.write(f"REF: {ref}\n")
                f.write(f"HYP: {pred}\n")

    return metrics


def quick_inference(
    model_path: str,
    processor_path: str,
    audio_path: str = None,
    audio_array=None,
    sample_rate: int = 16000,
    device: str = None
) -> str:
    """
    Quick inference for a single audio file or array.

    Args:
        model_path: Path to trained model
        processor_path: Path to processor
        audio_path: Path to audio file (optional)
        audio_array: Audio as numpy array (optional)
        sample_rate: Sample rate of input audio
        device: Device to use (auto-detected if None)

    Returns:
        Transcription string
    """
    import librosa

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model.to(device)
    model.eval()

    # Load audio if path provided
    if audio_path:
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)

    # Resample if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    # Process
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    transcription = transcription.replace("|", " ").strip()

    return transcription


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
