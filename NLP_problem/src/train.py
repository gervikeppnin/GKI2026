"""
Training script for Wav2Vec2 CTC on Icelandic ASR.

HARDWARE CONSTRAINTS (RTX 4060 8GB):
- MUST use --fp16 (mandatory)
- MUST use --batch_size 2 (max safe)
- MUST use --gradient_accumulation_steps 8 (effective batch = 16)
- DO NOT increase batch size or disable FP16

Features:
- Streaming data loading for 1M sample dataset
- VRAM monitoring with peak usage logging
- ETA (Estimated Time of Arrival) based on step speed
- TensorBoard logging
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import VOCAB_DICT, VOCAB_SIZE, get_training_config
from data_loader import create_processor, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 CTC for Icelandic ASR")

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53",
        help="Base pretrained model"
    )

    # Hardware (8GB VRAM safe defaults)
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 (MANDATORY for 8GB)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (max 2 for 8GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation")

    # Training
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Data
    parser.add_argument("--max_audio_duration", type=float, default=10.0, help="Max audio duration (seconds)")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000, help="Shuffle buffer size")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=200, help="Number of eval samples")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def create_model(base_model: str, gradient_checkpointing: bool = True) -> Wav2Vec2ForCTC:
    """Create Wav2Vec2ForCTC model with proper CTC configuration."""
    config = Wav2Vec2Config.from_pretrained(base_model)
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = VOCAB_DICT["[PAD]"]
    config.ctc_loss_reduction = "mean"  # CRITICAL: use mean, not sum
    config.ctc_zero_infinity = True  # Prevent NaN losses
    config.gradient_checkpointing = gradient_checkpointing

    model = Wav2Vec2ForCTC.from_pretrained(
        base_model,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Freeze feature extractor (standard XLSR practice)
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.wav2vec2.feature_projection.parameters():
        param.requires_grad = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def evaluate(
    model: Wav2Vec2ForCTC,
    processor,
    device: torch.device,
    num_samples: int = 200,
    fp16: bool = True
) -> dict:
    """Run evaluation on test set subset."""
    model.eval()

    eval_dataloader = create_dataloader(
        split="test",
        processor=processor,
        batch_size=4,
        max_samples=num_samples
    )

    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if fp16:
                with torch.amp.autocast("cuda"):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            total_loss += outputs.loss.item()
            num_batches += 1

            # Decode predictions
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predicted_ids.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

            if num_batches * 4 >= num_samples:
                break

    # Decode strings
    predictions_str = []
    labels_str = []
    for pred, label in zip(all_predictions, all_labels):
        pred_tokens = [p for p in pred if p not in [VOCAB_DICT["[PAD]"], -100]]
        pred_str = processor.tokenizer.decode(pred_tokens).replace("|", " ")
        predictions_str.append(pred_str)

        label_tokens = [l for l in label if l != -100]
        label_str = processor.tokenizer.decode(label_tokens).replace("|", " ")
        labels_str.append(label_str)

    model.train()

    return {
        "eval_loss": total_loss / max(num_batches, 1),
        "predictions": predictions_str,
        "labels": labels_str
    }


def train(args):
    """Main training loop with VRAM monitoring and ETA logging."""

    # Validate hardware constraints
    if args.batch_size > 2:
        print(f"WARNING: batch_size={args.batch_size} is risky for 8GB VRAM. Using 2.")
        args.batch_size = 2

    # Set seed
    torch.manual_seed(args.seed)

    # Setup device
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU (very slow).")
        device = torch.device("cpu")
        args.fp16 = False
    else:
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Create processor and save
    print("\nCreating processor...")
    processor = create_processor(vocab_dir=str(output_dir / "vocab"))
    processor.save_pretrained(str(output_dir / "processor"))

    # Create model
    print(f"Loading model from {args.base_model}...")
    model = create_model(args.base_model, gradient_checkpointing=True)
    model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.1f}%)")

    # Create dataloader
    print("Creating streaming dataloader...")
    train_dataloader = create_dataloader(
        split="train",
        processor=processor,
        batch_size=args.batch_size,
        max_duration_sec=args.max_audio_duration,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # Setup FP16
    scaler = None
    if args.fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("FP16 mixed precision enabled")

    # Print config
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"FP16: {args.fp16}")
    print(f"Max audio duration: {args.max_audio_duration}s")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Reset VRAM stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    best_eval_loss = float("inf")
    start_time = time.time()
    step_times = []

    progress_bar = tqdm(total=args.max_steps, desc="Training")

    accumulation_step = 0

    for batch in train_dataloader:
        if global_step >= args.max_steps:
            break

        step_start = time.time()

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if args.fp16 and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

        accumulation_step += 1
        running_loss += loss.item() * args.gradient_accumulation_steps

        # Optimizer step after accumulation
        if accumulation_step >= args.gradient_accumulation_steps:
            if args.fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            accumulation_step = 0
            global_step += 1
            progress_bar.update(1)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / args.logging_steps
                current_lr = scheduler.get_last_lr()[0]

                # Calculate ETA
                elapsed = time.time() - start_time
                avg_step_time = elapsed / global_step
                remaining_steps = args.max_steps - global_step
                eta_seconds = avg_step_time * remaining_steps

                # VRAM stats
                vram_str = ""
                if device.type == "cuda":
                    peak_vram = torch.cuda.max_memory_allocated() / 1e9
                    current_vram = torch.cuda.memory_allocated() / 1e9
                    vram_str = f" | VRAM: {current_vram:.1f}/{peak_vram:.1f}GB"
                    writer.add_scalar("memory/peak_vram_gb", peak_vram, global_step)
                    writer.add_scalar("memory/current_vram_gb", current_vram, global_step)

                # Log to TensorBoard
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("train/step_time_ms", avg_step_time * 1000, global_step)

                print(f"Step {global_step}: loss={avg_loss:.4f} | lr={current_lr:.2e} | ETA: {format_time(eta_seconds)}{vram_str}")
                running_loss = 0.0

            # Evaluation
            if global_step % args.eval_steps == 0:
                print(f"\nEvaluating at step {global_step}...")
                eval_results = evaluate(
                    model, processor, device,
                    num_samples=args.eval_samples,
                    fp16=args.fp16
                )

                writer.add_scalar("eval/loss", eval_results["eval_loss"], global_step)
                print(f"Eval loss: {eval_results['eval_loss']:.4f}")

                # Show sample predictions
                if eval_results["predictions"]:
                    print("\nSample predictions:")
                    for i in range(min(3, len(eval_results["predictions"]))):
                        print(f"  Pred: {eval_results['predictions'][i]}")
                        print(f"  True: {eval_results['labels'][i]}")
                        print()

                # Save best model
                if eval_results["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_results["eval_loss"]
                    print("New best model! Saving...")
                    model.save_pretrained(str(output_dir / "best_model"))

            # Save checkpoint
            if global_step % args.save_steps == 0:
                print(f"Saving checkpoint at step {global_step}...")
                model.save_pretrained(str(output_dir / f"checkpoint-{global_step}"))

    progress_bar.close()
    writer.close()

    # Final save
    print("\nSaving final model...")
    model.save_pretrained(str(output_dir / "final_model"))

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    if device.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
