"""
Benchmark script for Wav2Vec2 CTC Icelandic ASR.

Runs a 100-step smoke test to:
1. Validate VRAM fits within 8GB
2. Measure throughput (samples/second)
3. Estimate full training time

Run BEFORE full training to catch OOM issues early.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import VOCAB_DICT, VOCAB_SIZE, get_benchmark_config, TrainingConfig
from data_loader import create_processor, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Wav2Vec2 CTC training")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 (default: True)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--target_steps", type=int, default=5000, help="Target steps for time estimate")
    return parser.parse_args()


def create_model(base_model: str = "facebook/wav2vec2-large-xlsr-53") -> Wav2Vec2ForCTC:
    """Create Wav2Vec2ForCTC model with CTC configuration."""
    config = Wav2Vec2Config.from_pretrained(base_model)
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = VOCAB_DICT["[PAD]"]
    config.ctc_loss_reduction = "mean"
    config.ctc_zero_infinity = True
    config.gradient_checkpointing = True

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

    model.gradient_checkpointing_enable()

    return model


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def format_memory(bytes_val: float) -> str:
    """Format bytes into GB."""
    return f"{bytes_val / 1e9:.2f} GB"


def run_benchmark(args):
    """Run the benchmark and report statistics."""

    print("=" * 60)
    print("WAV2VEC2 CTC BENCHMARK")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Running on CPU (very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"Total VRAM: {gpu_memory:.1f} GB")

    print(f"\nBenchmark Configuration:")
    print(f"  FP16: {args.fp16}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Benchmark steps: {args.steps}")
    print(f"  Target steps for estimate: {args.target_steps}")

    # Create processor
    print("\nCreating processor...")
    processor = create_processor(vocab_dir="./benchmark_vocab")

    # Create model
    print("Loading model...")
    model = create_model()
    model.to(device)

    # Enable FP16 if requested
    scaler = None
    if args.fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("FP16 mixed precision enabled")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Create dataloader
    print("\nCreating streaming dataloader...")
    dataloader = create_dataloader(
        split="train",
        processor=processor,
        batch_size=args.batch_size,
        max_duration_sec=10.0,
        max_samples=args.steps * args.batch_size * 2  # Buffer for filtered samples
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )

    # Reset CUDA memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark loop
    print(f"\nRunning {args.steps} benchmark steps...")
    print("-" * 60)

    model.train()
    step_times = []
    total_samples = 0
    running_loss = 0.0

    start_time = time.time()
    progress_bar = tqdm(total=args.steps, desc="Benchmarking")

    step = 0
    for batch in dataloader:
        if step >= args.steps:
            break

        step_start = time.time()

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if args.fp16 and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)
        running_loss += loss.item()
        total_samples += batch["input_values"].shape[0]

        step += 1
        progress_bar.update(1)

        # Log every 20 steps
        if step % 20 == 0:
            avg_loss = running_loss / step
            if device.type == "cuda":
                current_vram = torch.cuda.memory_allocated() / 1e9
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.3f}",
                    "VRAM": f"{current_vram:.1f}/{peak_vram:.1f}GB"
                })

    progress_bar.close()
    total_time = time.time() - start_time

    # Collect statistics
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated()
        current_vram = torch.cuda.memory_allocated()
    else:
        peak_vram = 0
        current_vram = 0

    avg_step_time = sum(step_times) / len(step_times)
    samples_per_second = total_samples / total_time
    estimated_full_time = avg_step_time * args.target_steps

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nTiming:")
    print(f"  Total benchmark time: {format_time(total_time)}")
    print(f"  Average step time: {avg_step_time*1000:.1f} ms")
    print(f"  Throughput: {samples_per_second:.2f} samples/second")

    if device.type == "cuda":
        print(f"\nVRAM Usage:")
        print(f"  Peak VRAM: {format_memory(peak_vram)}")
        print(f"  Current VRAM: {format_memory(current_vram)}")

        # Safety check
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        vram_usage_pct = peak_vram / gpu_total * 100
        print(f"  Usage: {vram_usage_pct:.1f}% of available VRAM")

        if vram_usage_pct > 90:
            print("\n  ⚠️  WARNING: VRAM usage > 90%. OOM risk during training!")
            print("     Consider reducing batch_size or max_audio_duration.")
        elif vram_usage_pct > 80:
            print("\n  ⚠️  CAUTION: VRAM usage > 80%. Monitor during training.")
        else:
            print("\n  ✓  VRAM usage looks safe for training.")

    print(f"\nTraining:")
    print(f"  Average loss: {running_loss/args.steps:.4f}")
    print(f"  Samples processed: {total_samples}")

    print(f"\nTime Estimates:")
    print(f"  {args.target_steps} steps: {format_time(estimated_full_time)}")
    print(f"  10,000 steps: {format_time(avg_step_time * 10000)}")
    print(f"  50,000 steps: {format_time(avg_step_time * 50000)}")

    print("\n" + "=" * 60)

    # Return results for programmatic use
    return {
        "peak_vram_gb": peak_vram / 1e9,
        "samples_per_second": samples_per_second,
        "avg_step_time_ms": avg_step_time * 1000,
        "estimated_time_hours": estimated_full_time / 3600,
        "avg_loss": running_loss / args.steps,
    }


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmark(args)
