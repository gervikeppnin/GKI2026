#!/usr/bin/env python3
"""
Quick transcription script for testing the trained model.

Usage:
    python transcribe.py <audio_file>
    python transcribe.py --record  # Record from microphone (3 seconds)
"""

import sys
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_PATH = "./output/best_model"
PROCESSOR_PATH = "./output/processor"


def transcribe(audio_path: str = None, audio_array=None, sample_rate: int = 16000) -> str:
    """Transcribe audio file or array."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)
    model.to(device)
    model.eval()

    # Load audio if path provided
    if audio_path:
        print(f"Loading: {audio_path}")
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)

    # Resample if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    print(f"Audio duration: {len(audio_array)/16000:.2f}s")

    # Process
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    transcription = transcription.replace("|", " ").strip()

    return transcription


def record_audio(duration: float = 3.0) -> tuple:
    """Record audio from microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice: pip install sounddevice")
        sys.exit(1)

    sample_rate = 16000
    print(f"Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete!")
    return audio.flatten(), sample_rate


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python transcribe.py <audio_file.wav>")
        print("  python transcribe.py --record")
        sys.exit(1)

    if sys.argv[1] == "--record":
        audio, sr = record_audio(3.0)
        result = transcribe(audio_array=audio, sample_rate=sr)
    else:
        result = transcribe(audio_path=sys.argv[1])

    print("\n" + "="*50)
    print("TRANSCRIPTION:")
    print("="*50)
    print(result)
    print("="*50)
