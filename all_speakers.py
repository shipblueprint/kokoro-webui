import os
import onnxruntime as ort
from kokoro_onnx import Kokoro
import soundfile as sf

# Set environment variables for GPU use
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_CUDA_GPUREUSE_ENABLE"] = "1"
os.environ["ORT_EP"] = "CUDAExecutionProvider"

# Suppress ONNX Runtime logs
ort.set_default_logger_severity(3)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
kokoro_model_path = os.path.join(base_dir, "models", "kokoro-v1.0.onnx")
voices_model_path = os.path.join(base_dir, "models", "voices-v1.0.bin")
text_file_path = os.path.join(base_dir, "narration.txt")
output_dir = os.path.join(base_dir, "output")

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the Kokoro model
kokoro = Kokoro(kokoro_model_path, voices_model_path)

# Check for GPU availability
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    print("CUDAExecutionProvider is available! Running on GPU.")
else:
    print("CUDAExecutionProvider is NOT available! Running on CPU.")

# Read narration text from file
if not os.path.exists(text_file_path):
    raise FileNotFoundError(f"Text file not found: {text_file_path}")

with open(text_file_path, "r", encoding="utf-8") as f:
    narration_text = f.read().strip()

if not narration_text:
    raise ValueError("The narration.txt file is empty.")

# Generate audio for each voice
for voice in kokoro.get_voices():
    samples, sample_rate = kokoro.create(
        narration_text,
        voice=voice,
        speed=0.85
    )
    filename = f"{voice.replace(' ', '_')}_pyramid.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, samples, sample_rate)
    print(f"Created {filepath}")
