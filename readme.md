# Kokoro TTS WebUI (with Voice Blending & Custom Pauses)

An intuitive Gradio WebUI wrapper for the **Kokoro TTS** model. This project adds specialized features like custom sentence/paragraph pause durations and an easy voice blending ratio system.

## Credits & Credibility
This UI is built entirely on top of the excellent work by the creators of the Kokoro model. 
* **Core Model:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by Hexgrad.
* **Python API wrapper used:** [kokoro-onnx](https://github.com/theodorewoll/kokoro-onnx)

## Setup Instructions
1. Install requirements: `pip install gradio kokoro-onnx numpy soundfile`
2. Create a `models/` directory and download the `kokoro-v1.0.onnx` and `voices-v1.0.bin` files into it.
3. Run `python app.py`