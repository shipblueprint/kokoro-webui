import os
import logging
import gradio as gr
import numpy as np
import soundfile as sf
import re
from kokoro_onnx import Kokoro

# =============================================================================
# NOTE: This version has pause settings configured as requested:
# - SHORT_PAUSE_DURATION = 0.3 seconds (between sentences)
# - LONG_PAUSE_DURATION = 0.7 seconds (between paragraphs)
#
# Sample text to demonstrate pause behavior:
# "Hello there. This is a sample sentence.\n\nThis is a new paragraph. Notice the longer pause after the double newline."
# =============================================================================

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kokoro_model_path = os.path.join(BASE_DIR, 'models', 'kokoro-v1.0.onnx')
voices_model_path = os.path.join(BASE_DIR, 'models', 'voices-v1.0.bin')
DEFAULT_SPEED = 0.85
MAX_TEXT_LENGTH = 2000
# Pause settings (in seconds)
SHORT_PAUSE_DURATION = 0.3  # Between sentences
LONG_PAUSE_DURATION = 0.7   # Between paragraphs
# Default blend ratio (base voice percentage)
DEFAULT_BLEND_RATIO = 50

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'kokoro_tts.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize model with error handling
kokoro = None

def initialize_model():
    global kokoro
    if kokoro is not None:
        return kokoro
    
    try:
        # Check if model files exist
        if not os.path.exists(kokoro_model_path):
            raise FileNotFoundError(f"Kokoro model not found at: {kokoro_model_path}")
        if not os.path.exists(voices_model_path):
            raise FileNotFoundError(f"Voices model not found at: {voices_model_path}")
            
        # Initialize the model
        kokoro = Kokoro(kokoro_model_path, voices_model_path)
        logger.info("Models loaded successfully")
        return kokoro
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise

# Input validation
def validate_input(text):
    if not text.strip():
        return False, "Text cannot be empty"
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
    return True, "Valid input"

# Split text into sentences with proper pause detection
def split_text_into_sentences(text):
    # Split on sentence-ending punctuation followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    return sentences

# Create speech with natural pauses
def create_with_pauses(text: str, voice: str, blend_voice: str = None, 
                     blend_ratio: float = DEFAULT_BLEND_RATIO, speed: float = DEFAULT_SPEED,
                     short_pause: float = SHORT_PAUSE_DURATION, long_pause: float = LONG_PAUSE_DURATION):
    # Validate input
    is_valid, message = validate_input(text)
    if not is_valid:
        return (22050, np.zeros(22050)), message
    
    try:
        # Ensure model is initialized
        model = initialize_model()
        
        # Process blend voice if provided
        if blend_voice and blend_voice != "None":
            # Convert blend ratio from percentage to decimal
            base_ratio = blend_ratio / 100
            blend_part = 1.0 - base_ratio
            
            base_voice = model.get_voice_style(voice)
            blend_vector = model.get_voice_style(blend_voice)
            voice = np.add(base_voice * base_ratio, blend_vector * blend_part)
        
        # Split text into sentences
        sentences = split_text_into_sentences(text)
        
        if not sentences:
            return (22050, np.zeros(22050)), "No valid sentences found in text"
        
        # Generate speech for each sentence and add pauses
        all_samples = []
        sample_rate = 22050
        
        for i, sentence in enumerate(sentences):
            logger.info(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
            samples, sr = model.create(
                sentence,
                voice=voice,
                speed=speed,
                is_phonemes=False
            )
            sample_rate = sr
            all_samples.extend(samples)
            
            # Add pause after each sentence except the last one
            if i < len(sentences) - 1:
                pause_duration = short_pause
                # Use longer pause for paragraph breaks (double newline)
                if '\n\n' in text and i < len(sentences) - 1 and text.find(sentence) + len(sentence) < len(text):
                    next_char_pos = text.find(sentence) + len(sentence)
                    if next_char_pos < len(text) and text[next_char_pos:next_char_pos+2] == '\n\n':
                        pause_duration = long_pause
                
                pause_samples = int(sample_rate * pause_duration)
                all_samples.extend(np.zeros(pause_samples))
        
        # Convert to float32 and normalize
        all_samples = np.array(all_samples).astype(np.float32)
        max_val = np.max(np.abs(all_samples))
        if max_val > 0:
            all_samples = all_samples / max_val
        
        status = f"Generated speech with voice: {voice}{' and blend voice: ' + blend_voice + f' (ratio: {int(blend_ratio)}%/{int(100-blend_ratio)}%)' if blend_voice and blend_voice != 'None' else ''} (added natural pauses)"
        return (sample_rate, all_samples), status
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return (22050, np.zeros(22050)), f"Error: {str(e)}"

def generate_podcast(sentence1, sentence2, sentence3, voice1, voice2, voice3, speed=DEFAULT_SPEED):
    try:
        # Initialize model
        model = initialize_model()
        
        # Process each sentence
        all_samples = []
        sample_rate = 22050
        sentences = [sentence1, sentence2, sentence3]
        voices = [voice1, voice2, voice3]
        
        for i, (text, voice) in enumerate(zip(sentences, voices)):
            if not text.strip() or not voice:
                logger.info(f"Skipping sentence {i+1}")
                continue
            
            logger.info(f"Processing sentence {i+1}: {text[:30]}...")
            samples, sr = model.create(
                text=text,
                voice=voice,
                speed=speed,
                is_phonemes=False
            )
            sample_rate = sr
            all_samples.extend(samples)
            
            # Add a short silence between sentences
            if i < len(sentences) - 1:
                all_samples.extend(np.zeros(int(sample_rate * SHORT_PAUSE_DURATION)))
        
        if not all_samples:
            logger.warning("No audio samples generated for podcast")
            return None
        
        # Convert to float32 and normalize
        all_samples = np.array(all_samples).astype(np.float32)
        max_val = np.max(np.abs(all_samples))
        if max_val > 0:
            all_samples = all_samples / max_val
        
        # Save to temporary file
        temp_file = os.path.join(BASE_DIR, "temp_podcast.wav")
        sf.write(temp_file, all_samples, sample_rate)
        logger.info(f"Podcast generated successfully: {temp_file}")
        
        return temp_file
    except Exception as e:
        logger.error(f"Error generating podcast: {str(e)}", exc_info=True)
        return None

def create_app():
    with gr.Blocks(title="Kokoro TTS WebUI v7") as app:
        gr.Markdown("# Kokoro TTS WebUI v7")
        gr.Markdown("Generate speech with various voices and natural pauses")
        gr.Markdown("\n### Enhanced Features")
        gr.Markdown("- Voice blending with customizable ratio (70/30, 60/40, etc.)")
        gr.Markdown("- Adjustable pause durations between sentences and paragraphs")
        gr.Markdown("- Speech speed control")
        gr.Markdown("\n### Sample Text")
        gr.Markdown("\"Hello there. This is a sample sentence.\n\nThis is a new paragraph. Notice the longer pause after the double newline.")

        with gr.Tab("Single Voice"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Enter text",
                        lines=5,
                        placeholder="Type your text here..."
                    )
                with gr.Column(scale=1):
                    voice_input = gr.Dropdown(
                        label="Voice",
                        choices=sorted(initialize_model().get_voices()),
                        value="af_sky"
                    )
                    blend_voice_input = gr.Dropdown(
                        label="Blend Voice (Optional)",
                        choices=["None"] + sorted(initialize_model().get_voices()),
                        value="None"
                    )
                    blend_ratio_slider = gr.Slider(
                        label="Base Voice Ratio (%)",
                        minimum=10,
                        maximum=90,
                        value=DEFAULT_BLEND_RATIO,
                        step=5,
                        info="Controls how much of each voice is blended (e.g., 70 = 70% base voice, 30% blend voice)"
                    )
                    speed_input = gr.Slider(
                        label="Speech Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=DEFAULT_SPEED,
                        step=0.1
                    )
                    short_pause_slider = gr.Slider(
                        label="Sentence Pause Duration (s)",
                        minimum=0.1,
                        maximum=2.0,
                        value=SHORT_PAUSE_DURATION,
                        step=0.1
                    )
                    long_pause_slider = gr.Slider(
                        label="Paragraph Pause Duration (s)",
                        minimum=0.3,
                        maximum=3.0,
                        value=LONG_PAUSE_DURATION,
                        step=0.1
                    )
            generate_btn = gr.Button("Generate Speech", variant="primary")
            audio_output = gr.Audio(label="Output Audio")
            status_output = gr.Textbox(label="Status")

            generate_btn.click(
                fn=create_with_pauses,
                inputs=[text_input, voice_input, blend_voice_input, blend_ratio_slider, 
                       speed_input, short_pause_slider, long_pause_slider],
                outputs=[audio_output, status_output]
            )

        with gr.Tab("Podcast Generator"):
            gr.Markdown("Create a podcast with up to 3 sentences and different voices")
            sentences = []
            voices = []
            
            for i in range(3):
                with gr.Row():
                    with gr.Column(scale=3):
                        sentence = gr.Textbox(
                            label=f"Sentence {i+1}",
                            lines=2,
                            placeholder=f"Enter sentence {i+1}..."
                        )
                        sentences.append(sentence)
                    with gr.Column(scale=1):
                        voice = gr.Dropdown(
                            label=f"Voice {i+1}",
                            choices=sorted(initialize_model().get_voices()),
                            value="af_sky"
                        )
                        voices.append(voice)
            
            podcast_speed = gr.Slider(
                label="Speech Speed",
                minimum=0.5,
                maximum=2.0,
                value=DEFAULT_SPEED,
                step=0.1
            )
            podcast_output = gr.File(label="Podcast File")
            podcast_btn = gr.Button("Generate Podcast", variant="primary")

            podcast_btn.click(
                fn=generate_podcast,
                inputs=sentences + voices + [podcast_speed],
                outputs=[podcast_output]
            )

    return app

# Initialize the model when the app starts
if __name__ == "__main__":
    try:
        initialize_model()
        ui = create_app()
        # Use 127.0.0.1 instead of 0.0.0.0 to avoid port conflicts
        ui.launch(debug=True, server_port=7861, show_error=True, inbrowser=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please check if the model files are present in the models directory.")