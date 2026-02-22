from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import uuid
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
import wave
import base64

# Keep existing google-generativeai for backward compat
import google.generativeai as genai

# New SDK for Gemini TTS
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    genai_new_available = True
except ImportError:
    print("Warning: google-genai library not found. Gemini TTS will be disabled.")
    genai_new_available = False

# OpenAI-compatible client for OpenRouter
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    print("Warning: openai library not found. OpenRouter will be disabled.")
    openai_available = False

import requests as http_requests

try:
    import chromadb
    chromadb_available = True
except ImportError:
    print("Warning: chromadb library not found. Vector storage will be disabled.")
    chromadb_available = False

from gtts import gTTS
import tempfile
import re
import PyPDF2
import docx
from io import BytesIO
import logging

# Import docling correctly
try:
    from docling.document_converter import DocumentConverter
    docling_available = True
except ImportError:
    print("Warning: docling.document_converter not found. Using fallback text extraction.")
    docling_available = False

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================
# API Configuration
# ============================================================

# OpenRouter (Primary for text generation)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Gemini (Fallback for text, Primary for TTS)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize OpenRouter client (OpenAI-compatible)
openrouter_client = None
if OPENROUTER_API_KEY and openai_available:
    try:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        print(f"OpenRouter client initialized (model: {OPENROUTER_MODEL})")
    except Exception as e:
        print(f"Error initializing OpenRouter client: {e}")
else:
    if not OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY not found. OpenRouter will be disabled.")
    if not openai_available:
        print("WARNING: openai library not available. OpenRouter will be disabled.")

# Initialize Gemini (old SDK for text fallback)
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini API configured (fallback for text generation)")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
else:
    print("WARNING: GEMINI_API_KEY not found. Gemini fallback will be disabled.")

# Initialize Gemini new SDK (for TTS)
gemini_tts_client = None
if GEMINI_API_KEY and genai_new_available:
    try:
        gemini_tts_client = genai_new.Client(api_key=GEMINI_API_KEY)
        print("Gemini TTS client initialized (primary for voice generation)")
    except Exception as e:
        print(f"Error initializing Gemini TTS client: {e}")


# ============================================================
# LLM Helper: OpenRouter Primary → Gemini Fallback
# ============================================================

def call_llm(prompt, max_tokens=4096, temperature=0.7):
    """
    Call LLM with OpenRouter as primary and Gemini as fallback.
    Returns the generated text string.
    """
    # Try OpenRouter first
    if openrouter_client:
        try:
            logger.info(f"Calling OpenRouter ({OPENROUTER_MODEL})...")
            response = openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.choices[0].message.content
            if result and len(result.strip()) > 0:
                logger.info(f"OpenRouter response received ({len(result)} chars)")
                return result
            else:
                logger.warning("OpenRouter returned empty response, trying Gemini fallback")
        except Exception as e:
            logger.warning(f"OpenRouter failed: {e}, falling back to Gemini")

    # Fallback to Gemini
    if gemini_model:
        try:
            logger.info("Calling Gemini (fallback)...")
            response = gemini_model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                logger.info(f"Gemini fallback response received ({len(response.text)} chars)")
                return response.text
            else:
                logger.error("Gemini also returned empty response")
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")

    raise Exception("All LLM providers failed. Check your API keys and network connection.")


# ============================================================
# TTS Helpers: Gemini TTS Primary → OpenRouter Audio Fallback → gTTS Fallback
# No ffmpeg required - Gemini outputs WAV, gTTS outputs MP3 natively
# ============================================================

def save_wav(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """Save raw PCM audio data to a WAV file using Python's built-in wave module."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


def combine_wav_segments(wav_files, output_path, silence_ms=500):
    """
    Combine multiple WAV files into one, with silence gaps between segments.
    Uses Python's wave module only (no ffmpeg or pydub needed).
    All WAV files must have the same sample format.
    """
    if not wav_files:
        return False

    try:
        # Read first file to get audio params
        with wave.open(wav_files[0], 'rb') as first:
            params = first.getparams()
            channels = params.nchannels
            sampwidth = params.sampwidth
            framerate = params.framerate

        # Build silence bytes for the gap
        silence_frames = int(framerate * silence_ms / 1000)
        silence_bytes = b'\x00' * (silence_frames * channels * sampwidth)

        # Combine all audio data
        combined_data = bytearray()
        for i, wav_file in enumerate(wav_files):
            try:
                with wave.open(wav_file, 'rb') as wf:
                    # If format differs, try to read anyway
                    frames = wf.readframes(wf.getnframes())
                    combined_data.extend(frames)
                    # Add silence gap between segments (not after last)
                    if i < len(wav_files) - 1:
                        combined_data.extend(silence_bytes)
            except Exception as e:
                logger.warning(f"Skipping segment {wav_file}: {e}")
                continue

        if not combined_data:
            return False

        # Write combined WAV
        with wave.open(output_path, 'wb') as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(bytes(combined_data))

        logger.info(f"Combined {len(wav_files)} WAV segments into {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to combine WAV segments: {e}")
        return False


def combine_mp3_segments(mp3_files, output_path):
    """
    Combine multiple MP3 files by simple binary concatenation.
    This works because MP3 is a frame-based format.
    """
    if not mp3_files:
        return False

    try:
        with open(output_path, 'wb') as out:
            for mp3_file in mp3_files:
                try:
                    with open(mp3_file, 'rb') as f:
                        out.write(f.read())
                except Exception as e:
                    logger.warning(f"Skipping MP3 segment {mp3_file}: {e}")
                    continue

        logger.info(f"Combined {len(mp3_files)} MP3 segments into {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to combine MP3 segments: {e}")
        return False


def generate_tts_gemini(text, output_file, speaker_configs=None):
    """
    Generate TTS using Gemini gemini-2.5-flash-preview-tts.
    Outputs WAV file directly (no ffmpeg needed).
    Returns True on success, False on failure.
    """
    if not gemini_tts_client:
        logger.warning("Gemini TTS client not available")
        return False

    try:
        logger.info(f"Generating TTS with Gemini ({len(text)} chars)...")

        config_kwargs = {
            "response_modalities": ["AUDIO"],
        }

        if speaker_configs:
            config_kwargs["speech_config"] = genai_types.SpeechConfig(
                multi_speaker_voice_config=genai_types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_configs
                )
            )
        else:
            config_kwargs["speech_config"] = genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name='Kore',
                    )
                ),
            )

        response = gemini_tts_client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )

        pcm_data = response.candidates[0].content.parts[0].inline_data.data

        # Save as WAV directly (no ffmpeg needed)
        # Ensure output ends with .wav
        wav_output = output_file.rsplit('.', 1)[0] + '.wav'
        save_wav(wav_output, pcm_data)

        logger.info(f"Gemini TTS successful: {wav_output}")
        return True

    except Exception as e:
        logger.error(f"Gemini TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_tts_openrouter(text, output_file, voice="alloy"):
    """
    Fallback TTS using OpenRouter with openai/gpt-audio-mini.
    Uses chat completions with audio modality.
    Returns True on success, False on failure.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter API key not available for TTS fallback")
        return False

    try:
        logger.info(f"Generating TTS with OpenRouter gpt-audio-mini ({len(text)} chars)...")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "openai/gpt-audio-mini",
            "modalities": ["text", "audio"],
            "audio": {
                "voice": voice,
                "format": "wav",
            },
            "messages": [
                {
                    "role": "user",
                    "content": f"Please read the following text aloud exactly as written:\n\n{text}"
                }
            ],
        }

        response = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        # Extract audio data from response
        audio_data = None
        message = data.get("choices", [{}])[0].get("message", {})

        if "audio" in message and message["audio"].get("data"):
            audio_data = base64.b64decode(message["audio"]["data"])

        if not audio_data:
            logger.warning("No audio data in OpenRouter response")
            return False

        # Save audio as WAV
        wav_output = output_file.rsplit('.', 1)[0] + '.wav'
        with open(wav_output, 'wb') as f:
            f.write(audio_data)

        logger.info(f"OpenRouter TTS successful: {wav_output}")
        return True

    except Exception as e:
        logger.error(f"OpenRouter TTS fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_tts_gtts(text, output_file, language="English"):
    """
    Last-resort fallback TTS using Google Translate TTS (gTTS).
    Outputs MP3 directly (no ffmpeg needed).
    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Generating TTS with gTTS fallback ({len(text)} chars)...")
        lang_code = "en" if language.lower() == "english" else "ar" if language.lower() == "arabic" else "en"

        # gTTS outputs MP3 natively, no conversion needed
        mp3_output = output_file.rsplit('.', 1)[0] + '.mp3'
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(mp3_output)
        logger.info(f"gTTS fallback successful: {mp3_output}")
        return True
    except Exception as e:
        logger.error(f"gTTS fallback also failed: {e}")
        return False


def get_podcast_audio_path(base_path):
    """
    Find the actual podcast audio file (could be .wav or .mp3).
    Returns the path to the existing file, preferring WAV.
    """
    wav_path = base_path.rsplit('.', 1)[0] + '.wav'
    mp3_path = base_path.rsplit('.', 1)[0] + '.mp3'

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    elif os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
        return mp3_path
    return None


# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_FOLDER'] = 'extracted'
app.config['AUDIO_FOLDER'] = 'audio'

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['EXTRACTED_FOLDER'], app.config['AUDIO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Initialize ChromaDB
if chromadb_available:
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="documents")
    except Exception as e:
        print(f"Error initializing ChromaDB: {str(e)}")
        chromadb_available = False
else:
    collection = None


# ============================================================
# Text Extraction (unchanged)
# ============================================================

def extract_text_fallback(file_path):
    """Fallback method to extract text if docling is not available"""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_with_docling(file_path):
    """Extract text using docling's document converter"""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

MAX_FILE_SIZE = 512 * 1024  # 0.5 MB = 512 KB

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route called")
    for folder in [app.config['UPLOAD_FOLDER'], app.config['EXTRACTED_FOLDER'], app.config['AUDIO_FOLDER']]:
        os.makedirs(folder, exist_ok=True)

    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Check file size (read into memory to measure, then reset)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    if file_size > MAX_FILE_SIZE:
        size_mb = round(file_size / (1024 * 1024), 2)
        print(f"File too large: {size_mb} MB (max 0.5 MB)")
        return jsonify({"error": f"File is too large ({size_mb} MB). Maximum allowed size is 0.5 MB."}), 400

    try:
        original_filename = file.filename
        print(f"Processing file: {original_filename}")
        file_id = str(uuid.uuid4())

        # Extract extension from ORIGINAL filename (before secure_filename strips non-ASCII)
        original_ext = os.path.splitext(original_filename)[1].lower()  # e.g. '.docx'

        # Use secure_filename but fall back to UUID if it strips everything
        safe_name = secure_filename(file.filename)
        if not safe_name or not os.path.splitext(safe_name)[1]:
            # secure_filename stripped all chars (e.g. Arabic filename) — use UUID + original ext
            safe_name = f"{file_id}{original_ext}"

        filename = safe_name
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print(f"Saving file to: {file_path} (original: {original_filename})")
        file.save(file_path)

        if not os.path.exists(file_path):
            print(f"Failed to save file to {file_path}")
            return jsonify({"error": "Failed to save file"}), 500

        print(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")

        # Extract text using docling or fallback
        try:
            print("Extracting text...")
            if docling_available:
                print("Using docling for text extraction")
                text = extract_text_with_docling(file_path)
            else:
                print("Using fallback text extraction")
                text = extract_text_fallback(file_path)
            print(f"Text extraction completed. Length: {len(text)} characters")
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error extracting text: {str(e)}"}), 500

        # Save extracted text
        try:
            print("Saving extracted text...")
            extracted_filename = os.path.splitext(filename)[0] + ".txt"
            extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], extracted_filename)
            with open(extracted_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Extracted text saved to: {extracted_path}")
        except Exception as e:
            print(f"Error saving extracted text: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error saving extracted text: {str(e)}"}), 500

        # Chunk the text
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        print(f"Text chunked into {len(chunks)} segments")

        # Store in vector database
        if chromadb_available and collection:
            try:
                print("Storing chunks in vector database...")
                collection.add(
                    documents=chunks,
                    metadatas=[{"file_id": file_id, "chunk_id": i} for i in range(len(chunks))],
                    ids=[f"{file_id}_{i}" for i in range(len(chunks))]
                )
                print("Vector storage complete")
            except Exception as e:
                print(f"Error storing in vector database: {str(e)}")
                import traceback
                traceback.print_exc()

        print("Upload processing completed successfully")
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "text_length": len(text)
        })
    except Exception as e:
        print(f"Unexpected error during file upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/files', methods=['GET'])
def list_files():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_stats = os.stat(file_path)
        files.append({
            "filename": filename,
            "path": file_path,
            "size": file_stats.st_size,
            "uploaded_at": file_stats.st_mtime * 1000
        })
    return jsonify({"files": files})


@app.route('/ask', methods=['POST'])
def ask_question():
    """Q&A endpoint: OpenRouter primary, Gemini fallback."""
    print("Ask question endpoint called")

    try:
        data = request.json
        if not data:
            return jsonify({"answer": "No data received. Please try again."}), 400

        question = data.get('question', '')
        filename = data.get('filename', '')

        print(f"Question: '{question}' | Filename: '{filename}'")

        if not question or not filename:
            return jsonify({"answer": "Missing question or filename. Please try again."}), 400

        # Get the extracted text from file
        try:
            extracted_filename = os.path.splitext(filename)[0] + ".txt"
            extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], extracted_filename)

            if not os.path.exists(extracted_path):
                return jsonify({"error": f"File not found: {filename}. Please upload it first."}), 404

            with open(extracted_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content_length = len(content)
            print(f"Read file successfully. Content length: {content_length} chars")
        except Exception as file_error:
            print(f"Error reading file: {str(file_error)}")
            return jsonify({"answer": "Error reading your document. Please try uploading it again."}), 400

        # Create fallback text
        words = content.split()
        context = ' '.join(words[:min(300, len(words))])
        fallback_text = f"Here's information from the document that may help answer your question:\n\n{context}"

        # Use call_llm (OpenRouter primary → Gemini fallback)
        try:
            document_content = content[:4000]
            prompt = f"Document: {document_content}\n\nQuestion: {question}\n\nPlease answer the question based ONLY on the document content."

            print(f"Sending Q&A request, prompt length: {len(prompt)}")
            answer = call_llm(prompt, max_tokens=4096, temperature=0.3)
            return jsonify({"answer": answer.strip()})

        except Exception as api_error:
            print(f"All LLM providers failed for Q&A: {str(api_error)}")
            return jsonify({"answer": fallback_text})

    except Exception as e:
        print(f"Unexpected error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "The server encountered an unexpected error. Please try again."}), 500


@app.route('/generate-podcast', methods=['POST'])
def generate_podcast():
    """Generate podcast script and audio. OpenRouter primary for script, Gemini TTS for audio."""
    data = request.json
    filename = data.get('filename')
    num_speakers = data.get('num_speakers', 2)
    language = data.get('language', 'English')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    extracted_filename = os.path.splitext(filename)[0] + ".txt"
    extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], extracted_filename)

    if not os.path.exists(extracted_path):
        return jsonify({"error": "File not found"}), 404

    # Base output path (extension will be determined by which TTS succeeds)
    output_basename = f"{os.path.splitext(filename)[0]}_podcast"
    output_base_path = os.path.join(app.config['AUDIO_FOLDER'], output_basename)

    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

    try:
        with open(extracted_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Generate podcast script using call_llm (OpenRouter primary → Gemini fallback)
        try:
            print("Generating podcast script via LLM...")

            speaker_names = [f"Speaker{i+1}" for i in range(int(num_speakers))]
            speakers_str = ", ".join(speaker_names)

            prompt = (
                f"Create a podcast conversation between {num_speakers} speakers ({speakers_str}) "
                f"about this document: {content}\n\n"
                f"Make it engaging, in {language}, and cover the key points from the document.\n\n"
                f"Format the output like this:\n"
                f"Speaker1: [Text from Speaker1]\n"
                f"Speaker2: [Text from Speaker2]\n"
                f"And so on...\n\n"
                f"Include a brief introduction and summary at the end."
            )

            print(f"Sending podcast generation request, prompt length: {len(prompt)}")
            podcast_script = call_llm(prompt, max_tokens=8192, temperature=0.9)

            # If response is too short, try simpler prompt
            if len(podcast_script.strip()) < 100:
                print("First response too short, trying simpler prompt")
                fallback_prompt = (
                    f"Create a conversation between Speaker1 and Speaker2 discussing this text: "
                    f"{content[:2000]}\n\n"
                    f"Format as:\nSpeaker1: [text]\nSpeaker2: [text]"
                )
                podcast_script = call_llm(fallback_prompt, max_tokens=4096, temperature=0.9)

            # If still too short, create minimal script
            if len(podcast_script.strip()) < 100:
                podcast_script = (
                    f"Speaker1: Welcome to our podcast discussion about this document.\n\n"
                    f"Speaker2: Let's go through the key points from the document.\n\n"
                    f"Speaker1: The document covers: {content[:500]}...\n\n"
                    f"Speaker2: Thank you for listening to our discussion."
                )

            # Parse speaker segments
            speaker_pattern = r'(Speaker\d+):\s+(.*?)(?=Speaker\d+:|$)'
            matches = re.findall(speaker_pattern, podcast_script, re.DOTALL)

            if not matches:
                print("No speaker matches found in:", podcast_script[:100] + "...")
                return jsonify({
                    "success": True,
                    "podcast_script": podcast_script,
                    "audio_path": None,
                    "error": "Could not parse speakers in the generated script"
                })

            # Build display script and segments
            display_script = ""
            script_segments = []
            for speaker, text in matches:
                script_segments.append({
                    "speaker": speaker,
                    "text": re.sub(r'[*_#`~\[\]\(\)]', '', text.strip())
                })
                display_script += f"### {speaker}:\n\n{text.strip()}\n\n"

            # ============================================
            # Generate Audio per segment, then combine
            # Gemini TTS → per-segment gTTS fallback
            # No ffmpeg required!
            # ============================================
            audio_generated = False
            audio_file_path = None
            temp_dir = tempfile.mkdtemp(prefix="podcast_audio_")

            # --- Try 1: Gemini TTS (multi-speaker, whole script as one call) ---
            if gemini_tts_client and genai_new_available:
                try:
                    tts_text = "TTS the following conversation:\n"
                    for seg in script_segments:
                        tts_text += f"{seg['speaker']}: {seg['text']}\n"

                    voice_names = ['Kore', 'Puck', 'Charon', 'Fenrir']
                    speaker_voice_configs = []
                    unique_speakers = list(dict.fromkeys([seg['speaker'] for seg in script_segments]))

                    for i, speaker in enumerate(unique_speakers):
                        voice_name = voice_names[i % len(voice_names)]
                        speaker_voice_configs.append(
                            genai_types.SpeakerVoiceConfig(
                                speaker=speaker,
                                voice_config=genai_types.VoiceConfig(
                                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                        voice_name=voice_name,
                                    )
                                ),
                            )
                        )

                    audio_generated = generate_tts_gemini(
                        tts_text,
                        output_base_path + '.wav',
                        speaker_configs=speaker_voice_configs
                    )
                    if audio_generated:
                        # Verify the file has actual content
                        wav_path = output_base_path + '.wav'
                        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                            audio_file_path = wav_path
                            print("✅ Podcast audio generated with Gemini TTS (multi-speaker)")
                        else:
                            audio_generated = False
                            logger.warning("Gemini TTS produced empty/tiny file, falling back")
                except Exception as e:
                    logger.error(f"Gemini TTS attempt failed: {e}")

            # --- Try 2: Per-segment gTTS (each segment → MP3, then combine) ---
            if not audio_generated:
                try:
                    print("Generating audio per-segment with gTTS...")
                    lang_code = "en" if language.lower() == "english" else "ar" if language.lower() == "arabic" else "en"
                    segment_files = []

                    for i, seg in enumerate(script_segments):
                        text = seg['text'].strip()
                        if not text or len(text) < 3:
                            continue

                        seg_path = os.path.join(temp_dir, f"seg_{i:04d}.mp3")
                        try:
                            print(f"  Generating segment {i+1}/{len(script_segments)} ({len(text)} chars)...")
                            tts = gTTS(text=text, lang=lang_code, slow=False)
                            tts.save(seg_path)

                            # Verify segment was created and has content
                            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 100:
                                segment_files.append(seg_path)
                                print(f"  ✅ Segment {i+1} saved ({os.path.getsize(seg_path)} bytes)")
                            else:
                                print(f"  ⚠️ Segment {i+1} too small, skipping")
                        except Exception as seg_err:
                            logger.warning(f"Failed to generate segment {i+1}: {seg_err}")
                            continue

                    # Combine all segments into one file
                    if segment_files:
                        final_mp3 = output_base_path + '.mp3'
                        combine_success = combine_mp3_segments(segment_files, final_mp3)
                        if combine_success and os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 100:
                            audio_generated = True
                            audio_file_path = final_mp3
                            print(f"✅ Combined {len(segment_files)} segments into {final_mp3} ({os.path.getsize(final_mp3)} bytes)")
                        else:
                            logger.error("Failed to combine MP3 segments")
                    else:
                        logger.error("No audio segments were successfully generated")

                except Exception as e:
                    logger.error(f"Per-segment gTTS attempt failed: {e}")

            # Clean up temp files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

            error_message = None if audio_generated else "All TTS providers failed. Script was generated but no audio could be created."

            # Determine audio URL
            audio_url = None
            if audio_file_path:
                audio_filename = os.path.basename(audio_file_path)
                audio_url = f"/get-audio/{audio_filename}"

            return jsonify({
                "success": True,
                "podcast_script": podcast_script,
                "display_script": display_script,
                "audio_path": audio_url,
                "error": error_message
            })

        except Exception as e:
            print(f"Error generating podcast: {str(e)}")
            import traceback
            traceback.print_exc()

            return jsonify({
                "error": f"Error generating podcast: {str(e)}",
                "audio_path": None
            }), 500

    except Exception as e:
        print(f"Error generating podcast: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": f"Error generating podcast: {str(e)}",
            "audio_path": None
        }), 500


@app.route('/get-audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Stream the audio file directly as binary data. Supports WAV and MP3."""
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)

    try:
        if os.path.exists(audio_path) and os.path.isfile(audio_path):
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Determine mimetype based on extension
            if filename.endswith('.wav'):
                mimetype = "audio/wav"
            else:
                mimetype = "audio/mpeg"

            return Response(
                audio_data,
                mimetype=mimetype,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            print(f"Audio file not found at {audio_path}")
            return jsonify({"error": "Audio file not found"}), 404

    except Exception as e:
        print(f"Error serving audio: {str(e)}")
        return jsonify({"error": f"Error serving audio: {str(e)}"}), 500


@app.route('/delete-file', methods=['POST'])
def delete_file_route():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    try:
        # Delete the original file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        # Delete the extracted text file
        extracted_filename = os.path.splitext(filename)[0] + ".txt"
        extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], extracted_filename)
        if os.path.exists(extracted_path):
            os.remove(extracted_path)
            print(f"Deleted extracted text: {extracted_path}")

        # Delete any associated podcast audio (check both WAV and MP3)
        base_name = os.path.splitext(filename)[0]
        for ext in ['.mp3', '.wav']:
            podcast_path = os.path.join(app.config['AUDIO_FOLDER'], f"{base_name}_podcast{ext}")
            if os.path.exists(podcast_path):
                os.remove(podcast_path)
                print(f"Deleted podcast audio: {podcast_path}")

        # Try to delete from vector database
        if chromadb_available and collection:
            try:
                print(f"Note: Vector DB entries for {filename} not deleted (would require ID mapping)")
            except Exception as e:
                print(f"Error deleting from vector DB: {str(e)}")

        return jsonify({
            "success": True,
            "message": f"File {filename} and associated data deleted successfully"
        })
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error deleting file: {str(e)}"}), 500


if __name__ == '__main__':
    print("\n=== Server Information ===")
    print(f"Document upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"Extracted text folder: {os.path.abspath(app.config['EXTRACTED_FOLDER'])}")
    print(f"Audio output folder: {os.path.abspath(app.config['AUDIO_FOLDER'])}")
    print(f"Primary LLM: OpenRouter ({OPENROUTER_MODEL})")
    print(f"Fallback LLM: Gemini (gemini-1.5-flash)")
    print(f"Primary TTS: Gemini (gemini-2.5-flash-preview-tts)")
    print(f"Fallback TTS: OpenRouter (openai/gpt-audio-mini) → gTTS")
    print(f"Server: http://0.0.0.0:5016")
    print("=========================\n")

    app.run(host='0.0.0.0', port=5016, debug=True, use_reloader=False)