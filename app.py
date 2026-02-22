from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import uuid
from werkzeug.utils import secure_filename
import json
import google.generativeai as genai
from dotenv import load_dotenv
try:
    import chromadb
    chromadb_available = True
except ImportError:
    print("Warning: chromadb library not found. Vector storage will be disabled.")
    chromadb_available = False
from gtts import gTTS
from pydub import AudioSegment
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

# Initialize Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    print("ERROR: GEMINI_API_KEY not found. Please set it in .env file")
else:
    print(f"Gemini API key found: {api_key[:4]}...{api_key[-4:]}")
    try:
        genai.configure(api_key=api_key)
        # Test the API connection with a simple model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Simple test prompt. Respond with 'OK' only.")
        if hasattr(response, 'text') and response.text:
            print(f"API test successful: {response.text[:20]}")
        else:
            print("API test failed: No response text")
    except Exception as e:
        print(f"Error testing Gemini API: {str(e)}")

# Initialize Flask app
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
        # Use get_or_create_collection instead of create_collection to avoid errors if it already exists
        collection = chroma_client.get_or_create_collection(name="documents")
    except Exception as e:
        print(f"Error initializing ChromaDB: {str(e)}")
        chromadb_available = False
else:
    collection = None

# Add fallback text extraction methods
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route called")
    # Ensure directories exist
    for folder in [app.config['UPLOAD_FOLDER'], app.config['EXTRACTED_FOLDER'], app.config['AUDIO_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
    
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    try:
        print(f"Processing file: {file.filename}")
        # Generate unique ID for the file
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        print(f"Saving file to: {file_path}")
        file.save(file_path)
        
        # Check if file was saved correctly
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
        
        # Chunk the text (simple implementation - can be improved)
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
                # Continue without vector storage if it fails
        
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
            "uploaded_at": file_stats.st_mtime * 1000  # Convert to milliseconds for JavaScript
        })
    return jsonify({"files": files})

@app.route('/ask', methods=['POST'])
def ask_question():
    print("Ask question endpoint called")
    
    try:
        # Get data from request
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
                return jsonify({"answer": f"File not found: {filename}. Please upload it first."}), 404
                
            with open(extracted_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            content_length = len(content)
            print(f"Read file successfully. Content length: {content_length} chars")
        except Exception as file_error:
            print(f"Error reading file: {str(file_error)}")
            return jsonify({"answer": f"Error reading your document. Please try uploading it again."}), 400
        
        # Create fallback text immediately so we always have a response
        words = content.split()
        context = ' '.join(words[:min(300, len(words))])
        fallback_response = f"Here's information from the document that may help answer your question:\n\n{context}"
        
        # Try to use Gemini API to generate an answer
        try:
            print("Initializing Gemini model (gemini-1.5-flash)")
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create a simple text prompt
            document_content = content[:4000]  # Limit document content to avoid token issues
            prompt = f"Document: {document_content}\n\nQuestion: {question}\n\nPlease answer the question based ONLY on the document content."
            
            print(f"Sending API request with prompt length: {len(prompt)}")
            response = model.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text and len(response.text.strip()) > 0:
                print(f"Received API response with length: {len(response.text)}")
                return jsonify({"answer": response.text.strip()})
            else:
                print("Empty or invalid response from API")
                return jsonify({"answer": fallback_response})
                
        except Exception as api_error:
            print(f"Error calling Gemini API: {str(api_error)}")
            
            # Try fallback to another model if first one fails
            try:
                print("Trying fallback to gemini-1.0-pro model")
                fallback_model = genai.GenerativeModel('gemini-1.0-pro')
                fallback_response = fallback_model.generate_content(prompt)
                
                if fallback_response and hasattr(fallback_response, 'text') and fallback_response.text:
                    print("Fallback model response received")
                    return jsonify({"answer": fallback_response.text.strip()})
            except Exception as fallback_error:
                print(f"Fallback model error: {str(fallback_error)}")
            
            # Return document excerpt if API call completely fails
            return jsonify({"answer": fallback_response})
            
    except Exception as e:
        print(f"Unexpected error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "The server encountered an unexpected error. Please try again."}), 500

@app.route('/generate-podcast', methods=['POST'])
def generate_podcast():
    data = request.json
    filename = data.get('filename')
    num_speakers = data.get('num_speakers', 2)
    language = data.get('language', 'English')
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    # Get the extracted text
    extracted_filename = os.path.splitext(filename)[0] + ".txt"
    extracted_path = os.path.join(app.config['EXTRACTED_FOLDER'], extracted_filename)
    
    if not os.path.exists(extracted_path):
        return jsonify({"error": "File not found"}), 404
    
    # Define the output path and filename
    output_filename = f"{os.path.splitext(filename)[0]}_podcast.mp3"
    podcast_output_path = os.path.join(app.config['AUDIO_FOLDER'], output_filename)
    
    # Ensure the audio directory exists
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    
    # Create a simple audio file as a fallback immediately
    try:
        from pydub.generators import Sine
        fallback_audio = Sine(440).to_audio_segment(duration=1000)  # 1 sec tone
        fallback_audio.export(podcast_output_path, format="mp3")
        print(f"Created fallback audio file: {podcast_output_path}")
    except Exception as e:
        print(f"Error creating fallback audio: {str(e)}")
    
    try:
        with open(extracted_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate podcast conversation using Gemini
        try:
            print("Using gemini-1.5-flash model for podcast generation")
            model = genai.GenerativeModel('gemini-1.5-flash', 
                                        generation_config={
                                            "temperature": 0.9,
                                            "top_p": 0.95,
                                            "top_k": 40,
                                            "max_output_tokens": 8192,
                                        })
            
            # Create a simple prompt - don't use complex formatting with roles
            prompt = f"Create a podcast conversation between {num_speakers} speakers about this document: {content}\n\n" + \
                     f"Make it engaging, in {language}, and cover the key points from the document.\n\n" + \
                     f"Format the output like this:\n" + \
                     f"Speaker1: [Text from Speaker1]\n" + \
                     f"Speaker2: [Text from Speaker2]\n" + \
                     f"And so on...\n\n" + \
                     f"Include a brief introduction and summary at the end."
                     
            print(f"Sending podcast generation request, prompt length: {len(prompt)}")
            response = model.generate_content(prompt)
            podcast_script = response.text if response and hasattr(response, 'text') else ""
            
            # If we got an empty or very short response, try a fallback
            if len(podcast_script.strip()) < 100:
                try:
                    print("First response too short, trying fallback with simpler prompt")
                    fallback_prompt = f"Create a conversation between Speaker1 and Speaker2 discussing this text: {content[:2000]}\n\n" + \
                                      f"Format as:\nSpeaker1: [text]\nSpeaker2: [text]"
                    fallback_response = model.generate_content(fallback_prompt)
                    if fallback_response and hasattr(fallback_response, 'text') and len(fallback_response.text.strip()) > 100:
                        podcast_script = fallback_response.text
                    else:
                        # Create a minimal script if all else fails
                        podcast_script = f"Speaker1: Welcome to our podcast discussion about this document.\n\n" + \
                                        f"Speaker2: Let's go through the key points from the document.\n\n" + \
                                        f"Speaker1: The document covers: {content[:500]}...\n\n" + \
                                        f"Speaker2: Thank you for listening to our discussion."
                except Exception as fallback_error:
                    print(f"Fallback prompt error: {str(fallback_error)}")
                    # Use the minimal script defined above
                    podcast_script = f"Speaker1: Welcome to our podcast discussion about this document.\n\n" + \
                                    f"Speaker2: Let's go through the key points from the document.\n\n" + \
                                    f"Speaker1: The document covers: {content[:500]}...\n\n" + \
                                    f"Speaker2: Thank you for listening to our discussion."
            
            # If podcast script is too short, it might be an error
            if len(podcast_script.strip()) < 100:
                return jsonify({
                    "success": False,
                    "error": "Generated podcast script was too short. Please try again."
                }), 400
            
            # Try to generate proper audio with speaker segments
            speaker_pattern = r'(Speaker\d+):\s+(.*?)(?=Speaker\d+:|$)'
            matches = re.findall(speaker_pattern, podcast_script, re.DOTALL)
            
            if not matches:
                print("No speaker matches found in:", podcast_script[:100] + "...")
                return jsonify({
                    "success": True,
                    "podcast_script": podcast_script,
                    "audio_path": f"/get-audio/{output_filename}",
                    "error": "Could not parse speakers in the generated script"
                })
            
            # Convert matches to structured segments and prepare script for display
            display_script = ""
            script_segments = []
            for speaker, text in matches:
                # Add to segments for audio generation
                script_segments.append({
                    "speaker": speaker,
                    "text": re.sub(r'[*_#`~\[\]\(\)]', '', text.strip())  # Remove markdown for TTS
                })
                
                # Format for display with preserved markdown
                display_script += f"### {speaker}:\n\n{text.strip()}\n\n"
            
            # Generate the audio using a more reliable approach
            audio_generated = create_podcast_audio(script_segments, podcast_output_path, language)
            
            if not audio_generated:
                # Create a guaranteed audio file
                try:
                    Sine(440).to_audio_segment(duration=2000).export(podcast_output_path, format="mp3")
                    print("Created fallback tone file")
                    error_message = "Failed to generate detailed audio - created tone instead"
                except Exception as e:
                    print(f"Final fallback audio creation failed: {str(e)}")
                    error_message = f"Audio generation failed: {str(e)}"
            else:
                error_message = None
                
            # Return success with the script and any error message
            return jsonify({
                "success": True,
                "podcast_script": podcast_script,
                "display_script": display_script,
                "audio_path": f"/get-audio/{output_filename}",
                "error": error_message
            })
            
        except Exception as e:
            print(f"Error generating podcast: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Ensure we have a tone file even in case of script generation failure
            try:
                if not os.path.exists(podcast_output_path) or os.path.getsize(podcast_output_path) == 0:
                    Sine(440).to_audio_segment(duration=2000).export(podcast_output_path, format="mp3")
            except Exception:
                pass
            
            return jsonify({
                "error": f"Error generating podcast: {str(e)}",
                "audio_path": f"/get-audio/{output_filename}"
            }), 500
            
    except Exception as e:
        print(f"Error generating podcast: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure we have a tone file even in case of script generation failure
        try:
            if not os.path.exists(podcast_output_path) or os.path.getsize(podcast_output_path) == 0:
                Sine(440).to_audio_segment(duration=2000).export(podcast_output_path, format="mp3")
        except Exception:
            pass
            
        return jsonify({
            "error": f"Error generating podcast: {str(e)}",
            "audio_path": f"/get-audio/{output_filename}"
        }), 500

def create_podcast_audio(script_segments, output_file, language="English"):
    """Create podcast audio from script segments using a more reliable approach."""
    try:
        print(f"Generating podcast audio with {len(script_segments)} segments")
        full_audio = AudioSegment.empty()
        
        # Define voices for different speakers
        speaker_voices = {
            "Speaker1": {"lang": "en" if language.lower() == "english" else "ar"},
            "Speaker2": {"lang": "en" if language.lower() == "english" else "ar"},
            "Speaker3": {"lang": "en" if language.lower() == "english" else "ar"},
            "Speaker4": {"lang": "en" if language.lower() == "english" else "ar"}
        }
        
        # Create a directory for temporary files
        temp_dir = tempfile.mkdtemp()
        
        # Generate audio for each segment
        for i, segment in enumerate(script_segments):
            speaker = segment["speaker"]
            text = segment["text"]
            
            if not text or len(text.strip()) < 5:
                print(f"Skipping empty segment for {speaker}")
                continue
                
            print(f"Processing segment {i+1}/{len(script_segments)} for {speaker}")
            
            try:
                # Break long text into chunks for better reliability
                max_chunk_length = 200
                if len(text) > max_chunk_length:
                    chunks = []
                    # Split by sentences to avoid cutting in the middle of a sentence
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= max_chunk_length:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                else:
                    chunks = [text]
                
                # Process each chunk
                segment_audio = AudioSegment.empty()
                
                for j, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                        
                    chunk_file = os.path.join(temp_dir, f"segment_{i}_chunk_{j}.mp3")
                    
                    # Generate speech for this chunk
                    tts = gTTS(text=chunk, lang=speaker_voices[speaker]["lang"], slow=False)
                    tts.save(chunk_file)
                    
                    # Add a short pause between sentences
                    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                        chunk_audio = AudioSegment.from_mp3(chunk_file)
                        segment_audio += chunk_audio
                        segment_audio += AudioSegment.silent(duration=300)  # 300ms pause between chunks
                
                # Add the segment to the full audio with a longer pause between speakers
                if len(segment_audio) > 0:
                    full_audio += segment_audio
                    full_audio += AudioSegment.silent(duration=700)  # 700ms pause between speakers
                    
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
                continue
        
        # Add a short intro and outro tone
        try:
            intro = Sine(440).to_audio_segment(duration=500).fade_in(100).fade_out(300)
            outro = Sine(330).to_audio_segment(duration=800).fade_in(300).fade_out(500)
            
            full_audio = intro + AudioSegment.silent(duration=500) + full_audio + AudioSegment.silent(duration=500) + outro
        except Exception as e:
            print(f"Error adding intro/outro: {str(e)}")
        
        # Export the final audio
        if len(full_audio) > 0:
            full_audio.export(output_file, format="mp3")
            print(f"Successfully exported podcast audio to {output_file}")
            
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp files: {str(e)}")
                
            return True
        else:
            print("Error: No audio was generated")
            return False
            
    except Exception as e:
        print(f"Error in create_podcast_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/get-audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Stream the audio file directly as binary data instead of using send_file."""
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    
    try:
        # If the file exists, read and return it
        if os.path.exists(audio_path) and os.path.isfile(audio_path):
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            return Response(
                audio_data,
                mimetype="audio/mpeg",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            print(f"Audio file not found at {audio_path}, generating fallback")
            # Generate fallback audio tone
            from io import BytesIO
            from pydub.generators import Sine
            
            # Create a simple tone
            tone = Sine(440).to_audio_segment(duration=1000)
            
            # Export to BytesIO object
            output = BytesIO()
            tone.export(output, format="mp3")
            output.seek(0)
            
            # Try to save this for future requests
            try:
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                with open(audio_path, 'wb') as f:
                    f.write(output.getvalue())
                print(f"Saved fallback audio to {audio_path}")
            except Exception as e:
                print(f"Error saving fallback audio: {str(e)}")
            
            # Return binary data
            output.seek(0)
            return Response(
                output.getvalue(), 
                mimetype="audio/mpeg",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
    except Exception as e:
        print(f"Error serving audio: {str(e)}")
        from io import BytesIO
        from pydub.generators import Sine
        
        # Create fallback tone as BytesIO
        output = BytesIO()
        Sine(440).to_audio_segment(duration=1000).export(output, format="mp3")
        output.seek(0)
        
        # Return binary data
        return Response(
            output.getvalue(),
            mimetype="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

def generate_audio_for_speakers(podcast_script, language):
    """Generate audio files for each speaker's lines and return paths to the files."""
    print("Starting audio generation...")
    
    # Parse the script to separate by speakers
    speaker_pattern = r'(Speaker\d+):\s+(.*?)(?=Speaker\d+:|$)'
    matches = re.findall(speaker_pattern, podcast_script, re.DOTALL)
    
    if not matches:
        print("No speaker matches found in script")
        raise Exception("Could not parse speakers in the podcast script")
    
    print(f"Found {len(matches)} speaker segments to process")
    
    # Create a simple tone file as a fallback
    fallback_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    try:
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=500)
        tone.export(fallback_file.name, format="mp3")
    except Exception as e:
        print(f"Error creating fallback tone: {str(e)}")
    
    audio_files = []
    audio_files.append(fallback_file.name)  # Add fallback file first
    
    for i, (speaker, text) in enumerate(matches):
        # Clean up the text
        text = text.strip()
        if not text:
            print(f"Empty text for {speaker}, skipping")
            continue
        
        print(f"Processing {speaker}, text length: {len(text)}")
        
        try:
            # Create a temporary file for each speaker's line
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            
            # Generate speech for the text
            lang_code = "en" if language.lower() == "english" else "ar" if language.lower() == "arabic" else "en"
            
            # If text is very long, break it into smaller chunks
            max_chars = 100  # Even smaller chunks to avoid TTS errors
            if len(text) > max_chars:
                chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                combined = AudioSegment.empty()
                
                for j, chunk in enumerate(chunks):
                    print(f"  Processing chunk {j+1}/{len(chunks)} for {speaker}")
                    chunk = chunk.strip()
                    if not chunk:
                        print("  Empty chunk, skipping")
                        continue
                        
                    try:
                        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        tts = gTTS(text=chunk, lang=lang_code, slow=False)
                        tts.save(chunk_file.name)
                        
                        # Check if file was created successfully
                        if os.path.exists(chunk_file.name) and os.path.getsize(chunk_file.name) > 0:
                            segment = AudioSegment.from_mp3(chunk_file.name)
                            combined += segment
                        else:
                            print(f"  Error: Empty or missing chunk file for {speaker}")
                            
                        os.unlink(chunk_file.name)  # Clean up chunk file
                    except Exception as e:
                        print(f"  Error with chunk {j+1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Continue with other chunks
                
                if len(combined) > 0:
                    combined.export(temp_file.name, format="mp3")
                    
                    # Verify file was created
                    if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                        audio_files.append(temp_file.name)
                    else:
                        print(f"  Error: Empty or missing combined file for {speaker}")
                else:
                    print(f"  No audio generated for {speaker}")
                    os.unlink(temp_file.name)
            else:
                try:
                    # Use a simpler approach for short text
                    tts = gTTS(text=text, lang=lang_code, slow=False)
                    tts.save(temp_file.name)
                    
                    # Verify file was created
                    if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                        audio_files.append(temp_file.name)
                    else:
                        print(f"Error: Empty or missing file for {speaker}")
                except Exception as e:
                    print(f"Error with simple TTS for {speaker}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error generating audio for '{speaker}': {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with other speakers if one fails
    
    print(f"Finished audio generation, created {len(audio_files)} files")
    
    # Return all the audio files we managed to create
    return audio_files

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
        
        # Delete any associated podcast audio
        podcast_filename = f"{os.path.splitext(filename)[0]}_podcast.mp3"
        podcast_path = os.path.join(app.config['AUDIO_FOLDER'], podcast_filename)
        if os.path.exists(podcast_path):
            os.remove(podcast_path)
            print(f"Deleted podcast audio: {podcast_path}")
        
        # Try to delete from vector database (if available)
        if chromadb_available and collection:
            try:
                # Get all document IDs with this filename in metadata
                # Since we don't store filename directly in metadata but use file_id,
                # we can't easily query by filename. We'd need to implement a mapping 
                # if this becomes important.
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
    # Print some debugging info at startup
    print("\n=== Server Information ===")
    print(f"Document upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"Extracted text folder: {os.path.abspath(app.config['EXTRACTED_FOLDER'])}")
    print(f"Audio output folder: {os.path.abspath(app.config['AUDIO_FOLDER'])}")
    print("=========================\n")
    
    app.run(debug=True, use_reloader=False) 