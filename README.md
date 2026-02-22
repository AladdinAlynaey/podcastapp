# Document Podcast Generator

A Python Flask application that transforms documents into podcast-style conversations with multiple speakers, using Gemini AI and text-to-speech technology.

## Features

- **File Upload**: Upload PDF, Word, or other document types
- **Text Extraction**: Extract text content from uploaded documents using docling
- **Interactive Q&A**: Ask questions about the document content and get AI-powered responses
- **Podcast Generation**: Transform document content into engaging conversations
- **Audio Synthesis**: Convert the generated conversation into an audio podcast with distinct voices for each speaker
- **Multiple Languages**: Support for different languages including English and Arabic
- **Customizable**: Adjust the number of speakers in the conversation

## Requirements

- Python 3.7+
- Google Gemini API key

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd podcast-generator
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory by copying the example:
```
cp env.example .env
```

4. Edit the `.env` file and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and go to `http://localhost:5000`

3. Upload a document through the web interface

4. Once uploaded, you can:
   - Ask questions about the document content
   - Generate a podcast-style conversation
   - Listen to the podcast directly in the browser
   - Download the podcast audio file

## Directory Structure

- `/uploads`: Stores uploaded documents
- `/extracted`: Stores extracted text from documents
- `/audio`: Stores generated podcast audio files
- `/templates`: Contains HTML templates for the web interface

## Technology Stack

- **Backend**: Flask (Python)
- **Document Processing**: docling
- **Vector Database**: ChromaDB
- **AI**: Google Gemini API
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Text-to-Speech**: gTTS (Google Text-to-Speech)
- **Audio Processing**: pydub

## License

This project is licensed under the MIT License - see the LICENSE file for details. 