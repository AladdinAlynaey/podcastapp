<p align="center">
  <img src="https://img.icons8.com/fluency/96/podcast.png" alt="DocsPodcast Logo" width="96" height="96"/>
</p>

<h1 align="center">🎙️ Alaadin's DocsPodcast</h1>

<p align="center">
  <strong>Transform any document into an AI-powered podcast or chat with your files using intelligent Q&A</strong>
</p>

<p align="center">
  <a href="#features"><img src="https://img.shields.io/badge/✨_Features-blue?style=for-the-badge" alt="Features"/></a>
  <a href="#demo"><img src="https://img.shields.io/badge/🎬_Live_Demo-ff006e?style=for-the-badge" alt="Demo"/></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/🚀_Quick_Start-3fb950?style=for-the-badge" alt="Quick Start"/></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/🏗️_Architecture-bc8cff?style=for-the-badge" alt="Architecture"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-2.0-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/OpenRouter-GPT--4o--mini-6366f1?style=flat-square" alt="OpenRouter"/>
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash_TTS-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=flat-square" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Theme-Dark_Mode-0d1117?style=flat-square" alt="Dark Mode"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 🌟 What is DocsPodcast?

**DocsPodcast** is a full-stack AI application that takes your documents (PDF, DOCX, TXT) and turns them into two powerful experiences:

| 💬 **Smart Q&A Chat** | 🎙️ **Podcast Generator** |
|:---:|:---:|
| Ask any question about your document and get instant, context-aware AI answers | Generate multi-speaker podcast conversations from your documents with real AI voices |
| Powered by GPT-4o-mini + Gemini fallback | Gemini 2.5 Flash TTS with multi-speaker support |
| Full markdown rendering with syntax highlighting | Per-segment audio generation with automatic combining |

---

<a name="features"></a>
## ✨ Features

### 🧠 AI-Powered Intelligence
- **Dual LLM Architecture** — OpenRouter (GPT-4o-mini) as primary, Gemini 1.5 Flash as fallback
- **RAG Pipeline** — Documents are chunked and stored in ChromaDB vector database for semantic retrieval
- **Context-Aware Responses** — Q&A answers are grounded exclusively in your document content

### 🎙️ Podcast Generation Engine
- **Multi-Speaker Scripts** — Generates natural conversations between 2-4 speakers
- **Tiered TTS System**:
  - 🥇 **Gemini 2.5 Flash TTS** — Multi-speaker voices (Kore, Puck, Charon, Fenrir)
  - 🥈 **Google TTS (gTTS)** — Per-segment generation with automatic MP3 combining
- **Language Support** — English and Arabic podcast generation
- **Segment Combining** — Individual speaker segments are generated separately and combined into one seamless audio file

### 🎨 Premium Dark Theme UI
- **GitHub-inspired** dark palette (`#0d1117` backgrounds, `#58a6ff`/`#bc8cff` accents)
- **Glassmorphism** effects with ambient background glows
- **Smooth micro-animations** — hover effects, glow borders, typing indicators
- **Responsive design** — works beautifully on mobile and desktop
- **Rich markdown rendering** — tables, code blocks, blockquotes, lists, and more

### 📄 Document Processing
- **Multi-format support** — PDF, DOCX, DOC, TXT
- **Unicode filename support** — Arabic, Chinese, and other non-Latin filenames handled correctly
- **0.5 MB file size limit** — validated on both client and server side
- **Drag & drop upload** with animated progress bar

### 🔒 Robust Error Handling
- **Graceful fallbacks** at every level (LLM, TTS, text extraction)
- **Clear error messages** — no more "Unknown error"
- **File validation** — type checking, size limits, duplicate handling

---

<a name="demo"></a>
## 🎬 Live Demo

> 🌐 **Try it now:** [ragpodcast.alaadin-alynaey.site](https://ragpodcast.alaadin-alynaey.site/)

---

<a name="architecture"></a>
## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Client)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │  Upload UI  │  │  Q&A Chat    │  │  Podcast Gen  │   │
│  │ (Drag&Drop) │  │  (Markdown)  │  │  (Audio Play) │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘   │
└─────────┼────────────────┼──────────────────┼───────────┘
          │                │                  │
          ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask Backend (app.py)                │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                 Text Extraction                  │   │
│  │   PDF (PyPDF2) │ DOCX (python-docx) │ TXT        │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │            ChromaDB Vector Store                 │   │
│  │     Document chunking → Embedding → Retrieval    │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼                               ▼               │
│  ┌─────────────┐                ┌──────────────┐        │
│  │   Q&A LLM   │                │    Podcast   │        │
│  │             │                │  Generator   │        │
│  └──────┬──────┘                └──────┬───────┘        │
│         │                              │                │
│         ▼                              ▼                │
│  ┌──────────────────────────────────────────────┐       │
│  │              AI Provider Cascade             │       │
│  │                                              │       │
│  │  Text Gen: OpenRouter ──▶ Gemini (fallback) │       │
│  │  TTS:       Gemini TTS ──▶ gTTS (fallback)  │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

<a name="quick-start"></a>
## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **API Keys:**
  - [OpenRouter API Key](https://openrouter.ai/) (free tier available)
  - [Google Gemini API Key](https://aistudio.google.com/apikey) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/AladdinAlynaey/podcastapp.git
cd podcastapp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# OpenRouter API (Primary LLM)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini

# Google Gemini API (Fallback LLM + Primary TTS)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application

```bash
python app.py
```

The app will start at **http://localhost:5016** 🎉

---

## 📁 Project Structure

```
podcastapp/
├── 🐍 app.py                  # Flask backend — routes, AI logic, TTS
├── 📄 requirements.txt         # Python dependencies
├── 🔐 .env                     # API keys (not committed)
├── 🔐 .env.example             # Environment template
├── 🚫 .gitignore               # Git exclusions
│
├── 📂 templates/
│   └── index.html              # Main HTML page
│
├── 📂 static/
│   ├── css/
│   │   └── main.css            # Premium dark theme (950+ lines)
│   └── js/
│       └── main.js             # Frontend logic, chat, audio player
│
├── 📂 uploads/                  # Uploaded documents (gitignored)
├── 📂 extracted/                # Extracted text files (gitignored)
└── 📂 audio/                    # Generated podcast audio (gitignored)
```

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | HTML5, CSS3, JavaScript | UI with dark theme, animations |
| **Backend** | Flask 2.0 (Python) | REST API, file processing |
| **Primary LLM** | OpenRouter (GPT-4o-mini) | Q&A, podcast script generation |
| **Fallback LLM** | Google Gemini 1.5 Flash | Backup text generation |
| **Primary TTS** | Gemini 2.5 Flash Preview TTS | Multi-speaker voice synthesis |
| **Fallback TTS** | Google TTS (gTTS) | Per-segment MP3 generation |
| **Vector Store** | ChromaDB | Document embedding & retrieval |
| **PDF Parsing** | PyPDF2 | Extract text from PDFs |
| **DOCX Parsing** | python-docx | Extract text from Word files |
| **Markdown** | marked.js + DOMPurify | Safe rich text rendering |

---

## 🎯 How It Works

### Document Upload Flow
```
Upload File → Validate (size ≤ 0.5MB, type check)
           → Extract Text (PyPDF2 / python-docx / raw)
           → Chunk Text (1000 char segments)
           → Embed in ChromaDB vector store
           → Ready for Q&A and Podcast generation
```

### Q&A Chat Flow
```
User Question → Retrieve relevant chunks from ChromaDB
             → Build context prompt
             → OpenRouter GPT-4o-mini (or Gemini fallback)
             → Markdown-formatted answer with typing animation
```

### Podcast Generation Flow
```
Document Content → Generate multi-speaker script via LLM
               → Parse speaker segments (Speaker1, Speaker2, ...)
               → For each segment: Generate TTS audio
               → Combine all segments into single audio file
               → Stream audio with custom player
```

---

## 🌍 Language Support

| Language | Q&A Chat | Podcast Script | TTS Audio |
|:---------|:--------:|:--------------:|:---------:|
| 🇬🇧 English | ✅ | ✅ | ✅ |
| 🇸🇦 Arabic | ✅ | ✅ | ✅ |

---

## ⚙️ Configuration

| Variable | Required | Description |
|:---------|:--------:|:------------|
| `OPENROUTER_API_KEY` | ✅ | Your OpenRouter API key |
| `OPENROUTER_MODEL` | ❌ | LLM model (default: `openai/gpt-4o-mini`) |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for TTS + fallback |
| `FLASK_APP` | ❌ | Flask entry point (default: `app.py`) |
| `FLASK_ENV` | ❌ | Environment (default: `development`) |

---

## 🛡️ Security & Limits

- 🔒 API keys stored in `.env` (never committed to git)
- 📏 **Max file size:** 0.5 MB (512 KB) — enforced client + server side
- 🧹 File inputs sanitized via `secure_filename` with Unicode fallback
- 🛡️ HTML output sanitized via DOMPurify
- 🔐 No user data sent to third parties beyond the AI API calls

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built by <a href="https://alaadin-alynaey.site">Alaadin Alynaey</a></strong>
</p>

<p align="center">
  <a href="https://github.com/AladdinAlynaey/podcastapp">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-0d1117?style=for-the-badge&logo=github&logoColor=white" alt="Star"/>
  </a>
</p>