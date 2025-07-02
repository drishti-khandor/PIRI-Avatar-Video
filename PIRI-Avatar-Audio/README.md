# PIRI Avatar Audio

A real-time conversational AI application featuring voice interaction with an AI avatar. This project combines a FastAPI backend with a Next.js frontend, leveraging FastRTC for low-latency audio streaming and processing to deliver a seamless, natural voice-based chat experience.

## ✨ Features

- 🎙️ Real-time voice-to-voice conversation using advanced streaming audio technology
- 🤖 Powered by Azure OpenAI for natural language understanding and dialogue
- 🦾 FastRTC integration for efficient, low-latency audio streaming and voice activity detection
- 🎭 Interactive AI avatar interface
- 🚀 Built with Next.js and FastAPI for optimal performance
- 🎨 Modern UI with Tailwind CSS and Framer Motion

## ⚡ About FastRTC

FastRTC is a Python library designed for real-time audio streaming, voice activity detection (VAD), and seamless integration with conversational AI systems. In this project, FastRTC enables:

- Low-latency, bidirectional audio streaming between the client and backend, crucial for natural conversational experiences.
- Voice Activity Detection (VAD) using Silero VAD, ensuring the system responds only when the user is speaking.
- Streaming speech-to-text (STT) and text-to-speech (TTS) pipelines, allowing for incremental processing and immediate feedback.
- Efficient resource usage by processing audio in small chunks and pausing/resuming as needed.

By leveraging FastRTC, the backend can process live audio, transcribe speech, generate AI responses, and synthesize voice replies—all in real time. This enables a fluid, interactive voice chat with the AI avatar, closely mimicking human conversation.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+
- Azure OpenAI service credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PIRI-Avatar-Audio
   ```

2. **Set up the backend**
   ```bash
   cd fastRTC-app/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Environment Variables**
   Create a `.env` file in the `backend` directory with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd fastRTC-app/backend
   uvicorn main:app --reload
   ```

2. **Start the frontend development server**
   ```bash
   cd fastRTC-app/frontend
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python |
| AI | Azure OpenAI |
| Audio | FastRTC (real-time streaming, VAD, STT, TTS) |
| Styling | Tailwind CSS, Framer Motion |

## 📂 Project Structure

```
PIRI-Avatar-Audio/
├── fastRTC-app/
│   ├── backend/           # FastAPI server
│   │   ├── main.py        # Main application file
│   │   └── .env          # Environment variables
│   └── frontend/         # Next.js application
│       ├── app/          # Next.js app directory
│       ├── components/    # React components
│       └── public/       # Static files
└── README.md
```

## 💡 Why FastRTC?

- **Real-time experience**: FastRTC's efficient audio streaming and chunked processing minimize latency, making conversations with the AI avatar feel natural and immediate.

- **Flexible integration**: Easily connects with speech-to-text, language models, and text-to-speech modules, supporting advanced conversational pipelines.

- **Robust voice detection**: Built-in VAD ensures the system only processes relevant audio, saving resources and improving responsiveness.

- **Production-ready**: Designed for scalable, low-latency applications in voice AI, customer support bots, and interactive avatars.
