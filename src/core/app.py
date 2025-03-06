import os
import re
import subprocess
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.use_cases.speech_classifier import SpeechClassifier
from src.use_cases.audio_context_analyser import generate_explanation

# Import DTO's
from src.core.dtos.analisys_dtos import Analisys
from src.core.dtos.steps_dtos import Steps

# Base directory for audio files
AUDIO_DIR = "resources/audios"

app = FastAPI(
    title="SPECTRA - Speech Classification & Transcription Analysis",
    description="API for speech analysis to determine if speech is read or spontaneous",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def entrypoint():
    return {"message": "Welcome to SPECTRA API, read the docs at /docs"}

@app.get("/analyze", response_model=Analisys)
async def audio_analyzer(file_name: str = Query(..., description="Name of the audio file in resources/audios directory")):
    audio_path = os.path.join(AUDIO_DIR, file_name)


    if file_name.endswith((".mp3", ".m4a", ".webm")) and os.path.exists(audio_path):
        wav_file = os.path.join(AUDIO_DIR, re.sub(r'\.(mp3|m4a|webm)$', '.wav', file_name))
        if not os.path.exists(wav_file):
            # Convert audio to wav format
            subprocess.run(["ffmpeg", "-i", os.path.abspath(audio_path), "-c:a", "pcm_f32le", os.path.abspath(wav_file)])

            # Update audio path
            audio_path = wav_file
        elif os.path.exists(wav_file):
            # Update audio path
            audio_path = wav_file

    # Check if the file exists and it's a wav file
    if not os.path.exists(audio_path) or not os.path.isfile(audio_path) or not audio_path.endswith(".wav"):
        raise HTTPException(status_code=404, detail=f"Audio file '{file_name}' not found in resources/audios directory")

    try:
        # Run analysis
        classifier = SpeechClassifier()

        # Classify and generate explanation for classification percentage
        features = classifier.execute(audio_path)
        explanation = generate_explanation(features)

        # Define every step of the analysis
        steps = [
            Steps(
                step_name="Reading Analysis",
                description=explanation,
                probability=features.get("reading_percentage", 0.0)
            )
        ]

        return Analisys(
            file_name=file_name,
            steps_probability=steps
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing audio file: {str(e)}")

def start():
    import uvicorn
    uvicorn.run("src.core.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
