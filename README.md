# SPECTRA - Speech Classification & Transcription Analysis

SPECTRA is a speech analysis tool that can determine whether speech in an audio file is read from a prepared script or spoken spontaneously. It analyzes various speech features including word length, pauses, and speech rate to make this determination.

## Features

- Analyze audio files to classify speech as read or spontaneous
- Calculate reading probability as a percentage
- Generate human-readable explanations of analysis results
- Support for multiple audio formats (WAV, MP3, M4A, WebM)
- RESTful API for integration with other applications

## Architecture

The project follows a clean architecture pattern:

- `src/core/` - Core application logic and DTOs
- `src/use_cases/` - Business logic implementation
- `tests/` - Unit tests

## How It Works

SPECTRA uses a novel approach to classify read and spontaneous speech based on three key features:
1. **Active average word length**: The average length of words in the speech
2. **Inactive alphabets per second**: The frequency of pauses
3. **Words per second**: The overall speech rate

These features are combined to compute a readability score that indicates how likely the speech is to be read from a script.

## Dependencies

- Python 3.8+
- FastAPI
- Parselmouth
- Faster-Whisper
- NumPy
- LangChain
- Ollama (for explanation generation)
- FFmpeg (for audio conversion)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mechamogeo/spectra.git
   cd spectra
   ```

2. Install dependencies:
   ```
   pip install fastapi uvicorn parselmouth faster-whisper numpy langchain-ollama
   ```

3. Install FFmpeg for audio conversion:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Install Ollama for LLM explanations:
   Follow the instructions at [ollama.ai](https://ollama.ai/)

5. Pull the required model:
   ```
   ollama pull granite3.2:2b
   ```

## Running the API

1. Start the API server:
   ```
   python -m src.core.app
   ```

2. The API will be available at http://localhost:8000

3. Access the API documentation at http://localhost:8000/docs

## API Usage

### Analyze an audio file

```
GET /analyze?file_name=your_audio_file.wav
```

Make sure your audio file is placed in the `resources/audios/` directory.

## Running Tests

To run the test suite:

```
pytest tests/
```

## References

```bibtex
@misc{kopparapu2023novelschemeclassifyread,
      title={A Novel Scheme to classify Read and Spontaneous Speech},
      author={Sunil Kumar Kopparapu},
      year={2023},
      eprint={2306.08012},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2306.08012},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
