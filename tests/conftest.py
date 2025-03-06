import os
import pytest
import tempfile
import numpy as np
import soundfile as sf

@pytest.fixture
def sample_audio_path():
    """Create a temporary test WAV file."""
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    # Create a sample audio file (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Write to a temporary WAV file
    file_path = os.path.join(temp_dir.name, "test_audio.wav")
    sf.write(file_path, audio_data, sample_rate)

    yield file_path

    # Cleanup
    temp_dir.cleanup()

@pytest.fixture
def mock_transcription():
    """Return a mock transcription with some pauses and words."""
    return "hello□world⋄⋄⋄testing□this□audio", {"language": "en"}

@pytest.fixture
def mock_features():
    """Return mock features for testing."""
    return {
        "audio_duration_seconds": 5.0,
        "language": "en",
        "features": [
            {"value": 5.2, "description": "Active average word length"},
            {"value": 8.4, "description": "Inactive alphabets per second"},
            {"value": 1.9, "description": "Words per second"}
        ],
        "readability_score": 2.1,
        "readability_assessment": {
            "level": "High",
            "explanation": "Clear and structured speech"
        },
        "utterance_analysis": {
            "count": 3,
            "average_duration": 1.5,
            "quality": "Fluent speech with good pacing"
        },
        "defined_classification": "Read Speech",
        "is_reading": True,
        "reading_percentage": 70.0,
        "interpret_reading_percentage": "Mostly structured, possibly prepared remarks"
    }
