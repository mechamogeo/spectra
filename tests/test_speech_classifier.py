import pytest
from unittest.mock import patch

from src.use_cases.speech_classifier import SpeechClassifier

def test_extract_features():
    """Test feature extraction from transcription."""
    classifier = SpeechClassifier()

    # Test with a simple transcription
    transcription = "hello□world⋄⋄⋄test□audio"
    duration = 5.0  # 5 seconds

    f1, f2, f3 = classifier.extract_features(transcription, duration)

    # Assert feature values are as expected
    # f1: Active avg word length (letters per word)
    assert f1 == pytest.approx(6.333333333333333)

    # f2: Inactive alphabets per second (pauses per second)
    assert f2 == pytest.approx(0.6)

    # f3: Words per second
    assert f3 == pytest.approx(0.6)

def test_compute_readability_score():
    """Test the readability score computation."""
    classifier = SpeechClassifier()

    # Test with various feature combinations
    score1 = classifier.compute_readability_score(7.0, 5.0, 2.0)
    score2 = classifier.compute_readability_score(5.0, 15.0, 1.5)

    # Check scores are within expected ranges (0-3)
    assert 0 <= score1 <= 3
    assert 0 <= score2 <= 3

    # Higher f1 and f3, lower f2 should yield higher readability score
    assert score1 > score2

def test_is_reading():
    """Test the reading classification threshold."""
    classifier = SpeechClassifier()

    # Test threshold behavior
    assert classifier.is_reading(1.85) == True
    assert classifier.is_reading(1.75) == False

    # Test with custom threshold
    assert classifier.is_reading(1.70, threshold_R=1.65) == True

def test_calculate_reading_percentage():
    """Test conversion of readability score to percentage."""
    classifier = SpeechClassifier()

    # Test various score conversions
    assert classifier.calculate_reading_percentage(0.0) == 0.0
    assert classifier.calculate_reading_percentage(1.5) == 50.0
    assert classifier.calculate_reading_percentage(3.0) == 100.0

    # Test boundary protection
    assert classifier.calculate_reading_percentage(-1.0) == 0.0
    assert classifier.calculate_reading_percentage(4.0) == 100.0

@patch("parselmouth.Sound")
@patch.object(SpeechClassifier, "transcribe_audio")
@patch.object(SpeechClassifier, "extract_features")
@patch.object(SpeechClassifier, "compute_readability_score")
@patch.object(SpeechClassifier, "calculate_utterance")
def test_execute(mock_utterance, mock_r_score, mock_features,
                 mock_transcribe, mock_sound, sample_audio_path):
    """Test the full execution flow of the classifier."""

    # Configure mocks
    class TranscriptionData:
        def __init__(self, language):
            self.language = language

    mock_sound.return_value.get_total_duration.return_value = 5.0
    mock_transcribe.return_value = ("test□transcription⋄with□pauses", TranscriptionData("en"))
    mock_features.return_value = (5.2, 8.4, 1.9)  # f1, f2, f3
    mock_r_score.return_value = 2.1
    mock_utterance.return_value = {"utterance_count": 3, "average_duration": 1.5}

    # Run the classifier
    classifier = SpeechClassifier()
    result = classifier.execute(sample_audio_path)

    # Verify result structure and values
    assert result["audio_duration_seconds"] == 5.0
    assert result["language"] == "en"
    assert result["readability_score"] == 2.1
    assert result["is_reading"] == True
    assert "reading_percentage" in result
    assert "utterance_analysis" in result
