from dataclasses import dataclass

@dataclass
class Transcription:
    """ Data Transfer Object for Transcription """
    content: str
    duration: float
    speech_rate: float
    pause_ratio: float
    reading_probability: float
