import logging
import parselmouth
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

class SpeechClassifier:
    def __init__(self, whisper_model="base", silence_threshold=0.5):
        self.whisper_model = whisper_model
        self.silence_threshold = silence_threshold

    def transcribe_audio(self, audio_path):
        """
        This function provides the transcription step with some casual definitions.
          - '⋄' represents an inactive alphabet (e.g. silence/pause)
          - '□' is used as a word separator.
        """
        model = WhisperModel(self.whisper_model, device="cpu", compute_type="float32")
        batched_model = BatchedInferencePipeline(model=model)

        segments, info = batched_model.transcribe(audio_path, beam_size=5)
        transcription_tokens = []
        prev_end = 0.0
        for segment in segments:
            start, end, text = segment.start, segment.end, segment.text.strip()

            # If there's a significant gap between the previous segment and the current one,
            # insert a '⋄' token to indicate a pause/silence.
            gap = start - prev_end
            if gap >= self.silence_threshold:
                num_tokens = int(gap / self.silence_threshold)
                transcription_tokens.extend(["⋄"] * num_tokens)

            # Append the transcribed text; here we convert it to lowercase
            # and replace spaces with the word separator token '□' if required.
            words = text.lower().split()
            transcription_tokens.append("□".join(words))

            prev_end = end

        transcription = "□".join(transcription_tokens)
        return transcription, info

    def extract_features(self, transcription, duration):
        """
        Extracts features based on the transcription and audio duration.

        Features:
          f1: Active average word length (active alphabets per word)
          f2: Inactive alphabets per second (InActive aps)
          f3: Word rate (words per second; wps)

        Assumes that:
          - Words are separated by the token "□"
          - Inactive alphabets are denoted by '⋄'
          - Active alphabets are all alphabetic characters (ignoring the '⋄')
        """
        # Split transcription into words using '□' as the delimiter.
        words = [word for word in transcription.split("□") if word.strip()]
        num_words = len(words)

        # Count total alphabet characters (ignoring special symbols).
        total_alphabets = sum(1 for c in transcription if c.isalpha())
        # Count the number of inactive alphabets (assumed to be represented by '⋄')
        inactive_alphabets = transcription.count('⋄')
        # Active alphabets: total alphabets (assumed here to be all alphabetic chars)
        # In a full implementation, you might have a different token for inactive characters.
        active_alphabets = total_alphabets

        # Derived features (ensure we do not divide by zero)
        f1 = active_alphabets / num_words if num_words > 0 else 0
        f2 = inactive_alphabets / duration if duration > 0 else 0
        f3 = num_words / duration if duration > 0 else 0

        return f1, f2, f3

    def compute_readability_score(self, f1, f2, f3, lambdas=(1, 1, 1), thresholds=(6, 10, 1.80)):
        """
        Computes the readability score (R) using the three features.
        The formula (from the paper) is:

          R = sigmoid1 + sigmoid2 + sigmoid3

        where:
          sigmoid1 = 1 / (1 + exp(-λ1*(f1 - τ1)))
          sigmoid2 = 1 / (1 + exp(λ2*(f2 - τ2)))
          sigmoid3 = 1 / (1 + exp(-λ3*(f3 - τ3)))

        Default parameters: λ1,2,3 = 1, τ1 = 6, τ2 = 10, τ3 = 1.75.
        """
        l1, l2, l3 = lambdas
        tau1, tau2, tau3 = thresholds
        s1 = 1 / (1 + np.exp(-l1 * (f1 - tau1)))
        s2 = 1 / (1 + np.exp(l2 * (f2 - tau2)))
        s3 = 1 / (1 + np.exp(-l3 * (f3 - tau3)))
        R = s1 + s2 + s3
        return R

    def is_reading(self, R, threshold_R=1.80):
        """ Determines if the speech is read based on the readability score. """
        return R >= threshold_R

    def classify_speech(self, R, threshold_R=1.80):
        """ Classifies the speech based on the computed readability score only. """
        return "Read Speech" if self.is_reading(R, threshold_R) else "Spontaneous Speech"


    def calculate_utterance(self, audio_path, min_pitch=75, max_pitch=500, min_pause=0.3):
        """
        Calculate utterances in the audio file.

        Parameters:
        - audio_path: Path to the audio file
        - min_pitch: Minimum pitch for voice analysis
        - max_pitch: Maximum pitch for voice analysis
        - silence_threshold: Silence threshold (in intensity)
        - min_pause: Minimum pause duration (in seconds)

        Returns:
        - Dictionary with utterance count and average duration
        """
        sound = parselmouth.Sound(audio_path)

        # Calculate intensity
        intensity = sound.to_intensity()

        # Extract intensity contour
        intensity_values = intensity.values.T[0]
        time_step = intensity.time_step

        # Find silent intervals
        silent_intervals = []
        in_silence = False
        silence_start = 0

        for i, intensity_val in enumerate(intensity_values):
            time = i * time_step

            if intensity_val < self.silence_threshold:
                if not in_silence:
                    in_silence = True
                    silence_start = time
            else:
                if in_silence:
                    in_silence = False
                    silence_duration = time - silence_start
                    if silence_duration >= min_pause:
                        silent_intervals.append((silence_start, time))

        # Add final silence if audio ends with silence
        if in_silence:
            silence_duration = sound.get_total_duration() - silence_start
            if silence_duration >= min_pause:
                silent_intervals.append((silence_start, sound.get_total_duration()))

        # Determine utterances (between silences)
        utterances = []
        if not silent_intervals:
            # No silences detected, entire audio is one utterance
            utterances.append((0, sound.get_total_duration()))
        else:
            # Add utterance before first silence if any
            if silent_intervals[0][0] > 0:
                utterances.append((0, silent_intervals[0][0]))

            # Add utterances between silences
            for i in range(len(silent_intervals) - 1):
                utterances.append((silent_intervals[i][1], silent_intervals[i+1][0]))

            # Add utterance after last silence if any
            if silent_intervals[-1][1] < sound.get_total_duration():
                utterances.append((silent_intervals[-1][1], sound.get_total_duration()))

        # Calculate statistics
        utterance_count = len(utterances)
        utterance_durations = [end - start for start, end in utterances]
        avg_utterance_duration = sum(utterance_durations) / utterance_count if utterance_count > 0 else 0

        return {
            "utterance_count": utterance_count,
            "average_duration": avg_utterance_duration,
        }

    def evaluate_utterance_quality(self, utterance_data, count_threshold=5, duration_threshold=2.0):
        """
        Evaluate the quality of speech based on utterance statistics.

        Parameters:
        - utterance_data: Output from calculate_utterance function
        - count_threshold: Threshold for utterance count
        - duration_threshold: Threshold for average utterance duration

        Returns:
        - String indicating the quality of the utterances
        """
        count = utterance_data["utterance_count"]
        avg_duration = utterance_data["average_duration"]

        if count <= count_threshold and avg_duration >= duration_threshold:
            return "Fluent speech with good pacing"
        elif count > count_threshold and avg_duration < duration_threshold:
            return "Choppy speech with frequent pauses"
        elif count <= count_threshold and avg_duration < duration_threshold:
            return "Brief utterances, possibly hesitant speech"
        else:
            return "Long but numerous utterances, possibly rambling speech"

    def assess_readability(self, readability_score, threshold=1.80):
        # Additional readability metrics
        if readability_score >= 2.5:
            level = "Very High"
            explanation = "Extremely clear and well-structured speech"
        elif readability_score >= 2.0:
            level = "High"
            explanation = "Clear and structured speech"
        elif readability_score >= threshold:
            level = "Moderate"
            explanation = "Readable but with some spontaneous elements"
        elif readability_score >= 1.0:
            level = "Low"
            explanation = "Predominantly spontaneous speech"
        else:
            level = "Very Low"
            explanation = "Highly spontaneous, potentially disorganized speech"

        return {
            "readability_level": level,
            "explanation": explanation,
        }

    def calculate_reading_percentage(self, R):
        """
        Converts the readability score (R) to a percentage indicating
        how much the audio sounds like read speech vs. spontaneous speech.

        The score R ranges from 0 to 3, where 3 is definitely read speech
        and 0 is definitely spontaneous speech.

        Returns a percentage between 0-100.
        """
        # Normalize the score to a percentage (0-100)
        # R=0 → 0%, R=3 → 100%
        percentage = (R / 3) * 100

        # Ensure we stay within 0-100 bounds
        percentage = max(0, min(100, percentage))

        return round(percentage, 1)  # Round to 1 decimal place

    def interpret_reading_percentage(self, percentage):
        """Provides a qualitative interpretation of the reading percentage."""
        if percentage >= 80:
            return "Highly structured, likely read from a script"
        elif percentage >= 60:
            return "Mostly structured, possibly prepared remarks"
        elif percentage >= 40:
            return "Mix of prepared and spontaneous elements"
        elif percentage >= 20:
            return "Mostly spontaneous with some preparation"
        else:
            return "Highly spontaneous, unscripted speech"

    def execute(self, audio_path):
        # Load the audio file using parselmouth
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        logging.info("Audio Duration (sec): %s", duration)

        # Transcribe the audio with pauses and word separators
        transcription, transcription_info = self.transcribe_audio(audio_path)

        # Extract features f1, f2, f3
        f1, f2, f3 = self.extract_features(transcription, duration)
        logging.info("Extracted Features:")
        logging.info("  f1 (Active avg word length) = %s", f1)
        logging.info("  f2 (Inactive alphabets per sec) = %s", f2)
        logging.info("  f3 (Words per sec) = %s", f3)

        # Compute the readability score R
        R = self.compute_readability_score(f1, f2, f3)
        logging.info("Computed Readability Score R: %s", R)

        # Assess readability based on the score
        RA = self.assess_readability(R)
        logging.info("Readability Assessment:")
        logging.info("  Readability Level: %s", RA['readability_level'])
        logging.info("  Explanation: %s", RA['explanation'])

        # Calculate utterance statistics
        utterance_data = self.calculate_utterance(audio_path)
        utterance_quality = self.evaluate_utterance_quality(utterance_data)

        logging.info("Utterance Analysis:")
        logging.info("  Utterance Count: %s", utterance_data['utterance_count'])
        logging.info("  Average Utterance Duration: %s seconds", utterance_data['average_duration'])
        logging.info("  Utterance Quality: %s", utterance_quality)

        # Classify the audio segment
        classification = self.classify_speech(R)
        reading_percentage = self.calculate_reading_percentage(R)
        interpret_reading_percentage = self.interpret_reading_percentage(reading_percentage)

        logging.info("Final classification: %s", classification)
        logging.info("Reading Percentage: %s", reading_percentage)
        logging.info("Interpreted Reading Percentage: %s", interpret_reading_percentage)

        # Prepare and return results as a dictionary
        result = {
            "audio_duration_seconds": duration,
            "language": transcription_info.language,
            "features": [
                {
                    "value": f1,
                    "description": "Active average word length",
                },
                {
                    "value": f2,
                    "description": "Inactive alphabets per second",
                },
                {
                    "value": f3,
                    "description": "Words per second",
                }
            ],
            "readability_score": R,
            "readability_assessment": {
                "level": RA["readability_level"],
                "explanation": RA["explanation"]
            },
            "utterance_analysis": {
                "count": utterance_data["utterance_count"],
                "average_duration": utterance_data["average_duration"],
                "quality": utterance_quality
            },
            "defined_classification": classification,
            "is_reading": self.is_reading(R),
            "reading_percentage": reading_percentage,
            "interpret_reading_percentage": interpret_reading_percentage,
        }

        return result

if __name__ == '__main__':
    # Replace 'path_to_audio.wav' with the actual path to your .wav file
    audio_file = "resources/audios/nao-leitura.wav"

    # Execute the run function of new class
    result = SpeechClassifier().execute(audio_file)
    print(result)
