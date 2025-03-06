import os

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from src.use_cases.speech_classifier import SpeechClassifier

# Configuração do log para debug
# import logging
# logging.basicConfig(level=logging.INFO)

# Modelo LLM para análise de contexto
ollama_model = "granite3.2:2b" # Modelo LLM para análise de contexto

def generate_explanation(features):
    template = """
    You are an expert speech analysis system. Analyze the following speech data and determine whether the speech was read or spontaneous.

    Input Features:
    {input_features}

    Percentage of Reading: {percentage}%
    User Language: {language}

    Based on these features, determine, in natural language a detailed explanation of why the
    percentage of reading (how much of the speech appears to be read from a script).

    Your analysis should consider:
    - Audio duration and word metrics
    - Readability scores and assessments
    - Utterance quality analysis
    - The final decision it's from the defined classification criteria
    - Use the sample of defined classification as a reference Reading (above 70%) vs. spontaneous speech (below 30%)

    Guidelines for your explanation:
    - Use natural, conversational language in {language}
    - Avoid technical jargon or complex terms even if relevant or necessary
    - Explain your reasoning in terms a non-expert would understand (e.g., a child)
    - Refer to specific metrics from the input that influenced your decision
    - Do not use analogies or examples in your explanation
    - Limit your explanation to 244 characters or less
    - Provide a clear, concise, and accurate explanation with simple language

    Provide your response as a human-readable explanation in text only.

    Only return the explanation text, nothing else.
    """

    llm = OllamaLLM(model=ollama_model, temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["language","input_features","percentage"],
        template=template
    )

    chain = prompt | llm
    return chain.invoke({"language": features["language"], "input_features": features, "percentage": features["reading_percentage"]})

if __name__ == '__main__':

    # For each .wav inside resources/audios/*.wav make for each one the classification
    for audio_file in os.listdir("resources/audios"):
        if audio_file.endswith(".wav"):
            features = SpeechClassifier().execute(f"resources/audios/{audio_file}")

            # Execute the run function of new classification
            explanation = generate_explanation(features)
            print({ "file": audio_file, "is_reading": bool(features["is_reading"]), "percentage": f"{features["reading_percentage"]:.2f}%", "description": explanation })
