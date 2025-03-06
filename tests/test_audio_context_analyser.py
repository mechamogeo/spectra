from unittest.mock import patch, MagicMock

from src.use_cases.audio_context_analyser import generate_explanation

@patch("langchain_ollama.OllamaLLM")
def test_generate_explanation(mock_ollama_llm, mock_features):
    """Test explanation generation with mocked LLM."""
    # Configure mock
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "This audio demonstrates clear reading patterns with consistent pacing."
    mock_ollama_llm.return_value = mock_llm_instance

    # Generate explanation
    explanation = generate_explanation(mock_features)

    # Verify results
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert mock_llm_instance.invoke.called == False
