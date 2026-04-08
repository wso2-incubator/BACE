import os
from pathlib import Path

import pytest
from loguru import logger

from coevolution.utils.config import _load_yaml_file
from infrastructure.llm_client.factory import create_llm_client
from infrastructure.llm_client.openai import OpenAIChatClient, OpenAIClient


def test_groq_llm_live_generation() -> None:
    """
    Live integration test for Groq GPT-OSS-120B.
    Loads the config, initializes the client, and sends a real prompt.

    Requires environment variables:
    - GROQ_API_KEY
    """
    # Check for required environment variables
    if "GROQ_API_KEY" not in os.environ:
        pytest.fail("Missing required environment variable for live test: GROQ_API_KEY")

    config_path = Path("configs/llm/groq-gpt-oss-120b.yaml")
    assert config_path.exists(), f"Config file not found: {config_path}"

    logger.info(f"Loading config from {config_path}")
    llm_config = _load_yaml_file(config_path)

    logger.info(f"Initializing client for model: {llm_config.get('model')}")
    client = create_llm_client(**llm_config)

    # Note: Depending on the provider in the yaml, this could be OpenAIClient or OpenAIChatClient
    assert isinstance(client, (OpenAIClient, OpenAIChatClient))

    prompt = "Hi, tell me a long story of two chapters"
    logger.info(f"Sending live prompt to Groq: {prompt}")

    try:
        response = client.generate(prompt)

        logger.info("Received response from Groq")
        logger.debug(f"Response (first 500 chars): {response[:500]}...")

        assert response is not None
        assert len(response) > 0
        assert (
            "chapter" in response.lower()
            or "once upon a time" in response.lower()
            or len(response) > 100
        )

        print(f"\n--- Groq Response ---\n{response}\n-------------------------")

    except Exception as e:
        logger.error(f"Live generation failed: {e}")
        pytest.fail(f"Live generation failed: {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    test_groq_llm_live_generation()
