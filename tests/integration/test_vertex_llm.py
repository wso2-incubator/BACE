import os
from pathlib import Path

import pytest
from loguru import logger

from coevolution.utils.config import _load_yaml_file
from infrastructure.llm_client.factory import create_llm_client
from infrastructure.llm_client.openai import OpenAIChatClient


def test_vertex_llm_live_generation() -> None:
    """
    Live integration test for Vertex AI GPT-OSS-120B.
    Loads the config, initializes the client, and sends a real prompt.

    Requires environment variables:
    - GOOGLE_PROJECT_ID
    - GOOGLE_REGION
    - GOOGLE_ENDPOINT
    """
    # Check for required environment variables
    required_vars = ["GOOGLE_PROJECT_ID", "GOOGLE_REGION", "GOOGLE_ENDPOINT"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        pytest.fail(
            f"Missing required environment variables for live test: {missing_vars}"
        )

    config_path = Path("configs/llm/vertex-gpt-oss-120b.yaml")
    assert config_path.exists(), f"Config file not found: {config_path}"

    logger.info(f"Loading config from {config_path}")
    llm_config = _load_yaml_file(config_path)

    logger.info(f"Initializing client for model: {llm_config.get('model')}")
    client = create_llm_client(**llm_config)

    assert isinstance(client, OpenAIChatClient)

    prompt = "Hi, tell me a long story of two chapters"
    logger.info(f"Sending live prompt: {prompt}")

    try:
        response = client.generate(prompt)

        logger.info("Received response from Vertex AI")
        logger.debug(f"Response (first 500 chars): {response[:500]}...")

        assert response is not None
        assert len(response) > 0
        assert (
            "chapter" in response.lower()
            or "once upon a time" in response.lower()
            or len(response) > 100
        )

        print(f"\n--- Vertex AI Response ---\n{response}\n-------------------------")

    except Exception as e:
        logger.error(f"Live generation failed: {e}")
        pytest.fail(f"Live generation failed: {e}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    test_vertex_llm_live_generation()
