"""
services/openai_client.py
-------------------------
Service client for interacting with the OpenAI API.
Provides a reusable generate_response function with built-in logging and error handling.
"""

import logging
from typing import Any, Dict, Optional
from openai import OpenAI
from config.settings import OPENAI_API_KEY

# Set up logging for the module
logger = logging.getLogger(__name__)

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured. Please set it in your .env file.")
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.exception("Failed to initialize the OpenAI client.")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e
    return _client


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate text using an OpenAI model.

    Parameters
    ----------
    prompt : str
        The user query or prompt.
    system_prompt : Optional[str], default=None
        Optional instructions to guide the model's behavior.
    model : str, default="gpt-4o-mini"
        The ID of the model to use on OpenAI (e.g. "gpt-4o-mini", "gpt-4o").
    temperature : float, default=0.7
        Sampling temperature (0.0 for deterministic, higher for more creative).
    max_tokens : Optional[int], default=None
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The generated response text.

    Raises
    ------
    RuntimeError
        If there is an API error or failure.
    """
    logger.info("Sending request to OpenAI API (model: %s)...", model)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Construct the API parameters
        params: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Call the API
        completion = get_client().chat.completions.create(**params)
        
        # Extract and validate response
        response_text = completion.choices[0].message.content
        if response_text is None:
            logger.warning("Received null response from OpenAI API.")
            return ""

        logger.info("Successfully generated response from OpenAI API (tokens generated: %s).",
                    getattr(completion.usage, 'completion_tokens', 'unknown'))
        return response_text

    except Exception as e:
        logger.exception("Error occurred while calling OpenAI API: %s", e)
        raise RuntimeError(f"OpenAI API error: {e}") from e
