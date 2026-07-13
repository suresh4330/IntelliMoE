"""
services/gemini_client.py
-------------------------
Service client for interacting with the Google Gemini API.
Dual-compatible: supports both the new google-genai and legacy google-generativeai SDKs.
"""

import logging
from typing import Optional
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_ID

# Set up logging for the module
logger = logging.getLogger(__name__)

# Try to load the new SDK to support newer AQ. API keys
HAS_NEW_SDK = False
try:
    from google import genai as new_genai
    new_client = new_genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1"})
    HAS_NEW_SDK = True
    logger.info("Successfully loaded new google-genai Client wrapper.")
except Exception as load_exc:
    logger.warning("Could not load new google-genai client: %s. Using legacy SDK...", load_exc)
    import google.generativeai as old_genai
    try:
        old_genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.exception("Failed to configure legacy Google Generative AI client.")


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = GEMINI_MODEL_ID,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using a Gemini model, automatically routing through the best available SDK.
    """
    logger.info("Sending request to Gemini API (model: %s, new_sdk: %s)...", model, HAS_NEW_SDK)

    if HAS_NEW_SDK:
        try:
            # Map parameters to the new SDK structure
            config = {"temperature": temperature}
            if system_prompt:
                config["system_instruction"] = system_prompt

            response = new_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            
            response_text = response.text
            if response_text is not None:
                return response_text
        except Exception as new_exc:
            logger.warning("Generation failed using new google-genai SDK: %s. Trying legacy fallback...", new_exc)

    # Legacy SDK Fallback
    try:
        import google.generativeai as old_genai  # noqa: PLC0415
        config = old_genai.types.GenerationConfig(temperature=temperature)
        
        kwargs = {}
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        response_text = None
        model_name = model
        
        try:
            model_instance = old_genai.GenerativeModel(
                model_name=model_name,
                generation_config=config,
                **kwargs
            )
            response = model_instance.generate_content(prompt)
            response_text = response.text
        except Exception as e1:
            logger.warning("Legacy attempt with '%s' failed: %s. Retrying with '%s'...", model_name, e1, GEMINI_MODEL_ID)
            try:
                model_name = GEMINI_MODEL_ID
                model_instance = old_genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=config,
                    **kwargs
                )
                response = model_instance.generate_content(prompt)
                response_text = response.text
            except Exception as e2:
                logger.warning("Legacy attempt with '%s' failed: %s. Retrying with 'gemini-flash-latest'...", model_name, e2)
                model_name = "gemini-flash-latest"
                model_instance = old_genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=config,
                    **kwargs
                )
                response = model_instance.generate_content(prompt)
                response_text = response.text

        if response_text is None:
            logger.warning("Received null response from Gemini API.")
            return ""

        logger.info("Successfully generated response from legacy Gemini API (used model: %s).", model_name)
        return response_text

    except Exception as e:
        logger.exception("All attempts to call Gemini API failed: %s", e)
        raise RuntimeError(f"Gemini API error: {e}") from e
