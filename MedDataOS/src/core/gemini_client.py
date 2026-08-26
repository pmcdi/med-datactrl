"""
Gemini API client setup and configuration.
"""
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

def create_client():
    """Create and return a Gemini API client."""
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")

    return genai.Client(api_key=GEMINI_API_KEY)

def create_config(tools, system_instruction):
    """Create a GenerateContentConfig with tools and system instruction."""
    return types.GenerateContentConfig(
        tools=[tools],
        system_instruction=system_instruction
    )

def create_contents(history):
    """Create contents array from conversation history."""
    return list(history) if history else []

def create_response(client, model, contents, config=None):
    """Generate content using the Gemini API."""
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
