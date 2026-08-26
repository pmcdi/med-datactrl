"""
Main entry point for the multi-agent data analysis system.
Orchestrates specialized agents using Google's Gemini API.
"""
from src.core.orchestrator import run_pipeline


if __name__ == "__main__":
    run_pipeline()
