"""
Configuration module for Product Finder.
Loads environment variables from .env file or Streamlit secrets.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
load_dotenv()


def get_secret(key: str, default: str = None) -> str:
    """
    Get secret from environment or Streamlit secrets.
    Supports both local (.env) and Streamlit Cloud deployment.
    """
    # First try environment variable
    value = os.getenv(key)
    if value:
        return value
    
    # Then try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return default


# API Keys
VOYAGE_API_KEY = get_secret("VOYAGE_API_KEY")
ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY", "6mPyaZWFhvbmBKftxcq7")

# Roboflow SAM3 Configuration
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE = "gfg-l9meh"
ROBOFLOW_WORKFLOW_ID = "sam3-with-prompts"

# Voyage AI Configuration
VOYAGE_API_URL = "https://api.voyageai.com/v1/multimodalembeddings"
VOYAGE_MODEL = "voyage-multimodal-3.5"

# Image Processing
DEFAULT_JPEG_QUALITY = 95
CROP_PADDING = 20

# Modal SAM3 Configuration
MODAL_SAM3_URL = "https://mattoboard--sam3-segmentation-fastapi-app.modal.run/"

# Output directories
OUTPUT_DIR = "output_masks"
VOYAGE_READY_DIR = "voyage_ready"
