"""
Root-level entry point for Streamlit Cloud.
Streamlit Cloud looks for streamlit_app.py in the root directory.
"""
import sys
import os

# Add src to path and import the main app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from streamlit_app import main

if __name__ == "__main__":
    main()
