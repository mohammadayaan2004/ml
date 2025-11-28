#!/usr/bin/env python3
"""
Launcher script for the RAG Chatbot Web Interface
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching RAG Chatbot Web Interface...")
    print("Please wait while the application starts...")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Launch the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_interface.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching the application: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())