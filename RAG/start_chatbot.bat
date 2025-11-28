@echo off
echo Starting RAG Chatbot Web Interface...
echo.
echo If prompted by Streamlit, you can skip the email registration by leaving it blank and pressing Enter.
echo.
timeout /t 2 /nobreak >nul
python -m streamlit run web_interface.py