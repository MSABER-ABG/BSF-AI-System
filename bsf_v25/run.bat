@echo off
title BSF AI Banking Intelligence System
color 1F
echo.
echo  ========================================
echo   BSF AI Banking Intelligence System
echo   Developed by Accord Business Group
echo  ========================================
echo.
echo  Starting Streamlit server...
echo  Opening browser at http://localhost:8501
echo.
cd /d "%~dp0"
pip install -r requirements.txt -q
streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
