@echo off
REM Start the InfoSeek Retrieval Server
cd /d "%~dp0"

echo ============================================
echo   InfoSeek Retrieval Server Launcher
echo ============================================
echo.

call venv\Scripts\activate.bat
echo [INFO] Virtual environment activated
echo.

set HF_ENDPOINT=https://hf-mirror.com
echo [INFO] Starting retrieval server...
echo [INFO] Port: 28706
echo [INFO] API Docs: http://localhost:28706/docs
echo [INFO] Health: http://localhost:28706/health
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

python launch_retrieval_server.py

pause

