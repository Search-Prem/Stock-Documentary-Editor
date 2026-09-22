@echo off
REM One-time setup on Windows. Requires Python 3.10+ and FFmpeg on PATH (winget install Gyan.FFmpeg).
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
if not exist .env copy .env.example .env
echo.
echo Setup done. Open .env in Notepad and paste your PEXELS_API_KEY, then double-click run.bat
pause
exit /b 0
:err
echo Setup failed - see the messages above.
pause
exit /b 1
