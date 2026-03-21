@echo off
title GRID-X Server
echo Starting GRID-X Backend...
cd "C:\Users\Faiz Ahmed\OneDrive\Desktop\GRID-X\API"
start "" cmd /k "python main.py"
echo Waiting for server to start...
timeout /t 15 /nobreak > nul
start "" "http://localhost:8000/frontend/index.html"
