@echo off
echo ========================================
echo Twitter NER Project Startup
echo ========================================
echo.

echo Starting Backend Server...
start "NER Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --port 8000 --reload"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting Frontend UI...
start "NER Frontend" cmd /k "cd /d "%~dp0frontend" && streamlit run app.py --server.port 8501"

echo.
echo ========================================
echo Application is starting!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend UI: http://localhost:8501
echo.
echo Press any key to open the UI in your browser...
pause > nul

start http://localhost:8501

echo.
echo To stop the application, close both terminal windows.
echo.
