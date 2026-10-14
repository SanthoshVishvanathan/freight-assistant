@echo off
REM FreightTiger Smart Assistant Runner (Windows)
echo ============================================================
echo Running FreightTiger Smart Assistant Pipeline...
echo ============================================================
python run_assistant.py run
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo Running Automated Evaluation Harness...
echo ============================================================
python eval_harness.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Evaluation harness failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] All tasks completed successfully!
if "%1" neq "--no-pause" (
    REM Can pause if launched by double-clicking in explorer
)
