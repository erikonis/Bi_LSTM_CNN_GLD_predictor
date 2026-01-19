@echo off
setlocal
echo -----------------------------------------
echo   Initializing BSP3 System
echo -----------------------------------------

:: 1. Check for Python Environment
if not exist "python_engine\.venv" (
    echo [Step 1/2] Virtual Environment not found. Creating...
    :: Call the bundled uv.exe specifically
    .\bin\uv.exe sync --project python_engine
) else (
    echo [Step 1/2] Python environment OK.
)

:: 2. Launch the Java Application
echo [Step 2/2] Launching User Interface...
:: We use the Fat JAR created by Maven Shade
java -jar java_brain/target/java_brain-1.0-SNAPSHOT.jar

pause