#!/bin/bash
echo "-----------------------------------------"
echo "   Initializing BSP3 System (Unix)"
echo "-----------------------------------------"

# 1. Ensure uv is available
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Mac/Linux users should have uv installed via 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
    exit
fi

# 2. Sync Python
if [ ! -d "python_engine/.venv" ]; then
    echo "[Step 1/2] Creating virtual environment..."
    uv sync --project python_engine
else
    echo "[Step 1/2] Python environment OK."
fi

# 3. Launch Java
echo "[Step 2/2] Launching User Interface..."
java -jar java_brain/target/java_brain-1.0-SNAPSHOT.jar