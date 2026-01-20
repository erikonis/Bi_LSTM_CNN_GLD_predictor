# BSP3 System Instructions

This project is a dual-engine application combining a Java Graphical Interface with a Python Inference Engine. Follow the steps below based on your operating system.

---

## Prerequisites

Regardless of your OS, you must have the following installed:

1. Java Runtime (JRE/JDK): Version 17 or 24 is required.
   * Check via: `java -version`
2. Internet Connection: Required for the first launch to download Python dependencies.

---

## Windows Instructions (Easy Launch)

We have bundled the uv package manager for Windows inside the bin/ folder to make setup automatic.

1. Download/Unzip the project folder.
2. Locate the file named launch.bat in the root folder.
3. Double-click launch.bat.
   * First Run: A console window will appear and download the required Python libraries. This may take 1-2 minutes depending on your connection.
   * Success: The Java GUI will open automatically once the environment is synchronized.

---

## macOS and Linux Instructions

Non-Windows users need to install the uv manager manually once before launching.

### 1. Install Dependencies
Open your terminal and run:

```bash
# Install uv (Astral Python Manager)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Restart terminal or refresh your profile
source $HOME/.cargo/env
```

### 2. Set Permissions
Navigate to the BSP3 folder in your terminal and allow the script to execute:

```bash
chmod +x launch.sh
```

### 3. Launch
Run the script:
```
./launch.sh
```
---

## Project Structure Overview

* **java_brain/**: Contains the GUI and logic for managing the system.
* **python_engine/**: Contains the AI/ML logic and models.
* **data/**: Where your models and processed datasets are stored.
* **config/**: Contains tickers.json which tracks available models.
* **logs/**: Check here if anything goes wrong.



---

## Troubleshooting

### Python Module Errors
If the logs report ModuleNotFoundError (e.g., No module named 'dtale' or 'rich'):

1. Delete the `python_engine/.venv` folder.
2. Run the launcher (`launch.bat` or `launch.sh`) again. This forces uv to rebuild the environment from the `uv.lock` file.

### Port Conflicts
The system uses two ports (assigned dynamically). If you see a "Port already in use" error, ensure no other instances of the app are running. You can check your Task Manager for any stray `java.exe` or `python.exe` processes.

### Log Locations
If the application crashes, check the `/logs` folder:

* **java_gui.log**: Issues with the interface or buttons.
* **python_engine.log**: Issues with AI models or data processing.
* **comms.log**: Issues with the socket connection between Java and Python.


---

(c) 2026 BSP3 Project, Erikas Kadiša