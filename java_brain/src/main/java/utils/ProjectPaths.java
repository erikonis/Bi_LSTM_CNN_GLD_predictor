package utils;
import java.io.File;
import java.nio.file.Path;

public enum ProjectPaths {
    DATA("data"),
    CONFIG("config"),
    LOGS("data/logs"),
    PROCESSED("data/processed"),
    RAW("data/raw"),
    MARKET_DATA("data/raw/market"),
    NEWS_DATA("data/raw/news"),
    DATASETS("data/processed/datasets"),
    MODELS("data/models"),
    QUARANTINE("data/models/quarantine"),
    BRIDGE("python_engine/src/gui_backend/java_listener.py"),
    COMMS_LOG("logs/comms.log"),
    BRAIN_LOG("logs/java_gui.log"),
    PYTHON_BRAIN_LOG("logs/python_engine.log"),
    PYTHON_CWD("python_engine/"),
    MODEL_INFO("config/tickers.json");

    private final String relativePath;

    ProjectPaths(String relativePath) {
        this.relativePath = relativePath;
    }

    /**
     * Resolves the absolute path based on the Application Root.
     */
    public Path getPath() {
        return PathManager.getAppRoot().resolve(relativePath);
    }

    public File getFile() {
        return getPath().toFile();
    }
}