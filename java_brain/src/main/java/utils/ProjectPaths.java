package utils;
import java.io.File;
import java.nio.file.Path;

/**
 * ProjectPaths lists common repository-relative paths used throughout the
 * application and provides helpers to resolve them to absolute `Path` or
 * `File` instances based on the discovered application root.
 */
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
     * Resolve the configured relative path to an absolute {@link java.nio.file.Path}
     * using the detected application root.
     *
     * @return absolute Path for the enum entry
     */
    public Path getPath() {
        return PathManager.getAppRoot().resolve(relativePath);
    }

    /**
     * Convenience to obtain a {@link java.io.File} for the path.
     *
     * @return File corresponding to the resolved path
     */
    public File getFile() {
        return getPath().toFile();
    }
}
