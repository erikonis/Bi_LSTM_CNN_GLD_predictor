package utils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class PathManager {

    private static Path appRoot = null;

    /**
     * Looks for a specific folder (like "java_brain") in parents.
     */
    public static Path getAppRoot() {
        if (appRoot != null) return appRoot;

        Path current = Paths.get("").toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("java_brain"))) {
                appRoot = current;
                return appRoot;
            }
            current = current.getParent();
        }
        
        // Fallback to current working directory
        appRoot = Paths.get("").toAbsolutePath();
        return appRoot;
    }

    /**
     * Gets model and result directories, creating them if they don't exist.
     */
    public static Path[] getModelDir(String ticker, String modelName) {
        Path modelFolder = ProjectPaths.MODELS.getPath().resolve(ticker).resolve(modelName);
        Path resultFolder = modelFolder.resolve("results");

        modelFolder.toFile().mkdirs();
        resultFolder.toFile().mkdirs();

        return new Path[]{modelFolder, resultFolder};
    }

    /**
     * Gets dataset directory, creating it if it doesn't exist.
     */
    public static Path getDatasetDir(String ticker) {
        Path datasetFolder = ProjectPaths.DATASETS.getPath().resolve(ticker.toUpperCase());
        datasetFolder.toFile().mkdirs();
        return datasetFolder;
    }

    /**
     * Initializes all standard directories (mkdir exist_ok=True)
     */
    public static void initializeDirectories() {
        for (ProjectPaths p : ProjectPaths.values()) {
            p.getFile().mkdirs();
        }
    }
}