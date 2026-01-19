package utils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class PathManager {

    /**
     * Utility methods to discover the application root and manage common
     * project directories (models, datasets, logs, etc.).
     */

    private static Path appRoot = null;

    /**
     * Look for the repository/application root by searching parent
     * directories for the presence of the `java_brain` folder.
     *
     * @return absolute Path to the application root
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
     * Ensure and return the model folder and its `results` subfolder for a
     * given ticker and model name. Directories are created if missing.
     *
     * @param ticker ticker symbol (folder name)
     * @param modelName model metadata folder name
     * @return array with [0]=modelFolder, [1]=resultFolder
     */
    public static Path[] getModelDir(String ticker, String modelName) {
        Path modelFolder = ProjectPaths.MODELS.getPath().resolve(ticker).resolve(modelName);
        Path resultFolder = modelFolder.resolve("results");

        modelFolder.toFile().mkdirs();
        resultFolder.toFile().mkdirs();

        return new Path[]{modelFolder, resultFolder};
    }

    /**
     * Ensure and return the dataset folder for the provided ticker.
     *
     * @param ticker ticker symbol (case-insensitive)
     * @return Path to the dataset folder
     */
    public static Path getDatasetDir(String ticker) {
        Path datasetFolder = ProjectPaths.DATASETS.getPath().resolve(ticker.toUpperCase());
        datasetFolder.toFile().mkdirs();
        return datasetFolder;
    }

    /**
     * Create all standard project directories defined in `ProjectPaths`.
     * This is safe to call multiple times.
     */
    public static void initializeDirectories() {
        for (ProjectPaths p : ProjectPaths.values()) {
            p.getFile().mkdirs();
        }
    }
}