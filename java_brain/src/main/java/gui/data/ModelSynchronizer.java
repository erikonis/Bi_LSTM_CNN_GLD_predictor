package gui.data;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import static utils.ProjectPaths.MODELS;
import static utils.ProjectPaths.MODEL_INFO;
import static utils.ProjectPaths.QUARANTINE;

public class ModelSynchronizer {

    private static final ObjectMapper mapper = new ObjectMapper();
    private static final String METADATA_FILE = "metadata.json";
    private static final String CONFIG_PATH = MODEL_INFO.getPath().toString();
    private static final Path QUARANTINE_DIR = QUARANTINE.getPath();
    private static final Path MODELS_ROOT = MODELS.getPath();



    /**
     * Synchronize models by scanning the models root directory and
     * updating the `tickers.json` configuration.
     */
    public static void sync() {
        sync(MODELS_ROOT);
    }

    /**
     * Perform synchronization using the provided models root path. This will
     * scan per-ticker folders, validate model metadata, quarantine
     * corrupted folders and write the resulting registry to disk.
     *
     * @param modelsRoot absolute path to the models folder
     */
    public static void sync(Path modelsRoot) {
        Map<String, Map<String, Object>> availableModels = new HashMap<>();
        
        File rootFile = modelsRoot.toFile();
        File[] tickers = rootFile.listFiles(File::isDirectory);
        if (tickers == null) return;

        for (File tickerDir : tickers) {
            if (tickerDir.getName().equals("quarantine")) continue;

            File[] models = tickerDir.listFiles(File::isDirectory);
            if (models == null) continue;

            for (File modelDir : models) {
                processAndValidateModel(tickerDir.getName(), modelDir, availableModels);
            }
        }

        saveToConfig(availableModels);
    }

    /**
     * Validate a single model folder by reading its `metadata.json`. If
     * the file is missing or invalid the folder is moved to quarantine.
     *
     * @param ticker parent ticker folder name
     * @param modelDir directory of the model to validate
     * @param availableModels accumulator map to add valid model info
     */
    private static void processAndValidateModel(String ticker, File modelDir, Map<String, Map<String, Object>> availableModels) {
        Path metaPath = modelDir.toPath().resolve(METADATA_FILE);
        String name = modelDir.getName();

        try {
            if (!Files.exists(metaPath)) throw new IOException("Metadata missing");

            JsonNode root = mapper.readTree(metaPath.toFile());
            JsonNode details = root.path("model_details");

            if (details.isMissingNode() || details.isEmpty()) {
                moveToQuarantine(modelDir);
                return;
            }

            // Extract data exactly like the Python dictionary
            Map<String, Object> info = new HashMap<>();
            info.put("ticker", ticker);
            info.put("trained_at", details.path("created_at").asText("unknown"));
            info.put("date_range", details.path("date_range").asText("unknown"));

            availableModels.put(name, info);

        } catch (IOException e) {
            System.err.println("Model " + name + " corrupted. Quarantining...");
            moveToQuarantine(modelDir);
        }
    }

    /**
     * Move a problematic model directory into the quarantine folder.
     *
     * @param modelDir directory to move
     */
    private static void moveToQuarantine(File modelDir) {
        try {
            Files.createDirectories(QUARANTINE_DIR);
            String folderName = modelDir.getParentFile().getName() + "_" + modelDir.getName();
            Files.move(modelDir.toPath(), QUARANTINE_DIR.resolve(folderName), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            System.err.println("Could not quarantine " + modelDir.getName());
        }
    }

    /**
     * Persist the discovered models map into the tickers JSON config and
     * ensure the `available_tickers` list is synchronized.
     *
     * @param availableModels mapping of modelName -> metadata map
     */
    private static void saveToConfig(Map<String, Map<String, Object>> availableModels) {
        try {
            File configFile = new File(CONFIG_PATH);
            ObjectNode root = (configFile.exists()) ? (ObjectNode) mapper.readTree(configFile) : mapper.createObjectNode();

            // Set the "models" object
            root.set("models", mapper.valueToTree(availableModels));

            // Sync the "available_tickers" list (unique set)
            Set<String> tickerSet = new HashSet<>();
            if (root.has("available_tickers")) {
                root.get("available_tickers").forEach(t -> tickerSet.add(t.asText()));
            }
            availableModels.values().forEach(m -> tickerSet.add((String) m.get("ticker")));

            root.set("available_tickers", mapper.valueToTree(tickerSet));

            // Write back to disk
            configFile.getParentFile().mkdirs();
            mapper.writerWithDefaultPrettyPrinter().writeValue(configFile, root);
            
        } catch (IOException e) {
            System.err.println("Failed to write tickers.json: " + e.getMessage());
        }
    }
}