package gui.data;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import utils.Constants;
import utils.ProjectPaths;

public class Models {
    private static final Logger logger = Logger.getLogger(Constants.LOG_BRAIN_NAME.getValue());

    private static final ObjectMapper mapper = new ObjectMapper();

    public static Map<String, List<String>> modelToTickers;
    public static Map<String, List<String>> tickerToModels;
    private static JsonNode rootNode;

    /**
     * Initialize the model mappings using the default model info path.
     */
    public static void initialize() {
        initialize(ProjectPaths.MODEL_INFO.getPath().toString());
    }

    /**
     * Load model metadata from the provided JSON file and build mapping
     * structures used by the UI.
     *
     * @param jsonPath path to the `tickers.json` metadata file
     */
    public static void initialize(String jsonPath) {
        modelToTickers = new HashMap<>();
        tickerToModels = new HashMap<>();
        load(jsonPath);
        reverseMapping();
    }

    /**
     * Read the JSON model registry and populate `modelToTickers`.
     *
     * @param jsonPath path to JSON model registry file
     */
    private static void load(String jsonPath) {
        try {
            File file = new File(jsonPath);
            if (!file.exists()) {
                logger.warning("Model registry file not found at: " + jsonPath);
                return;
            }

            rootNode = mapper.readTree(file);
            modelToTickers.clear();

            JsonNode modelsNode = rootNode.path("models");

            // Iterate over all field names (model keys like "bilstmcnn")
            Iterator<Map.Entry<String, JsonNode>> fields = modelsNode.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                String modelKey = entry.getKey();
                String ticker = entry.getValue().path("ticker").asText();

                // Populate the modelToTickers map
                modelToTickers.computeIfAbsent(modelKey, k -> new ArrayList<>()).add(ticker);
            }

            logger.info("Model Registry loaded. Found " + modelToTickers.size() + " models.");

        } catch (IOException e) {
            logger.severe("Failed to read model JSON: " + e.getMessage());
        }
    }

    // Helper method to get the training date for a specific model
    public static String getTrainedDate(String modelKey) {
        JsonNode node = rootNode.path("models").path(modelKey).path("trained");
        if (node.isMissingNode())
            return "Unknown";

        try {
            // Parse the ISO string (e.g., 2026-01-14T21:18:55.647552)
            LocalDateTime dateTime = LocalDateTime.parse(node.asText());
            // Format to the desired pattern
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
            return dateTime.format(formatter);
        } catch (Exception e) {
            return node.asText(); // Fallback to raw string if parsing fails
        }
    }

    /**
     * Return the date range listed in the model metadata for the provided model.
     *
     * @param modelKey model identifier
     * @return list of date strings or empty list
     */
    public static List<String> getDateRange(String modelKey) {
        List<String> range = new ArrayList<>();
        JsonNode rangeNode = rootNode.path("models").path(modelKey).path("date_range");

        if (rangeNode.isArray()) {
            for (JsonNode date : rangeNode) {
                range.add(date.asText());
            }
        }
        return range;
    }

    /**
     * Build the reverse lookup from ticker -> models using `modelToTickers`.
     */
    private static void reverseMapping() {
        for (var entry : modelToTickers.entrySet()) {
            for (String ticker : entry.getValue()) {
                tickerToModels.computeIfAbsent(ticker, k -> new ArrayList<>()).add(entry.getKey());
            }
        }
    }

    /**
     * Deletes a model from memory and synchronizes both mapping directions.
     * 
     * @param modelKey The unique name/key of the model to remove.
     */
    /**
     * Delete a model from the in-memory registry and remove reverse links.
     * This does not delete files on disk; callers should coordinate physical
     * deletion if desired.
     *
     * @param modelKey The unique name/key of the model to remove.
     */
    public static void deleteModel(String modelKey) {
        if (modelKey == null || !modelToTickers.containsKey(modelKey)) {
            logger.warning("Attempted to delete non-existent model from memory: " + modelKey);
            return;
        }

        // 1. Get the list of tickers associated with this model before removing it
        List<String> associatedTickers = modelToTickers.get(modelKey);

        // 2. Remove from tickerToModels (the reverse map)
        if (associatedTickers != null) {
            for (String ticker : associatedTickers) {
                List<String> modelsForTicker = tickerToModels.get(ticker);
                if (modelsForTicker != null) {
                    // Remove the specific model from this ticker's list
                    modelsForTicker.remove(modelKey);

                    // Optional: If no models are left for this ticker, remove the ticker entry
                    if (modelsForTicker.isEmpty()) {
                        tickerToModels.remove(ticker);
                    }
                }
            }
        }

        // 3. Finally, remove from the primary modelToTickers map
        modelToTickers.remove(modelKey);

        logger.info("Model " + modelKey + " successfully removed from memory registry.");
    }
}
