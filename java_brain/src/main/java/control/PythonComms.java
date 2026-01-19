package control;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;

import utils.Constants;
import utils.ProjectPaths;

/**
 * PythonComms is a high-level client wrapper that exposes blocking,
 * Java-friendly methods to interact with the Python engine via
 * `PythonService`. It translates method arguments into JSON payloads
 * and parses JSON responses into Java structures.
 */
public class PythonComms {
    private static final Logger logger = Logger.getLogger(Constants.LOG_COMMS_NAME.getValue());

    private final PythonService service;

    /**
     * Initialize the comms wrapper and start the underlying Python service.
     */
    public PythonComms() {
        this.service = new PythonService(ProjectPaths.BRIDGE.getPath().toString());
        try {
            this.service.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Run a series of smoke tests against the Python engine. This is a
     * convenience method intended for manual debugging and verification.
     */
    public static void test_all() {
        PythonComms comms = new PythonComms();
        logger.info("=== PERFORMING TEST ===");
        try {
            logger.info("=TEST= Starting Python Comms Test");
            JsonNode response = comms.test();
            logger.info("=TEST= Test from Python: " + response.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
        try{
            logger.info("=TEST= Starting Python Prediction Test");
            
            ArrayList<Float> response = comms.getPrediction("GLD", "predictor_bilstmCNN_GLD", 310, 320, 300, 315, 1000000);
            if (response != null && !response.isEmpty()) {
                logger.info("=TEST= Prediction from Python: " + response.toString());
            } else {
                logger.warning("=TEST= No prediction received from Python.");
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        try {
            logger.info("=TEST= Starting Python Required Inputs Test");
            ArrayList<ArrayList<String>> response = comms.getRequiredInputs("GLD", "predictor_bilstmCNN_GLD");
            if (response != null && response.size() == 2) {
                logger.info("=TEST= Market Inputs from Python: " + response.get(0).toString());
                logger.info("=TEST= Sentiment Inputs from Python: " + response.get(1).toString());
            } else {
                logger.warning("=TEST= No required inputs received from Python.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Request a basic health/test response from Python.
     *
     * @return JsonNode payload returned by the Python test endpoint
     */
    public JsonNode test() {
        return service.sendCommand("test", "{}");
    }

    /**
     * Request a prediction from the Python model for a single OHLCV sample.
     *
     * @param ticker ticker symbol
     * @param model model identifier (metadata folder name)
     * @param o open price
     * @param h high price
     * @param l low price
     * @param c close price
     * @param v volume
     * @return list of float predictions or null on failure
     */
    public ArrayList<Float> getPrediction(String ticker, String model, double o, double h, double l, double c, long v) {
        // Construct the nested data JSON
        String dataJson = String.format(
                "{\"ticker\":\"%s\", \"model\":\"%s\", \"open\":%f, \"high\":%f, \"low\":%f, \"close\":%f, \"volume\":%d}",
                ticker, model, o, h, l, c, v);

        JsonNode response = service.sendCommand("infer", dataJson, 30);
        if (response.get("status").asText().equals("success")) {
            JsonNode predictions = response.get("predictions");

            if (predictions != null && predictions.isArray()) {
                ArrayList<Float> result = new ArrayList<>();
                for (JsonNode prediction : predictions) {
                    result.add(prediction.floatValue());
                }
                return result;
            }
        }
        return null;
    }

    /**
     * Query the Python engine for the required market and sentiment input
     * column names for a given ticker and model.
     *
     * @param ticker ticker symbol
     * @param model model identifier
     * @return nested list where index 0 is market columns and index 1 is sentiment columns
     */
    public ArrayList<ArrayList<String>> getRequiredInputs(String ticker, String model) {
        String dataJson = String.format(
                "{\"ticker\":\"%s\", \"model\":\"%s\"}",
                ticker, model);

        JsonNode response = service.sendCommand("get_inputs", dataJson);

        if (response.get("status").asText().equals("success")) {
            JsonNode marketCols = response.get("market_cols");
            JsonNode sentimentCols = response.get("sentiment_cols");

            if (marketCols != null && marketCols.isArray()) {
                ArrayList<ArrayList<String>> result = new ArrayList<>();
                ArrayList<String> marketList = new ArrayList<>();
                for (JsonNode col : marketCols) {
                    marketList.add(col.asText());
                }
                result.add(marketList);
                ArrayList<String> sentimentList = new ArrayList<>();
                for (JsonNode col : sentimentCols) {
                    sentimentList.add(col.asText());
                }
                result.add(sentimentList);
                return result;
            }
        }
        return null;
    }

    /**
     * Trigger a training run in the Python engine. The provided map is
     * serialized into a flat JSON object and passed to the Python trainer.
     *
     * @param trainingData training configuration and metadata
     * @return true if Python reported success, false otherwise
     */
    public boolean train(Map<String, Object> trainingData) {
        // Convert trainingData map to JSON string
        StringBuilder dataJsonBuilder = new StringBuilder("{");
        for (Map.Entry<String, Object> entry : trainingData.entrySet()) {
            dataJsonBuilder.append(String.format("\"%s\":\"%s\",", entry.getKey(), entry.getValue().toString()));
        }
        // Remove trailing comma and close JSON
        if (dataJsonBuilder.length() > 1) {
            dataJsonBuilder.setLength(dataJsonBuilder.length() - 1);
        }
        dataJsonBuilder.append("}");
        String dataJson = dataJsonBuilder.toString();
        JsonNode response = service.sendCommand("train", dataJson, 300);
        System.out.println(response.toString());
        return response.get("status").asText().equals("success");
    }

    /**
     * Request deletion of a model from the Python-side model registry.
     *
     * @param modelName name of the model metadata folder to delete
     * @return true on success
     */
    public boolean deleteModel(String modelName){
        String dataJson = String.format("{\"model_name\":\"%s\"}", modelName);
        JsonNode response = service.sendCommand("delete_model", dataJson, 30);
        return response.get("status").asText().equals("success");
    }

    /**
     * Gracefully stop the underlying Python service.
     */
    public void shutdown() {
        service.stop();
    }
    
    /**
     * Forcefully terminate the underlying Python process.
     */
    public void shutdownForcefully() {
        service.forcekill();
    }

    /**
     * Request the Python engine to update dataset(s) for a given ticker.
     *
     * @param ticker ticker symbol to update
     * @return true if the update was reported as successful
     */
    public boolean update(String ticker){
        String dataJson = String.format("{\"ticker\":\"%s\"}", ticker);
        JsonNode response = service.sendCommand("update_ticker", dataJson, 120);
        return response.get("status").asText().equals("success");
    }

}