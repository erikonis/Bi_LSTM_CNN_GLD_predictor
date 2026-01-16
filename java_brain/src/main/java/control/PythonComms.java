package control;

import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;

import utils.Constants;
import utils.ProjectPaths;

public class PythonComms {
    private static final Logger logger = Logger.getLogger(Constants.LOG_COMMS_NAME.getValue());

    private final PythonService service;

    public PythonComms() {
        this.service = new PythonService(ProjectPaths.BRIDGE.getPath().toString());
        try {
            this.service.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

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

    public JsonNode test() {
        return service.sendCommand("test", "{}");
    }

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

    public void shutdown() {
        service.stop();
    }

}