package gui.control;

import java.util.HashMap;
import java.util.Map;

import control.PythonComms;
import gui.data.Models;
import static gui.styles.Constants.DEFAULT_BORDER;
import static gui.styles.Constants.RED_BORDER;
import gui.view.MainView;
import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.TextField;

public class TrainController {

    /**
     * TrainController handles collecting training parameters from the UI,
     * validating them, and delegating the training request to the Python
     * backend. It also shows completion/error dialogs.
     */

    
    /**
     * Initialize UI selectors with supported values.
     *
     * @param view the main view containing training controls
     */
    public static void init(MainView view) {
        view.trainDataSplitSelector.getItems().setAll("expand", "slide");
        view.trainModelSelector.getItems().setAll("PredictorBiLSTMcnn", "PredictorBiLSTMcnnA");
        view.trainTickerSelector.getItems().setAll("GLD");
    }

    
    /**
     * Validate the values entered in the training pane (selectors and
     * hyperparameters). Visual feedback is applied to invalid fields.
     *
     * @param view the main view
     * @return true if all required fields are valid
     */
    public static boolean validate(MainView view) {
        boolean isValid = true;

        // 1. Required Selectors
        if (view.trainModelSelector.getValue() == null) {
            view.trainModelSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;
        }
        if (view.trainTickerSelector.getValue() == null) {
            view.trainTickerSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;
        }

        // 2. String check
        if (view.fieldModelName.getText().trim().isEmpty()) {
            view.fieldModelName.setStyle(RED_BORDER.getStyle());
            isValid = false;
        }

        // 3. Integer Validations (Epochs, Batch, Hidden)
        isValid &= checkInt(view.fieldEpochs);
        isValid &= checkInt(view.fieldBatch);
        isValid &= checkInt(view.fieldHidden);

        // 4. Float Validations (LR, Dropout)
        isValid &= checkFloat(view.fieldLR);
        isValid &= checkFloat(view.fieldDropout);

        return isValid;
    }

    
    /**
     * Trigger a training run using the provided `PythonComms` bridge. The
     * UI is disabled while the training runs in a background thread and
     * re-enabled when the run completes.
     *
     * @param view main view with parameters
     * @param pythonComms bridge to the Python engine
     */
    public static void handle(MainView view, PythonComms pythonComms) {
        if (validate(view)) {
            // 1. Gather values from the UI (Safe to do on FX Thread here)
            Map<String, Object> trainingData = TrainController.collectTrainingParams(view);

            // 2. Disable UI
            view.btnStartTrain.setDisable(true);
            view.btnStartTrain.setText("Training in progress...");

            // 3. Start background execution
            new Thread(() -> {
                try {
                    boolean success = pythonComms.train(trainingData);

                    Platform.runLater(() -> {
                        if (success) {
                            showComplete("Complete", "Model training completed.");
                            Models.initialize(); // Ensure this is thread-safe or wrap if it touches UI
                        } else {
                            showError("Training Error", "The Python training script exited with an error.");
                        }
                    });
                } catch (Exception e) {
                    Platform.runLater(() -> {
                        showError("Execution Failed", e.getMessage());
                    });
                } finally {
                    // MUST be wrapped in runLater because we are in a background thread
                    Platform.runLater(() -> {
                        view.btnStartTrain.setDisable(false);
                        view.btnStartTrain.setText("Train");
                    });
                }
            }).start();
        }
    }

    private static boolean checkInt(TextField f) {
        try {
            int val = Integer.parseInt(f.getText());
            if (val <= 0)
                throw new Exception();
            f.setStyle(DEFAULT_BORDER.getStyle());
            return true;
        } catch (Exception e) {
            f.setStyle(RED_BORDER.getStyle());
            return false;
        }
    }

    private static boolean checkFloat(TextField f) {
        try {
            float val = Float.parseFloat(f.getText());
            if (val < 0)
                throw new Exception();
            f.setStyle(DEFAULT_BORDER.getStyle());
            return true;
        } catch (Exception e) {
            f.setStyle(RED_BORDER.getStyle());
            return false;
        }
    }


    /**
     * Collect and return a map of training parameters from the UI. The
     * returned map is ready for JSON serialization and forwarding to
     * the Python training endpoint.
     *
     * @param view main view
     * @return map of parameter name -> value
     */
    public static Map<String, Object> collectTrainingParams(MainView view) {
        Map<String, Object> params = new HashMap<>();

        // Required String/Choice arguments
        params.put("model", view.trainModelSelector.getValue());
        params.put("name", view.fieldModelName.getText());
        params.put("ticker", view.trainTickerSelector.getValue());
        params.put("data_split", view.trainDataSplitSelector.getValue());

        // Numeric arguments (Parsing confirmed by validate())
        params.put("epochs", Integer.parseInt(view.fieldEpochs.getText()));
        params.put("batch", Integer.parseInt(view.fieldBatch.getText()));
        params.put("hidden", Integer.parseInt(view.fieldHidden.getText()));
        params.put("lr", Float.parseFloat(view.fieldLR.getText()));
        params.put("dropout", Float.parseFloat(view.fieldDropout.getText()));

        // Boolean Flags
        params.put("auto_feat", view.cbAutoFeat.isSelected());
        params.put("early_stop", view.cbEarlyStop.isSelected());

        // Metadata
        params.put("info", view.fieldInfo.getText());

        return params;
    }

    /**
     * Show a blocking error dialog.
     *
     * @param title dialog title
     * @param message dialog message
     */
    public static void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }


    /**
     * Show a blocking information dialog indicating successful completion.
     *
     * @param title dialog title
     * @param message dialog message
     */
    public static void showComplete(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}