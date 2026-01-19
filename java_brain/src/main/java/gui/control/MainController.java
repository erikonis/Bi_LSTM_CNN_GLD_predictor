package gui.control;

import java.util.ArrayList;
import java.util.Optional;

import control.PythonComms;
import gui.data.Models;
import static gui.styles.Constants.DEFAULT_BORDER;
import gui.view.MainView;
import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextField;

/**
 * MainController wires the `MainView` UI to application actions. It
 * handles navigation, validation, prediction requests and coordinating
 * other controllers (Management, Inference, Train).
 */
public class MainController {
    private MainView view;
    private PythonComms pythonComms;

    /**
     * Create a controller for the provided view and Python communication bridge.
     *
     * @param view main UI view
     * @param pythonComms bridge to the Python engine
     */
    public MainController(MainView view, PythonComms pythonComms) {
        this.view = view;
        this.pythonComms = pythonComms;
        ManagementControl.initSelectors(view);
        TrainController.init(view);
        initHandlers();
    }

    private void initHandlers() {
        // Navigation
        view.itemInfer.setOnAction(e -> view.showInfer());
        view.itemTrain.setOnAction(e -> view.showTrain());
        view.itemManagement.setOnAction(e -> view.showManagement());
        InferenceControl.initDynamicSelectors(view);
        fieldsTypeStyleHandle();
        // Prediction Logic
        view.btnPredict.setOnAction(e -> handlePrediction());
        view.btnModelDel.setOnAction(e -> handleDelete(view));
        view.btnTickerUpdate.setOnAction(e -> handleUpdate());
        view.btnStartTrain.setOnAction(e -> handleTraining());
    }

    /**
     * Handle a model deletion request initiated from the management pane.
     * The method validates selection, removes the model from the UI
     * registry and forwards the delete command to Python.
     *
     * @param view reference to the main view (used for selection widgets)
     */
    public void handleDelete(MainView view) {
        String selectedModel = view.managementModelSelector.getValue();
        if (ManagementControl.validateDel(view)) {
            Models.deleteModel(selectedModel);
            refreshModels();
            pythonComms.deleteModel(selectedModel);
        }
        //showError("Partially implemented", "This feature is not fully implemented yet. For now the model is just deleted from the application's memory, but not physically.");
    }

    /**
     * Trigger an update operation for the selected ticker. Delegates to
     * Python and shows an error dialog on failure.
     */
    public void handleUpdate() {
        //showError("Not Implemented", "Ticker update functionality is not yet implemented.");
        String selectedTicker = this.view.managementTickerSelector.getValue();
        if (ManagementControl.validateUpdate(this.view)) {
            if(!pythonComms.update(selectedTicker)){
                showError("Update Failed", "Failed to update ticker data. Check logs for details.");
            }
        }
    }

    /**
     * Refresh model selectors across the UI after model list changes.
     */
    public void refreshModels(){
        InferenceControl.refreshSelectors(view);
        ManagementControl.initSelectors(view);
    }

    private void fieldsTypeStyleHandle() {
        TextField[] fields = { view.fieldOpen, view.fieldHigh, view.fieldLow, view.fieldClose, view.fieldVolume };
        for (TextField field : fields) {
            field.textProperty().addListener((obs, oldVal, newVal) -> {
                if (!newVal.isEmpty()) {
                    field.setStyle(DEFAULT_BORDER.getStyle());
                }
            });
        }

        view.tickerSelector.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null)
                view.tickerSelector.setStyle(DEFAULT_BORDER.getStyle());
        });

        view.modelSelector.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null)
                view.modelSelector.setStyle(DEFAULT_BORDER.getStyle());
        });

         view.managementModelSelector.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null)
                view.managementModelSelector.setStyle(DEFAULT_BORDER.getStyle());
        });

         view.managementTickerSelector.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null)
                view.managementTickerSelector.setStyle(DEFAULT_BORDER.getStyle());
        });
    }

    private void handlePrediction() {
        view.fieldOutput.setText("");
        if (!InferenceControl.validateInputs(view)) {
            return;
        }

        view.fieldOutput.setText("Predicting...");
        view.btnPredict.setDisable(true);

        // Run in a thread so the UI doesn't freeze
        new Thread(() -> {
            try {

                String ticker = view.tickerSelector.getValue();
                String model = view.modelSelector.getValue();
                float o = Float.parseFloat(view.fieldOpen.getText());
                float h = Float.parseFloat(view.fieldHigh.getText());
                float l = Float.parseFloat(view.fieldLow.getText());
                float c = Float.parseFloat(view.fieldClose.getText());
                long v = Long.parseLong(view.fieldVolume.getText());

                ArrayList<Float> response = pythonComms.getPrediction(ticker, model, o, h, l, c, v);
                Platform.runLater(() -> {
                    if (response != null && !response.isEmpty()) {
                        String str = "";
                        for (int i = 0; i < response.size(); i++) {
                            if (i == 0) {
                                str += String.format("%.2f", response.get(i));
                                continue;
                            }
                            str += ", " + String.format("%.2f", response.get(i));
                            // Round to 4 decimal places
                            // response.set(i, Math.round(response.get(i) * 10000.0f) / 10000.0f);
                        }

                        view.fieldOutput.setText(str);
                    } else {
                        showError("Prediction Failed",
                                "No prediction received from Python. Check logs for details.");
                        view.fieldOutput.setText("Error");
                    }
                });

            } catch (Exception ex) {
                Platform.runLater(() -> {
                    showError("Connection Error", ex.getMessage());
                    view.fieldOutput.setText("");
                });
            } finally {
                Platform.runLater(() -> {
                    view.btnPredict.setDisable(false);
                });
            }
        }).start();
    }

    private void handleTraining() {
        TrainController.handle(view, pythonComms);
        refreshModels();
    }

    /**
     * Display a blocking error dialog to the user.
     *
     * @param title dialog title
     * @param message dialog message body
     */
    public void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    /**
     * Show a confirmation dialog and return whether the user accepted.
     *
     * @param title dialog title
     * @param message dialog message
     * @return true if the user confirmed, false otherwise
     */
    public boolean showConfirmation(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle(title);
        alert.setContentText(message);
        Optional<ButtonType> result = alert.showAndWait();
        return result.isPresent() && result.get() == ButtonType.OK;
    }
}