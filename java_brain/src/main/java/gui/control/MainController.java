package gui.control;

import java.util.ArrayList;
import java.util.Optional;

import control.PythonComms;
import static gui.styles.Constants.DEFAULT_BORDER;
import gui.view.MainView;
import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextField;

public class MainController {
    private MainView view;
    private PythonComms pythonComms;

    public MainController(MainView view, PythonComms pythonComms) {
        this.view = view;
        this.pythonComms = pythonComms;
        initHandlers();
    }
    
    private void initHandlers() {
        // Navigation
        view.itemInfer.setOnAction(e -> view.showInfer());
        view.itemTrain.setOnAction(e -> view.showTrain());
        view.itemSettings.setOnAction(e -> view.showSettings());
        InferenceControl.initDynamicSelectors(view);
        fieldsTypeStyleHandle();
        // Prediction Logic
        view.btnPredict.setOnAction(e -> handlePrediction());
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
                            //response.set(i, Math.round(response.get(i) * 10000.0f) / 10000.0f);
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

    public void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    public boolean showConfirmation(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle(title);
        alert.setContentText(message);
        Optional<ButtonType> result = alert.showAndWait();
        return result.isPresent() && result.get() == ButtonType.OK;
    }
}