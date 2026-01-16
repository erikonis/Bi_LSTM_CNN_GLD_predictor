package gui.control;

import java.util.List;

import gui.data.Models;
import static gui.styles.Constants.DEFAULT_BORDER;
import static gui.styles.Constants.RED_BORDER;
import gui.view.MainView;
import javafx.scene.control.TextField;

public class InferenceControl {
    private static final String NEUTRAL = "";
    private static boolean isUpdating = false;

    public static boolean validateInputs(MainView view) {
        boolean isValid = true;

        // Validate Ticker Selector
        if (view.tickerSelector.getValue() == null || view.tickerSelector.getValue().equals(NEUTRAL)) {
            view.tickerSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;

        } else {
            view.tickerSelector.setStyle(DEFAULT_BORDER.getStyle());
        }

        // Validate Model Selector
        if (view.modelSelector.getValue() == null || view.modelSelector.getValue().equals(NEUTRAL)) {
            view.modelSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;
        } else {
            view.modelSelector.setStyle(DEFAULT_BORDER.getStyle());
        }

        // List of all TextFields to check
        TextField[] fields = { view.fieldOpen, view.fieldHigh, view.fieldLow, view.fieldClose, view.fieldVolume };

        for (TextField field : fields) {
            if (field.getText().trim().isEmpty()) {
                field.setStyle(RED_BORDER.getStyle());
                isValid = false;
            } else {
                // Check if it's actually a valid number
                try {
                    Double.parseDouble(field.getText());
                    field.setStyle(DEFAULT_BORDER.getStyle());
                } catch (NumberFormatException e) {
                    field.setStyle(RED_BORDER.getStyle());
                    isValid = false;
                }
            }
        }

        return isValid;
    }

    public static void initDynamicSelectors(MainView view) {
        // Initial population
        resetModelList(view);
        resetTickerList(view);

        // 1. Model Selection Listener
        view.modelSelector.setOnAction(e -> {
            if (isUpdating) return;
            String selectedModel = view.modelSelector.getValue();

            isUpdating = true;
            if (selectedModel == null || selectedModel.equals(NEUTRAL)) {
                resetTickerList(view);
            } else {
                List<String> valid = Models.modelToTickers.get(selectedModel);
                
                // Rebuild list with Neutral always at the top
                view.tickerSelector.getItems().setAll(NEUTRAL);
                view.tickerSelector.getItems().addAll(valid);

                // Auto-select the first valid ticker
                if (!valid.isEmpty()) {
                    view.tickerSelector.getSelectionModel().select(valid.get(0));
                }
            }
            isUpdating = false;
        });

        // 2. Ticker Selection Listener
        view.tickerSelector.setOnAction(e -> {
            if (isUpdating) return;
            String selectedTicker = view.tickerSelector.getValue();

            isUpdating = true;
            if (selectedTicker == null || selectedTicker.equals(NEUTRAL)) {
                resetModelList(view);
            } else {
                List<String> valid = Models.tickerToModels.get(selectedTicker);
                
                // Rebuild list with Neutral always at the top
                view.modelSelector.getItems().setAll(NEUTRAL);
                view.modelSelector.getItems().addAll(valid);

                // Auto-select the first valid model
                if (!valid.isEmpty()) {
                    view.modelSelector.getSelectionModel().select(valid.get(0));
                }
            }
            isUpdating = false;
        });
    }

    // Helper methods to ensure the Neutral option is always there
    private static void resetModelList(MainView view) {
        view.modelSelector.getItems().setAll(NEUTRAL);
        view.modelSelector.getItems().addAll(Models.modelToTickers.keySet());
    }

    private static void resetTickerList(MainView view) {
        view.tickerSelector.getItems().setAll(NEUTRAL);
        view.tickerSelector.getItems().addAll(Models.tickerToModels.keySet());
    }
}
