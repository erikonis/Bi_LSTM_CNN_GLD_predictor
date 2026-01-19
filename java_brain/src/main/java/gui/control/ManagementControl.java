package gui.control;

import java.util.Set;

import gui.data.Models;
import static gui.styles.Constants.DEFAULT_BORDER;
import static gui.styles.Constants.RED_BORDER;
import gui.view.MainView;
import javafx.scene.control.Alert;

/**
 * ManagementControl provides helper methods for the management pane UI
 * such as populating selectors and validating user input.
 */
public class ManagementControl {
    private static final String NEUTRAL = "";

    /**
     * Populate the management pane selectors with available models and tickers.
     *
     * @param view the main view containing the selectors
     */
    public static void initSelectors(MainView view) {
        Set<String> allModels = Models.modelToTickers.keySet();
        Set<String> allTickers = Models.tickerToModels.keySet();

        view.managementModelSelector.getItems().setAll(allModels);
        view.managementTickerSelector.getItems().setAll(allTickers);
    }

    /**
     * Validate that a model is selected for deletion and provide visual
     * feedback by toggling the selector border style.
     *
     * @param view the main view containing management selectors
     * @return true if selection is valid
     */
    public static boolean validateDel(MainView view) {
        boolean isValid = true;

        // Validate Model Selector
        if (view.managementModelSelector.getValue() == null || view.managementModelSelector.getValue().equals(NEUTRAL)) {
            view.managementModelSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;
        } else {
            view.managementModelSelector.setStyle(DEFAULT_BORDER.getStyle());
        }
        return isValid;
    }

    /**
     * Validate that a ticker is selected for update and provide visual
     * feedback by toggling the selector border style.
     *
     * @param view the main view
     * @return true if selection is valid
     */
    public static boolean validateUpdate(MainView view) {
        boolean isValid = true;

        // Validate Ticker Selector
        if (view.managementTickerSelector.getValue() == null
                || view.managementTickerSelector.getValue().equals(NEUTRAL)) {
            view.managementTickerSelector.setStyle(RED_BORDER.getStyle());
            isValid = false;
        } else {
            view.managementTickerSelector.setStyle(DEFAULT_BORDER.getStyle());
        }
        return isValid;
    }

    /**
     * Display an error dialog.
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

}