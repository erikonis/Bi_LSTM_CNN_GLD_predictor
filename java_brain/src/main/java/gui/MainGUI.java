package gui;

import java.util.Optional;

import control.PythonComms;
import gui.control.MainController;
import gui.view.MainView;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

/**
 * MainGUI is the JavaFX application entry point. It initializes the
 * `PythonComms` bridge, constructs the main view and controller, and
 * manages clean shutdown behavior.
 */
public class MainGUI extends Application {
    private PythonComms pythonComms;

    @Override
    /**
     * Initialize and display the JavaFX primary stage with the main view.
     *
     * @param primaryStage primary JavaFX stage provided by the runtime
     */
    public void start(Stage primaryStage) {
        // Assume path is provided or handled via Main.java
        pythonComms = new PythonComms();

        MainView view = new MainView();
        MainController controller = new MainController(view, pythonComms);

        // 3:1 Ratio (e.g., 900x300)
        Scene scene = new Scene(view, 600, 500);
        primaryStage.setTitle("Financial Predictor");
        primaryStage.setScene(scene);
        primaryStage.setResizable(false);

        // Clean Shutdown
        primaryStage.setOnCloseRequest(event -> {
            // Prevent the window from closing immediately
            event.consume();

            // Show our custom confirmation
            showExitConfirmation();
        });
        primaryStage.show();
    }

    /**
     * Show a confirmation dialog to the user when closing the application.
     * Offers a "force shutdown" option which kills the Python process
     * immediately.
     *
     * @return true if the user confirmed exit, false otherwise
     */
    public boolean showExitConfirmation() {
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle("Exit Application");
        alert.setHeaderText("Confirm Shutdown");
        alert.setContentText("Are you sure you want to exit? This will stop the Python engine.");

        // Create the "Force Shutdown" CheckBox
        CheckBox forceShutdownTick = new CheckBox("Force shutdown (Kill process immediately)");
        forceShutdownTick.setSelected(true); // Default is ticked as requested

        // Add the checkbox to the dialog's layout
        VBox content = new VBox(10, new Label("Options:"), forceShutdownTick);
        alert.getDialogPane().setExpandableContent(content);
        alert.getDialogPane().setExpanded(true); // Ensure it's visible immediately

        Optional<ButtonType> result = alert.showAndWait();

        if (result.isPresent() && result.get() == ButtonType.OK) {
            handleApplicationExit(forceShutdownTick.isSelected());
            return true;
        }
        return false;
    }

    /**
     * Perform the actual shutdown flow: stop or kill Python side then
     * terminate the JavaFX application.
     *
     * @param force if true, forcefully terminate the Python process
     */
    private void handleApplicationExit(boolean force) {
        System.out.println("Initiating shutdown...");
        if (force) {
            pythonComms.shutdownForcefully();
        } else {
            pythonComms.shutdown();
        }
        // Close the Java application
        Platform.exit();
        System.exit(0);
    }
}