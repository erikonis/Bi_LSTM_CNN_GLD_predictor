package gui;

import control.PythonComms;
import gui.control.MainController;
import gui.view.MainView;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class MainGUI extends Application {
    private PythonComms pythonComms;

    @Override
    public void start(Stage primaryStage) {
        // Assume path is provided or handled via Main.java
        pythonComms = new PythonComms();

        MainView view = new MainView();
        MainController controller = new MainController(view, pythonComms);

        // 3:1 Ratio (e.g., 900x300)
        Scene scene = new Scene(view, 600, 300);
        primaryStage.setTitle("Financial Predictor");
        primaryStage.setScene(scene);
        primaryStage.setResizable(false);

        // Clean Shutdown
        primaryStage.setOnCloseRequest(event -> {
            if (controller.showConfirmation("Exit", "Are you sure you want to exit?")) {
                pythonComms.shutdown();
                System.exit(0);
            } else {
                event.consume(); // Cancel exit
            }
        });

        primaryStage.show();
    }
}