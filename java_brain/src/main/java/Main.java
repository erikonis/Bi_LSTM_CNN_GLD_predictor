import java.util.logging.Logger;

import gui.MainGUI;
import gui.data.ModelSynchronizer;
import gui.data.Models;
import utils.Constants;
import utils.LogConfig;
import utils.ProjectPaths;

/**
 * Application entry point for the Java GUI and orchestration layer.
 *
 * Initializes loggers, model registry and launches the JavaFX UI.
 */
public class Main {
    public static void main(String[] args) {
        LogConfig.wipeLogs();
        LogConfig.createLogger(Constants.LOG_COMMS_NAME.getValue(), ProjectPaths.COMMS_LOG.getPath().toString());
        LogConfig.createLogger(Constants.LOG_BRAIN_NAME.getValue(), ProjectPaths.BRAIN_LOG.getPath().toString());
        Logger commsLogger = Logger.getLogger(Constants.LOG_COMMS_NAME.getValue());
        Logger brainLogger = Logger.getLogger(Constants.LOG_BRAIN_NAME.getValue());
        commsLogger.info("Java Comms System Started");
        brainLogger.info("Java Brain System Started");

        ModelSynchronizer.sync();
        Models.initialize();        

        /**
         * Launch the JavaFX application. This call blocks until the JavaFX
         * runtime has been initialized and is running.
         *
         * @param args command-line arguments forwarded to the JavaFX runtime
         */
        javafx.application.Application.launch(MainGUI.class, args); 
    }
}
