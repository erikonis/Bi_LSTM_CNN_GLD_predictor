import java.util.logging.Logger;

import gui.MainGUI;
import gui.data.Models;
import utils.Constants;
import utils.LogConfig;
import utils.ProjectPaths;

public class Main {
    public static void main(String[] args) {
        LogConfig.wipeLogs();
        LogConfig.createLogger(Constants.LOG_COMMS_NAME.getValue(), ProjectPaths.COMMS_LOG.getPath().toString());
        LogConfig.createLogger(Constants.LOG_BRAIN_NAME.getValue(), ProjectPaths.BRAIN_LOG.getPath().toString());
        Logger commsLogger = Logger.getLogger(Constants.LOG_COMMS_NAME.getValue());
        Logger brainLogger = Logger.getLogger(Constants.LOG_BRAIN_NAME.getValue());
        commsLogger.info("Java Comms System Started");
        brainLogger.info("Java Brain System Started");

        Models.initialize(ProjectPaths.MODEL_INFO.getPath().toString());        

        javafx.application.Application.launch(MainGUI.class, args); 

        //PythonComms comms = new PythonComms();
        //comms.test();
        //PythonComms.test_all();
    }
}
