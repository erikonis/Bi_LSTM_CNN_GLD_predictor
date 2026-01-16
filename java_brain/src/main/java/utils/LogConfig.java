package utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LogConfig {
    public static void createLogger(String loggerName, String fileName) {
        Logger logger = Logger.getLogger(loggerName);
        try {
            // Disable inheriting handlers from the root logger
            // This prevents "Comms" logs from also going into the "Brain" file
            logger.setUseParentHandlers(false);

            // Create handler for this specific file
            FileHandler fh = new FileHandler(fileName, true);
            // Inside createLogger, replace SimpleFormatter with this:
            fh.setFormatter(new PrefixFormatter("JAVA-" + loggerName));
            logger.addHandler(fh);

            // Still add a ConsoleHandler so you can see it in VS Code
            ConsoleHandler ch = new ConsoleHandler();
            ch.setFormatter(new PrefixFormatter("JAVA-" + loggerName));

            logger.addHandler(ch);
            
            logger.setLevel(Level.INFO);
        } catch (IOException e) {
            System.err.println("Could not create logger " + loggerName + ": " + e.getMessage());
        }
    }

    public static void wipeOrCreateFile(String filePath) {
        try {
            Path path = Paths.get(filePath);
            
            // 1. Ensure the parent directories exist (very important!)
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }

            // 2. TRUNCATE_EXISTING wipes it; CREATE makes it if missing
            Files.write(path, new byte[0], 
                        StandardOpenOption.CREATE, 
                        StandardOpenOption.TRUNCATE_EXISTING);
            
            System.out.println("󰙨 Log file cleaned/created: " + filePath);
        } catch (Exception e) {
            System.err.println(" ERROR  Failed to clean log: " + e.getMessage());
        }
    }

    public static void wipeLogs(){
        wipeOrCreateFile(ProjectPaths.COMMS_LOG.getPath().toString());
        wipeOrCreateFile(ProjectPaths.BRAIN_LOG.getPath().toString());
        wipeOrCreateFile(ProjectPaths.PYTHON_BRAIN_LOG.getPath().toString());        
    }
}