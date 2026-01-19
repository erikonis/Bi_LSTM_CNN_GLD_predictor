package control;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import utils.Constants;
import utils.ProjectPaths;

/**
 * PythonService manages the lifecycle and communication channel to the
 * external Python engine. It starts the Python process, allocates TCP
 * ports for RPC and heartbeat, and provides methods to send commands and
 * monitor liveness.
 */
public class PythonService {
    private Process pythonProcess;
    private int port;
    private int heartPort;
    private final String scriptPath;
    private static final Logger logger = Logger.getLogger(Constants.LOG_COMMS_NAME.getValue());
    private final ScheduledExecutorService heartbeat = Executors.newSingleThreadScheduledExecutor();
    private static final ObjectMapper mapper = new ObjectMapper();
    private boolean connected = false;

    /**
     * Create a PythonService bound to the given Python script path.
     *
     * @param scriptPath filesystem path to the Python entry script to run
     */
    public PythonService(String scriptPath) {
        this.scriptPath = scriptPath;
    }

    /**
     * Start the Python process and allocate dynamic ports for RPC and
     * heartbeat. The method will launch the Python interpreter from the
     * project's virtual environment and begin the heartbeat warmup.
     *
     * @throws IOException if the process cannot be started or ports cannot
     *                     be allocated
     */
    public void start() throws IOException {
        // 1. Find a free port
        try (ServerSocket socket = new ServerSocket(0);
                ServerSocket socket2 = new ServerSocket(0)) {
            this.port = socket.getLocalPort();
            this.heartPort = socket2.getLocalPort();
        }
        logger.info("Allocated port " + this.port + " for Python service.");
        // 2. Start the process
        logger.info("Allocated heartbeat port " + this.heartPort + " for Python service.");

        File scriptFile = new File(this.scriptPath).getAbsoluteFile();

        Path venvPath = ProjectPaths.PYTHON_CWD.getPath().resolve(".venv/Scripts/python.exe");

        List<String> command = new ArrayList<>();
        command.add(venvPath.toString());
        command.add(scriptFile.getPath());
        command.add("--port");
        command.add(String.valueOf(this.port));
        command.add("--hport");
        command.add(String.valueOf(this.heartPort));

        ProcessBuilder pb = new ProcessBuilder(command);

        Map<String, String> env = pb.environment();
        env.put("PYTHONPATH", ProjectPaths.PYTHON_CWD.getPath().toString());

        pb.inheritIO(); // So you can see Python's print statements in Java console
        this.pythonProcess = pb.start();

        logger.info("Python service started on port: " + this.port);

        startHeartBeat(90);

    }

    private void startHeartBeat(int maxWarmupSeconds) {
        AtomicInteger failureCount = new AtomicInteger(0);
        int maxTries = 3;

        // Use a separate thread to handle the warmup so we don't block the main Java
        // GUI thread
        CompletableFuture.runAsync(() -> {
            logger.info("Entering warmup phase. Waiting for Python to bind to port " + this.heartPort + "...");

            this.connected = false;
            long startTime = System.currentTimeMillis();
            long timeoutMs = maxWarmupSeconds * 1000L;

            // 1. ACTIVE POLLING LOOP
            while (!this.connected && (System.currentTimeMillis() - startTime < timeoutMs)) {
                if (pythonProcess == null || !pythonProcess.isAlive()) {
                    logger.severe("Python process died during warmup.");
                    stop();
                    return;
                }

                try (Socket s = new Socket()) {
                    // Try to connect with a very short timeout
                    s.connect(new java.net.InetSocketAddress("127.0.0.1", this.heartPort), 500);
                    this.connected = true;
                    logger.info("Python is READY. Connection established after " +
                            (System.currentTimeMillis() - startTime) / 1000 + "s");
                } catch (IOException e) {
                    // Not ready yet, wait 1s before trying again
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {
                    }
                }
            }

            if (!this.connected) {
                logger.severe("Python failed to start within " + maxWarmupSeconds + " seconds. Aborting.");
                stop();
                return;
            }

            // 2. START THE REGULAR SCHEDULED HEARTBEAT
            heartbeat.scheduleAtFixedRate(() -> {
                if (pythonProcess == null || !pythonProcess.isAlive()) {
                    logger.severe("Python OS process is no longer running.");
                    stop();
                    return;
                }

                try (Socket s = new Socket()) {
                    s.connect(new java.net.InetSocketAddress("127.0.0.1", this.heartPort), 2000);
                    s.setSoTimeout(2000);

                    try (PrintWriter out = new PrintWriter(s.getOutputStream(), true);
                            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()))) {
                        // System.out.println("ping");
                        out.println("{\"status\": \"ping\"}");
                        String response = in.readLine();

                        if (response != null && response.contains("pong")) {
                            failureCount.set(0);
                        } else {
                            handleHeartbeatFailure(failureCount, maxTries);
                        }
                    }
                } catch (IOException e) {
                    handleHeartbeatFailure(failureCount, maxTries);
                }
            }, 0, 10, TimeUnit.SECONDS); // Start immediately (0) since we are already connected
        });
    }

    private void handleHeartbeatFailure(AtomicInteger failureCount, int maxTries) {
        int current = failureCount.incrementAndGet();
        logger.warning("Heartbeat failure (" + current + "/" + maxTries + ")");
        if (current >= maxTries) {
            logger.severe("Python heartbeat failed limit. Terminating...");
            stop();
        }
    }

    /**
     * Send a JSON-RPC style command to the Python service and wait for a
     * JSON response. This method blocks until a response is received or
     * the configured timeout elapses. Retries are attempted on transient
     * failures.
     *
     * @param action the command/action name understood by the Python side
     * @param dataJson the JSON-encoded payload to send as the "data" field
     * @param timeout timeout in seconds to wait for the Python reply
     * @return parsed JsonNode from the Python response or an error node on failure
     */
    public synchronized JsonNode sendCommand(String action, String dataJson, int timeout) {
        while (!this.connected) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException ignored) {
            }
        }

        int maxRetries = 3;
        for (int i = 0; i < maxRetries; i++) {
            try (Socket socket = new Socket("127.0.0.1", this.port);
                    PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

                socket.setSoTimeout(timeout * 1000); // Java-side timeout: Wait 5s for Python to reply

                String payload = String.format("{\"action\": \"%s\", \"data\": %s}", action, dataJson);
                out.println(payload);
                String rawResponse = in.readLine();
                if (rawResponse != null) {
                    return mapper.readTree(rawResponse);
                } else {

                    return createErrorNode("Python Process closed itself unexpectedly upon receiving the Command.");
                }

            } catch (IOException e) {
                if (i == maxRetries - 1) {
                    logger.severe("Final attempt failed: " + e.getMessage());

                    return createErrorNode("Attempts to send command exceeded.");

                } else {
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException ignored) {
                    }
                }
            }
        }

        return createErrorNode("Connection failed");
    }

    /**
     * Convenience overload sending a command using a default timeout of 5 seconds.
     *
     * @param action the command/action name
     * @param dataJson the JSON payload for the command
     * @return response JsonNode from Python or an error node
     */
    public JsonNode sendCommand(String action, String dataJson) {
        return sendCommand(action, dataJson, 5);
    }

    /**
     * Gracefully stop the Python service: stop heartbeat, send shutdown
     * command to Python and destroy the process.
     */
    public void stop() {
        heartbeat.shutdownNow(); // Stop the heartbeat first
        if (pythonProcess != null) {
            sendCommand("shutdown", "{}");
            pythonProcess.destroy();
        }
    }

    /**
     * Forcefully terminate the Python process. On Windows this uses
     * `taskkill` to kill the process tree; on other OSes it invokes
     * `destroyForcibly()`.
     */
    public void forcekill() {
        heartbeat.shutdownNow(); // Stop the heartbeat first
        if (pythonProcess != null && pythonProcess.isAlive()) {
            try {
                // Windows-specific: Kill the process and all its children (/T) forcibly (/F)
                if (System.getProperty("os.name").toLowerCase().contains("win")) {
                    long pid = pythonProcess.pid();
                    Runtime.getRuntime().exec("taskkill /F /T /PID " + pid);
                } else {
                    pythonProcess.destroyForcibly();
                }
            } catch (IOException e) {
                pythonProcess.destroyForcibly();
            }
        }
    }

    /**
     * @return the RPC port assigned to the Python service
     */
    public int getPort() {
        return port;
    }

    /**
     * @return the heartbeat port assigned to the Python service
     */
    public int getHeartPort() {
        return heartPort;
    }

    private JsonNode createErrorNode(String message) {
        return this.mapper.createObjectNode()
                .put("status", "error")
                .put("message", message);
    }
}