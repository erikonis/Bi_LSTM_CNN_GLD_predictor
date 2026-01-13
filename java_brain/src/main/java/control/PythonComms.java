package control;

import java.io.IOException;
import java.net.ServerSocket;

public class PythonComms {


    public void sendMessage(String message) {
        // Implementation for sending a message to Python
        System.out.println("Sending message to Python: " + message);
    }

    public static ProcessBuilder createPythonProcessBuilder(String scriptPath) throws IOException{
        ServerSocket socket = new ServerSocket(0);
        int freePort = socket.getLocalPort();
        socket.close(); // Free it so Python can grab it

        ProcessBuilder pb = new ProcessBuilder("python", scriptPath, "--port", String.valueOf(freePort));
        return pb;
    }
}