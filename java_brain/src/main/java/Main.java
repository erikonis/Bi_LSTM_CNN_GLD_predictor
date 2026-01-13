import control.PythonComms;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        PythonComms comms = new PythonComms();
        comms.sendMessage("Hello from Java!");
    }
}
