import socket
import json
import threading
import time
import argparse
from src.python_engine.inference.inference import ModelInference
from src.utils.paths import LOG_COMMS_PATH, LOG_BRAIN_PATH
from src.utils.logger import setup_logging

loggerComms = setup_logging("COMMS", LOG_COMMS_PATH)
loggerBrain = setup_logging("BRAIN", LOG_BRAIN_PATH)
loggerBrain.info("Java Listener started.")

class JavaBridge:
    def __init__(self, port, hport, timeout=30):
        self.host = '127.0.0.1'
        self.port = port
        self.hport = hport
        self.timeout = timeout
        self.running = True

    def start(self):

        h_thread = threading.Thread(target=self.run_heartbeat_listener, daemon=True)
        h_thread.start()
        self.run_command_listener()

    def run_heartbeat_listener(self):
        """Dedicated thread that shuts down the bridge if no ping is received for 15s."""
        last_ping_time = time.time() # Track the last successful ping
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.hport))
            s.listen(5)
            # We set a short accept timeout so we can check the 'last_ping_time' frequently
            s.settimeout(1.0) 
            
            loggerComms.info(f"Heartbeat listener active on port {self.hport}")
            
            while self.running:
                # Check: Has it been more than 15 seconds since the last successful ping?
                if time.time() - last_ping_time > 15:
                    loggerComms.error("Heartbeat timeout: No valid ping for 15s. Shutting down bridge.")
                    self.running = False
                    break

                try:
                    conn, _ = s.accept()
                    with conn:
                        conn.settimeout(2.0) # Timeout for reading the actual data
                        raw_data = conn.recv(1024).decode('utf-8')
                        
                        if raw_data:
                            try:
                                data = json.loads(raw_data)
                                if data.get("status") == "ping" or data.get("action") == "ping":
                                    response = {"status": "pong"}
                                    #print("pong")
                                    conn.sendall(json.dumps(response).encode('utf-8'))
                                    last_ping_time = time.time() # SUCCESS: Reset the clock
                            except json.JSONDecodeError:
                                loggerComms.warning("Received non-JSON data on heartbeat port.")
                                
                except socket.timeout:
                    # This is normal. It just means no one called in the last 1.0s.
                    # The loop will restart and check the 15s 'last_ping_time' logic.
                    continue
                except Exception as e:
                    if self.running:
                        loggerComms.error(f"Heartbeat Thread Error: {e}")

    def run_command_listener(self):
        """Main thread listener for heavy tasks (Inference/Training)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            loggerComms.info(f"Command listener active on port {self.port}")

            while self.running:
                try:
                    conn, addr = s.accept()
                    with conn:
                        conn.settimeout(None) # Allow heavy tasks to take their time
                        data = conn.recv(4096 * 10).decode('utf-8') # Increased buffer for large inputs
                        if not data: 
                            continue

                        response = self.handle_command(data)
                        conn.sendall(json.dumps(response).encode('utf-8'))

                except socket.timeout:
                    loggerComms.error("Command channel idle timeout. Shutting down.")
                    self.running = False
                except Exception as e:
                    loggerBrain.error(f"Command Server Error: {e}")
                    self.running = False


    def handle_command(self, raw_data):
        try:
            command_json = json.loads(raw_data)
            action = command_json.get("action")
            print("Received command: ", action)
            data = command_json.get("data", {})

            if action == "ping":
                loggerComms.info("Got Ping")
                return {"status": "pong", "message": "pong"}
            
            elif action == "infer":
                ticker = data.get("ticker")
                model = data.get("model")
                open = data.get("open")
                high = data.get("high")
                low = data.get("low")
                close = data.get("close")
                volume = data.get("volume")
                engine = ModelInference(ticker, model)
                preds = engine.infer(open, high, low, close, volume)

                if hasattr(preds, "tolist"):
                    preds = preds.tolist()

                return {"status": "success", "predictions": preds}
            
            elif action == "get_inputs":
                engine = ModelInference(data.get("ticker"), data.get("model"))
                mkt_cols, sent_cols = engine.get_required_inputs()
                return {"status": "success", "market_cols": mkt_cols, "sentiment_cols": sent_cols}

            elif action == "train":
                # Placeholder for training logic
                return {"status": "started", "job_id": "123"}
            
            elif action == "shutdown":
                self.running = False
                return {"status": "closing"}
            
            elif action == "test":
                loggerComms.info("Python received test action.")
                return {"status": "ok", "message": "Test action received"}

            loggerComms.warning(f"Unknown action received: {action}")
            return {"status": "error", "message": "Unknown action"}
        except Exception as e:
            loggerBrain.error(f"Error handling command: {e}")
            loggerComms.error(f"Failed to process command: {raw_data}.")
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hport", type=int, required=True)
    args = parser.parse_args()
    loggerComms.info(f"Received args: {args}")
    bridge = JavaBridge(port=args.port, hport=args.hport)
    loggerComms.info("Starting JavaBridge...")
    bridge.start()
    loggerComms.info("JavaBridge has shut down.")