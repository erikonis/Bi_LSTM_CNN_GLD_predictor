import socket
import json
import threading
import time
import argparse
from src.python_engine.training.Constants import ColNames, Training
from src.python_engine.inference.inference import ModelInference
from src.utils.paths import LOG_COMMS_PATH, LOG_BRAIN_PATH
from src.utils.logger import setup_logging
from src.python_engine.training.train_model import main as train_main
from src.python_engine.training.models import name_to_class
from src.utils.data_management import delete_model, update_data

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
        """Start the bridge: launch heartbeat thread and run the command listener.

        This method spawns a background heartbeat listener and then enters
        the (blocking) command listener loop.
        """
        h_thread = threading.Thread(target=self.run_heartbeat_listener, daemon=True)
        h_thread.start()
        self.run_command_listener()

    def run_heartbeat_listener(self):
        """Heartbeat thread: shuts down bridge if no ping received within timeout."""
        last_ping_time = time.time()  # track last successful ping
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.hport))
            s.listen(5)
            # short accept timeout so we can frequently check last_ping_time
            s.settimeout(1.0)
            
            loggerComms.info(f"Heartbeat listener active on port {self.hport}")
            
            while self.running:
                # If no ping for 15s, mark as stopped
                if time.time() - last_ping_time > 15:
                    loggerComms.error("Heartbeat timeout: No valid ping for 15s. Shutting down bridge.")
                    self.running = False
                    break

                try:
                    conn, _ = s.accept()
                    with conn:
                        conn.settimeout(2.0)  # read timeout
                        raw_data = conn.recv(1024).decode('utf-8')
                        
                        if raw_data:
                            try:
                                data = json.loads(raw_data)
                                if data.get("status") == "ping" or data.get("action") == "ping":
                                    response = {"status": "pong"}
                                    conn.sendall(json.dumps(response).encode('utf-8'))
                                    last_ping_time = time.time()  # reset heartbeat timer
                            except json.JSONDecodeError:
                                loggerComms.warning("Received non-JSON data on heartbeat port.")
                                
                except socket.timeout:
                    # socket.timeout is expected; loop and re-check heartbeat
                    continue
                except Exception as e:
                    if self.running:
                        loggerComms.error(f"Heartbeat Thread Error: {e}")

    def run_command_listener(self):
        """Main command server loop: accept JSON commands and return JSON responses.

        Handles actions such as `infer`, `train`, `delete_model`, `update_ticker`, and `shutdown`.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            s.settimeout(1.0)  # check self.running every second
            loggerComms.info(f"Command listener active on port {self.port}")

            while self.running:
                try:
                    conn, addr = s.accept()
                    with conn:
                        # allow heavy tasks to complete
                        data = conn.recv(4096 * 10).decode('utf-8')
                        if not data: continue
                        response = self.handle_command(data)
                        conn.sendall(json.dumps(response).encode('utf-8'))
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        loggerBrain.error(f"Command Server Error: {e}")
                        self.running = False


    def handle_command(self, raw_data):
        """Parse and execute a JSON command string from the Java front-end.

        Args:
            raw_data: JSON-encoded command string containing keys `action` and optional `data`.

        Returns:
            dict: Response dict serializable to JSON with at least a `status` key.
        """
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
            
            elif action == "shutdown":
                self.running = False
                return {"status": "closing"}
            
            elif action == "test":
                loggerComms.info("Python received test action.")
                return {"status": "ok", "message": "Test action received"}
            
            elif action == "delete_model":
                loggerBrain.info(f"Delete action received for model: {data.get('model_name')} of ticker: {data.get('ticker')}")
                model = data.get("model_name")
                delete_model(model)
                return {"status": "success", "message": "Model deleted successfully."}

            elif action == "update_ticker":
                ticker = data.get("ticker")
                update_data(ticker)
                return {"status": "success", "message": f"Ticker {ticker} updated successfully."}

            elif action == "train":
                loggerBrain.info(f"Starting training pipeline for ticker: {data.get('ticker')}")
                try:
                    # Map string name to Class reference
                    model_class = name_to_class(data.get("model"))
                    
                    # Invoke training pipeline with parameters from Java (use defaults when missing)
                    train_main(
                        predictor_class=model_class,
                        model_name=data.get("name"),
                        auto_feat_engineering=data.get("auto_feat", False),
                        early_stop=data.get("early_stop", False),
                        batch_size=int(data.get("batch", 64)),
                        dropout_rate=float(data.get("dropout", 0.3)),
                        feature_threshold=float(data.get("feat_threshold", 0)),
                        data_split=data.get("data_split", Training.DATASPLIT_EXPAND),
                        targets=data.get("targets", [ColNames.TARGET_C_NORM]), # Default list
                        not_considered_feat=data.get("not_considered_feat", []),
                        ticker=data.get("ticker"),
                        epochs=int(data.get("epochs", 100)),
                        learning_rate=float(data.get("lr", 0.001)),
                        hidden_dim=int(data.get("hidden", 64)),
                        information=data.get("info", "None")
                    )
                    
                    loggerBrain.info(f"Training successfully completed for {data.get('name')}")
                    return {"status": "success", "message": f"Model {data.get('name')} trained successfully."}
                
                except Exception as train_error:
                    loggerBrain.error(f"Pipeline Execution Error: {train_error}")
                    return {"status": "error", "message": str(train_error)}

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