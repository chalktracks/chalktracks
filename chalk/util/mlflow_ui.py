import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def add_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the mlflow_ui command."""
    parser.add_argument("mlflow_dir", help="Directory containing MLflow tracking data (e.g., workspace/trainin/runs/mlflow)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the MLflow server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the MLflow server to (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")


def open_browser_delayed(url: str, delay: int = 3):
    """Open browser after a delay."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"Opened browser to {url}")
    except Exception as e:
        print(f"Failed to open browser: {e}")
        print(f"Please open {url} manually")


def main(args):
    """
    Start MLflow UI server and optionally open browser.
    
    This command starts an MLflow tracking server and optionally opens
    a web browser to view the MLflow dashboard. The server will run
    until interrupted (Ctrl+C).
    """
    mlflow_dir = Path(args.mlflow_dir)
    
    if not mlflow_dir.exists():
        print(f"Error: MLflow directory {mlflow_dir} does not exist")
        return
    
    backend_store_uri = str(mlflow_dir.absolute())
    host = args.host
    port = args.port
    url = f"http://{host}:{port}"
    
    print(f"Starting MLflow server...")
    print(f"  Backend store URI: {backend_store_uri}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  URL: {url}")
    
    # Start browser opening thread if requested
    if not args.no_browser:
        browser_thread = threading.Thread(
            target=open_browser_delayed, 
            args=(url,),
            daemon=True
        )
        browser_thread.start()
        print(f"Browser will open in 3 seconds...")
    
    try:
        # Start MLflow server using subprocess
        # This is equivalent to: mlflow server --backend-store-uri <path> --host <host> --port <port>
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--host", host,
            "--port", str(port)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print("Press Ctrl+C to stop the server")
        print()
        
        # Start the MLflow server process
        process = subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        print(f"Make sure MLflow is installed and accessible")
    except FileNotFoundError:
        print("Error: 'mlflow' command not found")
        print("Make sure MLflow is installed: pip install mlflow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MLflow UI server and open browser")
    add_arg_parser(parser)
    args = parser.parse_args()
    main(args)
