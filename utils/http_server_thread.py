import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Optional

# ---------------------------------------------
# HTTP SERVER CLASS
# ---------------------------------------------

class HTTPServerThread(threading.Thread):
    """
    A multi-threaded HTTP server class that serves files from a specified directory.
    This class extends `threading.Thread` to run the HTTP server in a separate thread,
    preventing it from blocking the main program execution. It allows for easy
    instantiation, starting, and stopping of a basic HTTP server.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, directory: str = ".") -> None:
        """
        Initializes the HTTPServerThread with the specified host, port, and directory.
        Args:
            host (str): The hostname or IP address on which the server will listen.
                        Defaults to "localhost".
            port (int): The port number on which the server will listen.
                        Defaults to 8000.
            directory (str): The root directory from which files will be served.
                             Defaults to ".", meaning the current working directory.
        """
        super().__init__()
        self.host: str = host
        self.port: int = port
        self.directory: str = directory
        self.server: Optional[HTTPServer] = None

    # ---------------------------------------------
    # SERVER CONTROL METHODS
    # ---------------------------------------------

    def run(self) -> None:
        """
        Starts the HTTP server.
        This method is the entry point for the thread when `start()` is called.
        It changes the current working directory to the specified `self.directory`
        and then starts the HTTP server, which will serve indefinitely until `stop()` is called.
        """
        os.chdir(self.directory)
        self.server = HTTPServer((self.host, self.port), SimpleHTTPRequestHandler)
        print(f"Serving HTTP on {self.host}:{self.port} from directory: {self.directory}")
        self.server.serve_forever()

    def stop(self) -> None:
        """
        Stops the HTTP server.
        If the server is running, this method shuts it down gracefully.
        It prints a confirmation message upon successful shutdown.
        """

        if self.server:
            self.server.shutdown()
            print("HTTP Server stopped")