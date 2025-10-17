# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
import os
import threading
import socket
from PyQt6.QtCore import QThread, pyqtSignal
from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Optional, Dict
import logging

from src.utils.errors import ErrorHandler


# ------------------------------------------------
# Custom HTTP request handler with external routes
# ------------------------------------------------
class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """
    Custom HTTP handler that can serve files from multiple directories
    based on URL prefixes defined in `route_map`.
    """

    route_map: Dict[str, str] = {}

    def translate_path(self, path: str) -> str:
        """
        Overrides the default path translation.
        If a prefix in route_map matches the start of `path`,
        the request is mapped to the corresponding absolute directory.
        Otherwise, falls back to the default implementation.

        Args:
            path (str): The URL path requested by the client.

        Returns:
            str: The file system path corresponding to the requested URL path.
        """
        for prefix, target_dir in self.route_map.items():
            if path.startswith(prefix):
                rel_path = path[len(prefix) :].lstrip("/")
                normalized_rel_path = rel_path.replace("/", os.sep)    
                return os.path.normpath(os.path.join(target_dir, normalized_rel_path))
        return super().translate_path(path)

    def log_message(self, format, *args):
        """
        Override to completely suppress HTTP server access logs.
        These are already handled by the main logging system if needed.
        """
        return


# ------------------------------------------------
# LOCAL SERVER THREAD
# ------------------------------------------------
class HTTPServerThread(QThread):
    """
    A multi-threaded HTTP server class that serves files from a specified directory.
    Supports serving external folders through a configurable route map.
    """
    error_signal = pyqtSignal(str)
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        directory: str = "",
        route_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes the HTTPServerThread with the specified host, port, directory, and route map.

        Args:
            host (str): Hostname or IP address on which the server listens.
            port (int): Port number.
            directory (str): Root directory for default file serving.
            route_map (dict): Mapping of URL prefixes to absolute directories.
        """
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rel_vendors_dir = os.path.join(script_dir, "..", "..", "assets", "plotly")
    
        abs_vendors_dir = os.path.abspath(rel_vendors_dir)
        self.route_map = route_map or {"/js/": abs_vendors_dir}
        self.host = host or "localhost"
        self.port = port or 8000
        server_dir = os.path.join("src", "utils")
        self.directory = directory or os.path.abspath(server_dir)
        
        self.server = None
        self._running = False

    # ------------------------------------------------
    # SERVER CONTROL METHODS
    # ------------------------------------------------
    def run(self):
        """Start the HTTP server."""
        try:
            os.chdir(self.directory)
            CustomHTTPRequestHandler.route_map = self.route_map
            self.server = HTTPServer((self.host, self.port), CustomHTTPRequestHandler)
            self.server.timeout = 0.5
            self._running = True
            
            print(f"##### Serving HTTP on {self.host}:{self.port} from directory: {self.directory} #####")
            
            while self._running:
                try:
                    self.server.handle_request() 
                except socket.timeout:
                    continue 
                except Exception as e:
                    if self._running:  
                        self.error_signal.emit(str(e))
                        
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        """Stop the HTTP server."""
        self._running = False
        if self.server:
            try:
                self.server.server_close()
                print("HTTP Server stopped")
                
            except Exception as e:
                print(f"Error while stopping HTTP Server. {e}")   