import json
import os
from src.constants import FILE_MAPPING

class StructureConfig:
    """Class to manage configuration files for structures, elements, nodes, and units."""

    CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


    @staticmethod
    def load_config(keyword=None):
        """
        Load configuration based on the structure type and keyword.
        
        Parameters:
        - structure_type (str): The type of structure (e.g., "2D_Beam", "3D_Truss").
        - keyword (str): One of ["structure", "element", "node", "units"] to determine which JSON to load.

        Returns:
        - dict: The requested configuration data.
        """
        if keyword not in FILE_MAPPING:
            raise ValueError(f"Invalid keyword '{keyword}'. Must be one of {list(FILE_MAPPING.keys())}.")

        config_path = os.path.join(StructureConfig.CONFIG_DIR, "..", "data", "defaults", FILE_MAPPING[keyword])

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        return data 
