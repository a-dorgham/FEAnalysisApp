import json
import os

class StructureConfig:
    """Class to manage configuration files for structures, elements, nodes, and units."""
    
    # Define the directory where JSON files are stored
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # Mapping of keywords to corresponding JSON filenames
    FILE_MAPPING = {
        "structure": "structure_types.json",
        "element": "default_element.json",
        "node": "default_node.json",
        "units": "default_units.json",
        "properties": "default_properties.json",
        None: "structure_types.json" # Default
    }

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
        if keyword not in StructureConfig.FILE_MAPPING:
            raise ValueError(f"Invalid keyword '{keyword}'. Must be one of {list(StructureConfig.FILE_MAPPING.keys())}.")

        # Construct the full path to the selected JSON file
        config_path = os.path.join(StructureConfig.CONFIG_DIR, StructureConfig.FILE_MAPPING[keyword])

        # Load the JSON file
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        return data  # Return full JSON if no structure_type is specified
