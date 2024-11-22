import yaml
import os

def load_branches_config(file_path="config/branches.yml"):
    """
    Load the E and T branches configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file. Defaults to "config/branches.yml".

    Returns:
        dict: Dictionary with `E_branches` and `T_branches`.
    """
    # Ensure the default file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config