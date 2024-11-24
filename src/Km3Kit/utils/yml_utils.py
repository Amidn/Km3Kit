import yaml
import os
from datetime import datetime



class Loader:
    ALLOWED_DATA_TYPES = {"ROOT", "h5", "Panda", "CSV", "TXT"}
# Construct the correct path to the YAML file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print ("BASE_DIR", BASE_DIR)
    YAML_FILE_PATH = os.path.join(BASE_DIR, '../config', 'dataset_registry.yml')

    def __init__(self, name=None, date_added=None, data_type=None, comment=None, paths=None):
        self.name = name
        self.date_added = date_added
        self.data_type = data_type
        self.comment = comment
        self.paths = paths

    @classmethod
    def readYML(cls, dataset_name, verbose= True):
        """Reads the YAML file and returns a Loader instance for the dataset."""
        file_path = cls.YAML_FILE_PATH
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                for entry in data:
                    if entry['name'] == dataset_name:
                        data_type = entry.get('data_type')
                        # Check if data_type is allowed
                        if data_type not in cls.ALLOWED_DATA_TYPES:
                            raise ValueError(
                                f"Invalid data_type '{data_type}' for dataset '{dataset_name}'. "
                                f"Allowed types are: {', '.join(cls.ALLOWED_DATA_TYPES)}"
                            )
                        dataset_info = cls(
                            name=entry.get('name'),
                            date_added=entry.get('date_added'),
                            data_type=data_type,
                            comment=entry.get('comment'),
                            paths=entry.get('path', {})  # Ensure paths is a dictionary
                        )
                        if verbose:
                            dataset_info.print_dataset_info()
                        return dataset_info  # Return the Loader instance
        except FileNotFoundError:
            print(f"The file '{file_path}' does not exist.")
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
        except Exception as exc:
            print(f"An error occurred: {exc}")

        print(f"Dataset with name '{dataset_name}' not found.")
        return None

    def print_dataset_info(self):
        """Prints detailed information of the dataset."""
        print("Detailed information of the dataset:")
        print("Name:", self.name)
        print("Date Added:", self.date_added)
        print("Data Type:", self.data_type)
        print("Comment:", self.comment)
        print("Paths:")
        for data_key, paths_list in self.paths.items():
            print(f"  {data_key}:")
            for path in paths_list:
                print("   -", path)
        print("End of dataset information.")

    def read_paths(self, verbose=False):    
        """
        Collects all file paths from the dataset and formats them as dictionaries
        with a single key corresponding to the category and a set of file paths as the value.

        Args:
            verbose (bool): Whether to print detailed output.

        Returns:
            tuple: Dictionaries for data, muon, and neutrino paths.
        """
        # Format paths as dictionaries with one key and a set of values
        data_dict = {"data": "\n".join(self.paths.get("data", []))}
        muon_dict = {"muon": "\n".join(self.paths.get("muon", []))}
        neutrino_dict = {"neutrino": "\n".join(self.paths.get("neutrino", []))}

        # Verbose output for path details
        if verbose:
            print("Paths retrieved from dataset:")
            print("Data Files:\n", data_dict["data"])
            print("Muon Files:\n", muon_dict["muon"])
            print("Neutrino Files:\n", neutrino_dict["neutrino"])
    
        return data_dict, muon_dict, neutrino_dict
 

    def get_data_type(self):
        """Returns the data_type of the dataset."""
        return self.data_type

    @classmethod
    def list_datasets_and_types(cls, yaml_registry_path=None, verbose=False):
        """
        Extracts a list of dataset names and their data types from the YAML registry.

        Args:
            yaml_registry_path (str): Path to the YAML registry file. Defaults to the class YAML_FILE_PATH.
            verbose (bool): If True, prints detailed logs.

        Returns:
            list: A list of tuples [(dataset_name, data_type), ...].
        """
        yaml_registry_path = yaml_registry_path or cls.YAML_FILE_PATH
        try:
            with open(yaml_registry_path, "r") as file:
                data = yaml.safe_load(file)

            # Extract names and data types
            datasets = [(entry.get("name"), entry.get("data_type")) for entry in data]

            if verbose:
                print("Datasets and Types:")
                for name, dtype in datasets:
                    print(f"- {name}: {dtype}")

            return datasets

        except FileNotFoundError:
            print(f"Error: Registry file not found at {yaml_registry_path}")
            return []
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return []
        except Exception as exc:
            print(f"An unexpected error occurred: {exc}")
            return []
    

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


def add_dataset_to_registry(
    name,
    data_type,
    comment,
    directory_path,
    data_name=None,
    muon_name=None,
    neutrino_name=None,
    yml_registry_path="config/dataset_registry.yml",
    verbose=False,
):
    """
    Adds a new dataset to the dataset registry YAML file without modifying existing entries.

    Args:
        name (str): Name of the dataset.
        data_type (str): Type of the dataset (e.g., ROOT, FITS, Pandas).
        comment (str): Comment describing the dataset.
        directory_path (str): Base directory path where the files are located.
        data_name (str): Filename for the data file.
        muon_name (str): Filename for the muon file.
        neutrino_name (str): Filename for the neutrino file.
        yml_registry_path (str): Path to the dataset registry YAML file.
        verbose (bool): If True, prints detailed logs.

    Returns:
        bool: True if the dataset was added successfully, False otherwise.
    """
    if not os.path.exists(yml_registry_path):
        print(f"Error: Registry file '{yml_registry_path}' not found.")
        return False

    # Construct full paths for data, muon, and neutrino files
    data_paths = [os.path.join(directory_path, data_name)] if data_name else []
    muon_paths = [os.path.join(directory_path, muon_name)] if muon_name else []
    neutrino_paths = [os.path.join(directory_path, neutrino_name)] if neutrino_name else []

    # Create a new dataset entry
    new_dataset_entry = {
        "name": name,
        "date_added": datetime.now().strftime("%Y-%m-%d"),
        "data_type": data_type,
        "comment": comment,
        "folder": "",
        "path": {
            "data": data_paths,
            "data_livetime": [],
            "muon": muon_paths,
            "muon_livetime": [],
            "neutrino": neutrino_paths,
            "neutrino_livetime": [],
        },
    }

    try:
        # Load the existing registry
        with open(yml_registry_path, "r") as file:
            registry = yaml.safe_load(file) or []

        # Check if the dataset name already exists
        if any(entry["name"] == name for entry in registry):
            print(f"Dataset '{name}' already exists in the registry. Skipping addition.")
            return False

        # Append the new dataset entry
        registry.append(new_dataset_entry)
        
        # Save the updated registry back to the YAML file
        with open(yml_registry_path, "w") as file:
            yaml.dump(
                registry, 
                file, 
                default_flow_style=False, 
                sort_keys=False,
                width=1000,
            )
            file.write("\n") 
        if verbose:
            print(f"Dataset '{name}' added successfully to the registry.")
            print(f"Paths: {new_dataset_entry['path']}")

        return True

    except Exception as e:
        print(f"Error while updating the registry: {e}")
        return False




def readConfigs(config_file_name="config.yml"):
    """
    Reads a YAML configuration file and returns its contents.

    Args:
        config_file_name (str): Name of the YAML configuration file.
        
    Returns:
        dict: Dictionary containing the configuration data.
    """
    # Dynamically determine the base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(BASE_DIR, '../config', config_file_name)

    # Ensure the file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    # Load and parse the YAML file
    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_data