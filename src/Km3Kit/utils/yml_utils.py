import yaml
import os


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