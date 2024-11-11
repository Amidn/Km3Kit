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
    def readYML(cls, dataset_name, verbose=False):
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
        """Collects all file paths from the dataset."""
        neutrino_files = self.paths.get('neutrino', [])
        muon_files = self.paths.get('muon', [])
        data_files = self.paths.get('data', [])
        
        # Verbose output for path details
        if verbose:
            print("Paths retrieved from dataset:")
            print("Data Files:", data_files)
            print("Neutrino Files:", neutrino_files)
            print("Muon Files:", muon_files)
        
        return data_files, neutrino_files, muon_files

    def get_data_type(self):
        """Returns the data_type of the dataset."""
        return self.data_type