import os

def get_class_names(dataset_dir):
    """
    Scans the dataset directory and returns a list of class names,
    which are the names of the subdirectories within the dataset directory.

    Parameters:
    - dataset_dir (str): The path to the dataset directory.

    Returns:
    - class_names (list): A list of class names (subdirectory names).
    """
    # List all entries in the dataset directory
    entries = os.listdir(dataset_dir)
    
    # Filter out entries that are not directories
    class_names = [entry for entry in entries if os.path.isdir(os.path.join(dataset_dir, entry))]
    
    # Optionally, sort the class names
    class_names.sort()
    
    return class_names

# Example usage:
dataset_directory = 'PATH TO YOUR TRAIN DATASET'  # Replace with the actual path to your dataset directory
class_names = get_class_names(dataset_directory)

print("Class Names:", class_names)
