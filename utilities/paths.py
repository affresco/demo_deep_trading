import os
import pathlib

# This is just for reference:
# gets us to the project-level dir
PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()


# ##################################################################
# BASE PATH
# ##################################################################

def get_data_file_path(currency: str):
    f = get_data_folder_path()
    return f"{f}{str(currency).upper()}.csv"


def get_data_folder_path():
    return os.path.join(PROJECT_DIR, "data")


# Make sure the dir is there
os.makedirs(get_data_folder_path(), exist_ok=True)
