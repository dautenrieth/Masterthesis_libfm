# This module checks if the data files selected in the config file are present and if not, generates them.
# The data files are generated using the  data_generation.py and data_parts.py module.
import shutil

from logger import logging_setup
logger = logging_setup(__name__)

from configparser import ConfigParser, ExtendedInterpolation
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')

from pathlib import Path
from data_generation import create_data_files
from utils import data_folders_naming, filelist_data

def load_or_generate_data():
    '''
    Check if data files are present and if not, generate them.
    '''
    # Check if data folder exists and if not, create it
    # And check if folder with graph name exists and if not, create it

    if not Path(config['PATHS']['data_path']).is_dir():
        Path(config['PATHS']['data_path']).mkdir(exist_ok=True)
        logger.info(f"Created {config['PATHS']['data_path']} folder")
        # Generate data files
        logger.info(f"No data existing processed data found.")
        create_data_files(config=config)
    else:
        folder_path = data_folders_naming()
        folder_name = str(folder_path).split('/')[-1]
        if not Path(f"{folder_path}").is_dir():
            # Generate data files
            logger.info(f"No data found for {folder_name}.")
            create_data_files(config=config)
        else:
            # Check if all files are present
            # If not, generate data files
            files_exist = True
            filenames = filelist_data()
            for file in filenames:
                if not Path(f"{folder_path}/{file}").is_file():
                    # Generate data files
                    # Log that files are missing
                    logger.info(f"File {file} is missing in {folder_name}.")
                    files_exist = False

            if files_exist:
                # Copy all files from data folder to libfm folder
                for file in Path(f"{folder_path}").iterdir():
                    file_name = file.name
                    destination = Path(f"{config['PATHS']['libfm_path']}/{file_name}")
                    shutil.copy(file, destination)
                    # Log each file that is copied
                    logger.info(f"Copied {file_name} to {destination}")
            else:
                create_data_files(config=config)
        
    return


if __name__ == "__main__":
    load_or_generate_data()