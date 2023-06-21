import torch
import os
import shutil
from logger import logging_setup
from typing import List
from datetime import datetime
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

# Setup for module
logger = logging_setup(__name__)
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def write_file(data: List[str], path: Path):
    """
    Write data to file.
    """
    with open(Path(path), "w") as f:
        for line in data:
            f.write(line)
            f.write("\n")
    logger.info(f"Generated {path}")
    return


def write_neg_edges(data: torch.tensor):
    """
    Write negative edges to file using filename from config.ini
    """
    # Write negative edges to file
    filename = config["FILENAMES"]["negative_samples"]
    with open(filename, "w") as f:
        for edge in data:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")
    logger.info(f"Generated {filename}")
    return


def read_neg_edges():
    """
    Read negative edges from file using filename from config.ini
    """
    filename = config["FILENAMES"]["negative_samples"]
    # Read negative edges from file
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [s.strip("\n") for s in lines]
        lines = [s.split(" ") for s in lines]
        lines = [[int(s[0]), int(s[1])] for s in lines]
        lines = torch.tensor(lines)
    logger.info(f"Read {filename}")
    return lines


def load_file(filename: Path, logging: bool = True):
    """
    Load file and return list of lines
    """
    file1 = open(filename, "r")
    Lines = file1.readlines()
    Lines = [s.strip("\n") for s in Lines]
    file1.close()
    if logging:
        logger.info(f"Loaded {filename}")
    return Lines


def data_config_log(d_name):
    """
    Write config to a seperate log file and save it to the data folder
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    l_dt = len(dt_string)

    # Convert config to nice looking string using the DATASTRUCTURE section
    # find longest key and normalize all keys to that length when saving them in the string
    max_key_length = 7  # length of 'Version' and 'Created'
    for key in config["DATASTRUCTURE"]:
        if (
            len(key) > max_key_length and not "_abbreviation" in key
        ):  # remove key if it has _abbreviation in it
            max_key_length = len(key)
    max_key_length += 1  # add 1 for the space after the key
    _length = max_key_length + l_dt + 2

    config_string = "----DATA CONFIGURATION----\n"
    # Check data instances
    if config["NUMBERINSTANCES"].getboolean("ALL") is False:
        config_string += f"{'Instances'}: {config['NUMBERINSTANCES']['number']}\n"
    else:
        config_string += f"-FULL DATA-\n"

    # Abbreviations combined
    abr = ""
    for key in config["DATASTRUCTURE"]:
        if key != "Grouping" and "_abbreviation" in key:
            if config["DATASTRUCTURE"].getboolean(key.replace("_abbreviation", "")):
                abr += config["DATASTRUCTURE"][f"{key}"]
    config_string += abr

    # Add Graphname
    config_string += f"\nGraphname: {d_name}"

    config_string += f'\n{"-"*(_length)}\n'
    for key in config["DATASTRUCTURE"]:
        if key != "Grouping" and not "_abbreviation" in key:
            # Remove _ from key and make the first letter uppercase
            ckey = key.replace("_", " ")
            ckey = ckey.title()
            config_string += (
                f"{ckey.ljust(max_key_length)}: {config['DATASTRUCTURE'][key]}\n"
            )
    config_string += f'\n{"-"*(_length)}\n'
    # Add creation date
    config_string += f"{'Created'.ljust(max_key_length)}: {dt_string}\n"
    # Add version
    config_string += (
        f"{'Version'.ljust(max_key_length)}: {config['VERSION']['version']}\n"
    )
    config_string += f'\n{"-"*(_length)}\n'

    # Write config to file named data_config.txt
    filename = Path(
        config["PATHS"]["libfm_path"], config["FILENAMES"]["data_config_file"]
    )
    with open(filename, "w") as f:
        f.write(config_string)
    return


def check_folderstructure():
    """
    Check if folder structure matches config.ini. Returns True if it does, False if it doesn't.
    """
    # Check if folder structure matches config.ini
    # Check if libfm folder exists
    if not Path(config["PATHS"]["libfm_path"]).exists():
        logger.error(
            f"Folder {config['PATHS']['libfm_path']} does not exist.\
            Please add the libfm folder to the folder structure!"
        )
        return False

    # Check if libfm folder contains "libfm.exe", "transpose.exe", "convert.exe"
    if not Path(config["PATHS"]["libfm_path"], "libfm.exe").exists():
        logger.error(
            f"File {config['PATHS']['libfm_path']}libfm.exe does not exist.\
            Please download libfm from https://www.libfm.org/ and place the files in the libfm folder!"
        )
        return False
    if not Path(config["PATHS"]["libfm_path"], "transpose.exe").exists():
        logger.error(
            f"File {config['PATHS']['libfm_path']}transpose.exe does not exist.\
            Please download libfm from https://www.libfm.org/ and place the files in the libfm folder!"
        )
        return False
    if not Path(config["PATHS"]["libfm_path"], "convert.exe").exists():
        logger.error(
            f"File {config['PATHS']['libfm_path']}convert.exe does not exist.\
            Please download libfm from https://www.libfm.org/ and place the files in the libfm folder!"
        )
        return False

    # Check if neighborhood_folder exists
    if not Path(config["FOLDERNAMES"]["neighborhood_folder"]).exists():
        logger.error(
            f"Folder {config['FOLDERNAMES']['neighborhood_folder']} does not exist.\
            Please add the neighborhood folder to the folder structure!"
        )
        return False

    return True


def data_folders_naming() -> Path:
    # Naming convention for folder names for data files
    # Standard path for folder = data/graph_name
    # Filesnames will not have specific names that change from config.ini
    # Naming logic:
    # graphname_{parts included} following the order of the config file

    graph_name = config["STANDARD"]["graph_name"]
    # Put parts together in order when activated
    parts = ""
    for part in config["DATASTRUCTURE"]:
        if "_abbreviation" in part:
            # remove _abbreviation from part
            part_bool = part.replace("_abbreviation", "")
            if config["DATASTRUCTURE"].getboolean(part_bool):
                parts += config["DATASTRUCTURE"][part]
    # Check if all instances are used
    if config["NUMBERINSTANCES"].getboolean("all"):
        folder_path = Path(f"{config['PATHS']['data_path']}/{graph_name}_{parts}")
    else:
        folder_path = Path(
            f"{config['PATHS']['data_path']}/{graph_name}_{parts}_{config['NUMBERINSTANCES']['number']}"
        )
    return folder_path


def copy_data_files():
    # Copy data files to the data folder
    # Check if folder structure matches config.ini
    if not check_folderstructure():
        return

    # Check if data folder exists and create it if it doesn't
    if not Path(config["PATHS"]["data_path"]).exists():
        Path(config["PATHS"]["data_path"]).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder {config['PATHS']['data_path']}")

    filenames = filelist_data()
    folder_path = data_folders_naming()
    # Create folder if it doesn't exist
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder {folder_path}")
    for filename in filenames:
        file_path = Path(config["PATHS"]["libfm_path"], filename)
        if not file_path.exists():
            logger.error(
                f"File {filename} seems not be generated. Please check if this is correct!"
            )
        else:
            destination = Path(f"{folder_path}/{filename}")
            # Copy file to destination
            shutil.copy(file_path, destination)
    logger.info(f"Data files copied to {destination}")
    return


def filelist_data():
    # Get list of filenames from config.ini for following keys: libfm_train, libfm_test, libfm_valid, groups
    # Check if files exist and copy them to the data folder
    keys = ["libfm_train", "libfm_test", "libfm_valid"]
    filenames = []
    # get names of files from config.ini
    for key in keys:
        # Strip ending from key by using .
        key = config["FILENAMES"][key].split(".")[0]
        # Endings
        endings = [".libfm", ".x", ".y", ".xt"]
        for ending in endings:
            filenames.append(key + ending)

    # Add groups to keys
    filenames.append(config["FILENAMES"]["groups"])
    # Add config text file to keys
    filenames.append(config["FILENAMES"]["data_config_file"])

    return filenames


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return
