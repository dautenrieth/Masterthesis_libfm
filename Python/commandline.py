import os, shutil
from pathlib import Path
from logger import logging_setup
from configparser import ConfigParser, ExtendedInterpolation
import subprocess
import re
import utils as ut
import pandas as pd
from datetime import datetime
from utils import load_file
import clear_outputs as co
from openpyxl import load_workbook, Workbook

# Setup for module
logger = logging_setup(__name__)
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def external_program_call(command: str, path: Path):
    # Delete prediction file to prevent wrong evaluation
    co.delete_prediction_file()

    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        f"cd {path} && {command}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    out, err = process.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    logger.info(f"Finished running libfm")
    print(out)
    process_output(out, command)

    if err:
        logger.error(f"Error output: {err}")

    return


def process_output(out, command):
    # Function can be used to process output from libfm
    pattern = r"#Iter=\s*(\d+)\s*Train=(\S+)\s*Test=(\S+)"
    matches = re.findall(pattern, out)

    # Convert the matches to a DataFrame
    data_df = pd.DataFrame(matches, columns=["Iter", "Train", "Test"])

    # Convert types for the columns
    data_df["Iter"] = data_df["Iter"].astype(int)
    data_df["Train"] = data_df["Train"].astype(float)
    data_df["Test"] = data_df["Test"].astype(float)

    # Get the current timestamp as filename
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure the directory exists
    if not os.path.exists("steps"):
        os.makedirs("steps")

    # Prevent filename clashes
    count = 1
    new_filename = filename
    while os.path.exists(f"steps/{new_filename}.txt"):
        new_filename = f"{filename}_{count}"
        count += 1

    # Create the txt file and write the command to it
    with open(f"steps/{new_filename}.txt", "w") as file:
        file.write(f"Command; {command}\n")
        data_df.to_csv(file, sep=";", index=False, line_terminator="\n")

    # Take rlog file and save it with steps
    folder_path = Path(f'{config["PATHS"]["libfm_path"]}')
    # Setzen Sie den Pfad zur Datei
    file_path = Path(folder_path, config["FILENAMES"]["rlog_file"])

    # Überprüfen Sie, ob die Datei existiert
    if file_path.is_file():
        # Setzen Sie den Pfad für die neue Datei
        new_file_path = f"steps/rlog_{new_filename}.csv"

        # Kopieren Sie die Datei und benennen Sie sie um
        shutil.copy(file_path, new_file_path)

    print(f"Created steps/{new_filename}.txt")
    return


def convert_to_binary(filename: str, path: Path):
    # Function can be used to convert the files to binary
    logger.info(f"Convert file {filename} to binary")
    # Cut the file extension
    filename_wo_ext = filename.split(".")[0]
    # Create the command
    command = f"convert --ifile {filename} --ofilex {filename_wo_ext}.x --ofiley {filename_wo_ext}.y"
    output_stream = os.popen(f"cd{path} && {command}")
    out = output_stream.read()
    print(out)
    if "num_rows" not in out:
        logger.error(
            f"Error in converting {filename} to binary:\nOutput till \
            determination:\n {out}\nPlease check filename and path."
        )
        raise Exception(
            f"Error in converting {filename} to binary. See log for more information"
        )
    logger.info(f"Finished running convert function")
    return


def transpose_binary(filename: str, path: Path):
    # Function can be used to transpose the files to binary
    logger.info(f"Transpose binary file {filename}")
    # Cut the file extension
    filename_wo_ext = filename.split(".")[0]

    ## Convert x
    # Create the command
    command = f"transpose --ifile {filename_wo_ext}.x --ofile {filename_wo_ext}.xt"
    output_stream = os.popen(f"cd{path} && {command}")
    out = output_stream.read()

    if "num_rows" not in out:
        logger.error(
            f"Error in transposing {filename} to binary:\nOutput till \
            determination:\n {out}\nPlease check filename and path."
        )
        raise Exception(
            f"Error in transposing {filename} to binary. See log for more information"
        )

    ## Convert y
    # command = f'transpose --ifile {filename_wo_ext}.y --ofile {filename_wo_ext}.yt'
    # output_stream = os.popen(f'cd{path} && {command}')
    # out = output_stream.read()

    # if 'num_rows' not in out:
    #     logger.error(f'Error in transposing {filename} to binary:\nOutput till \
    #         determination:\n {out}\nPlease check filename and path.')
    #     raise Exception(f'Error in transposing {filename} to binary. See log for more information')

    logger.info(f"Finished running transpose function")
    return
