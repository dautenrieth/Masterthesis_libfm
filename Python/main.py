"""
This module contains the program flow and calls the other modules.
"""

from ogb.linkproppred import PygLinkPropPredDataset
import concurrent.futures
import numpy as np
from ogb.linkproppred import Evaluator
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
import data_generation as dg
import utils as ut
import commandline as cl
import clear_outputs as co
import datamanager as dm
from evaluator import calculate_results

from logger import logging_setup

logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def main():
    # Open txt file called command.txt and read the list of commands
    with open("commands.txt", "r") as f:
        commands = f.readlines()
    # Remove whitespace characters like `\n` at the end of each line
    commands = [x.strip() for x in commands]
    print(commands)

    for main_command in commands:
        # Clear the log file
        with open(config["FILENAMES"]["log_file"], "w") as f:
            # Write an empty string to the file, which will clears its contents
            f.write("")
        logger.info("Programm started")
        # Delete prediction file if it exists to not use old data
        co.delete_prediction_file()
        if ut.check_folderstructure() is False:
            logger.error(
                "Folder structure is not correct. Please check the config.ini file.\
                The Programm will now exit."
            )
            return

        if config["PARTACTIVATED"].getboolean("data_generation"):
            co.clear_outputs()
            dm.load_or_generate_data()

        if config["PARTACTIVATED"].getboolean("libfm_run"):
            print("Starting libfm")
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                n = int(config["RUNS"]["number"])
                commands = []
                for i in range(n):
                    command = main_command.replace(
                        "-out test.pred", f"-out predictions/test{i}.pred"
                    )
                    commands.append(command)
                paths = [Path(config["PATHS"]["libfm_path"])] * n
                executor.map(cl.external_program_call, commands, paths)

        if config["PARTACTIVATED"].getboolean("evaluation"):
            print("Evaluation started")
            calculate_results(command=main_command)
            print("Evaluation finished")

    logger.info("Programm finished")
    return


if __name__ == "__main__":
    main()
