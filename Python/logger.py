import logging
import re
import os
import pandas as pd
import datetime
from pathlib import Path


def logging_setup(module_name="default"):
    # Set up logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Set up logging to file
    file_handler = logging.FileHandler("log.txt")  # overwrite existing file
    file_handler.setLevel(logging.INFO)

    # Create a logger and set the logging level
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Add the console and file handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    return logger


# Log to excel
def save_pred(command, average, stddev, rand, config, logger):
    # Get data_config.txt from libfm folder
    folder_path = Path(f'{config["PATHS"]["libfm_path"]}')
    dataconfig = Path(folder_path, "data_config.txt")
    # Check if data_config.txt exists
    if not dataconfig.is_file():
        n = "Unknown"
        features = "Unknown"
    else:
        with open(dataconfig, "r") as f:
            dataconfig = f.read()
        # Check if either '-FULL DATA-' or 'Instances' is in dataconfig.txt
        if "-FULL DATA-" in dataconfig:
            n = "FULL DATA"
        elif "Instances" in dataconfig:
            # Get number behind 'Instances'
            n = dataconfig.split("Instances: ")[1].split("  ")[0].split("\n")[0]
        else:
            n = "Unknown"

        # Extract feature names and values using regular expressions
        feature_dict = {}
        for match in re.finditer(
            r"^\s*(.+)\s+:\s+(\w+)", dataconfig, flags=re.MULTILINE
        ):
            feature_name, feature_value = match.group(1), match.group(2)
            key = feature_name.strip().replace(
                " ", ""
            )  # remove whitespaces from the key
            value = feature_value.strip()  # remove whitespaces from the value
            if key != "Created" and key != "Version" and key != "Grouping":
                feature_dict[key] = value

        # Get feature abbreviations

        feature_string = dataconfig.split("\n")[2]
        if feature_string == "":
            feature_string = "No features found"
        # Get graph name
        graph_string = dataconfig.split("\n")[3]
        if "Graphname:" in graph_string:
            gname = graph_string.split(" ")[1]
        else:
            gname = "Not specified"
    today = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    df = pd.DataFrame(
        {
            "date": [today],
            "command": [command],
            "Average": [average],
            "Standard Deviation": [stddev],
            "random": [rand],
            "Number Instances": [n],
            "Features": [feature_string],
            "Graph Name": [gname],
        }
    )

    filename = "Runs.xlsx"
    append_to_excel(df, filename)
    return


def append_to_excel(df, filename):
    # Check if the file exists
    if not os.path.isfile(filename):
        # If it doesn't exist, create the Excel file and write the data
        df.to_excel(filename, index=False)
    else:
        # If the file exists, load the existing data, append the new data, and save
        df_existing = pd.read_excel(filename)
        df_total = pd.concat([df_existing, df])
        df_total.to_excel(filename, index=False)
    return
