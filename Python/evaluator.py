import os
import math
import matplotlib.pyplot as plt
from logger import save_pred
from logger import logging_setup

logger = logging_setup(__name__)

from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")

from ogb.linkproppred import Evaluator

import utils as ut
from pathlib import Path
import numpy as np


def count_pos_neg(libfm_test: Path):
    test_data = ut.load_file(libfm_test, logging=False)
    y_pred_pos = 0
    y_pred_neg = 0
    for i, line in enumerate(test_data):
        # Check if line[0] is int
        try:
            i_int = int(line[0])
        except ValueError:
            raise ValueError(
                f"The test data is not in the correct format. Value is not int {type(line[0])}."
            )
        # Check if line[0] is 0 or > 0
        if i_int == 0:
            y_pred_neg += 1
        elif i_int > 0:
            y_pred_pos += 1
        else:
            raise ValueError(
                f"The test data is not in the correct format. Value < 0 : {i_int}."
            )
    # print(f"lengths - y_pred_pos: {len(y_pred_pos)}, y_pred_neg: {len(y_pred_neg)}")
    return y_pred_pos, y_pred_neg


def positive_in_top_k_prob(x, k=50):
    if x <= k - 1:
        return 1.0
    P = (math.comb(1, 1) * math.comb((x + 1) - 1, k - 1)) / math.comb(x + 1, k)
    return P


def evaluate_file(file_path: Path):
    # Load test data
    libfm_test = Path(
        f'{config["PATHS"]["libfm_path"]}/{config["FILENAMES"]["libfm_test"]}'
    )
    test_data = ut.load_file(libfm_test, logging=False)

    y_pred = ut.load_file(file_path, logging=False)
    evaluator = Evaluator(name=config["STANDARD"]["graph_name"])
    # print(evaluator.expected_input_format)
    # print(evaluator.expected_output_format)

    # Iterate through test data and create 2 np.float lists of positive and negative edges from the prediction file
    y_pred_pos = []
    y_pred_neg = []
    for i, line in enumerate(test_data):
        # Check if line[0] is int
        try:
            i_int = int(line[0])
        except ValueError:
            raise ValueError(
                f"The test data is not in the correct format. Value is not int {type(line[0])}."
            )
        # Check if line[0] is 0 or > 0
        if i_int == 0:
            y_pred_neg.append(float(y_pred[i]))
        elif i_int > 0:
            y_pred_pos.append(float(y_pred[i]))
        else:
            raise ValueError(
                f"The test data is not in the correct format. Value < 0 : {i_int}."
            )
    # print(f"lengths - y_pred_pos: {len(y_pred_pos)}, y_pred_neg: {len(y_pred_neg)}")
    y_pred_pos = np.float_(y_pred_pos)
    y_pred_neg = np.float_(y_pred_neg)

    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result_dict = evaluator.eval(input_dict)
    logger.info(f"Result dict: {result_dict} - {file_path}")

    for key in result_dict.keys():
        if "hits@" not in key:
            logger.error("Key in result dict doesnt contain 'hits@':", key)
            break

    return list(result_dict.values())[0]


def calculate_results(command: str):
    folder_path = config["PATHS"]["predictions_path"]
    # Get a list of files in the folder
    files = os.listdir(folder_path)
    hits = []
    # Iterate through the files
    for file in files:
        file_path = Path(f"{folder_path}/{file}")
        # Call evaluate function
        hits.append(evaluate_file(file_path))
    # Calculate average and standard deviation
    avg = sum(hits) / len(hits)
    std = np.std(hits)
    logger.info(f"Average: {avg}, Standard deviation: {std}")

    # Calculate expected random result
    libfm_test = Path(
        f'{config["PATHS"]["libfm_path"]}/{config["FILENAMES"]["libfm_test"]}'
    )
    n_pos, n_neg = count_pos_neg(libfm_test)
    prob = positive_in_top_k_prob(n_neg)
    logger.info(f"Expected random result: {prob}")
    save_pred(command, avg, std, prob, config=config, logger=logger)

    return


if __name__ == "__main__":
    calculate_results(command="")
