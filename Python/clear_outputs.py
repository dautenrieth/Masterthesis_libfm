import os
from configparser import ConfigParser, ExtendedInterpolation

def clear_outputs(delete_neg_samples = False):
    '''
    Clear outputs from previous run
    '''
    # Setup for module
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('config.ini')

    # Get filenames and paths
    train_complt = config["FILENAMES"]["libfm_train"]
    train = train_complt.split('.')[0]
    test_complt = config["FILENAMES"]["libfm_test"]
    test = test_complt.split('.')[0]
    valid_complt = config["FILENAMES"]["libfm_valid"]
    valid = valid_complt.split('.')[0]
    prediction_file = config["FILENAMES"]["prediction"]
    groups = config["FILENAMES"]["groups"]
    negative_samples = config["FILENAMES"]["negative_samples"]
    data_config_file = config["FILENAMES"]["data_config_file"]

    files_to_delete = [f"{train_complt}", f"{train}.y", f"{train}.x", f"{train}.xt", f"{test_complt}",\
        f"{test}.xt", f"{test}.x", f"{test}.y", f"{prediction_file}", f"{valid_complt}", f"{valid}.x",\
            f"{valid}.xt", f"{valid}.y", f"{groups}", f"{data_config_file}"]
    if delete_neg_samples:
        files_to_delete.append(f"{negative_samples}")
    folder_path = config["PATHS"]["libfm_path"]

    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f'Deleted {file_path}')
        except:
            print(f'Could not delete {file_path} (file does not exist)')

    return

def delete_prediction_file():
    
    # Setup for module
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('config.ini')


    folder_path = config["PATHS"]["predictions_path"]
    # Delete all file in folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f'Deleted {file}')
        except:
            print(f'Could not delete {file_path} (file does not exist)')

    return

if __name__ == "__main__":
    clear_outputs()


