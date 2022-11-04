"""
    File management functions for saving/loading checkpoints during training.
"""

import os, sys
from datetime import datetime
from pathlib import Path
file = Path(__file__).resolve()
path2project = str(file.parents[0]) + '/'
path2currDir = str(Path.cwd()) + '/'
sys.path.append(path2project) # add top level directory -> geom_dl/


def make_checkpoint_run_folder(path2checkpoints, path2run):
    try:
        os.mkdir(path=path2checkpoints)
        print("Creating checkpoints folder")
    except FileExistsError as e:
        print(f"Checkpoints folder {path2checkpoints} already exists")

    try:
        os.mkdir(path=path2run)
        print(f"Creating folder for this run: {path2run}", flush=True)
    except FileExistsError as e:
        print(f"This run {path2run} already exists already exists")


def make_checkpoint_callback_dict(path2currDir, monitor, mode, task, loss, which_exp, rand_seed, run_directory, misc=None, subnets=False):
    assert ("=" not in path2currDir), f'= in path2CurrDir {path2currDir} not allowed!'
    assert ("=" not in run_directory), f'= in run_directory {run_directory} not allowed!'
    # make folders if they dont exist, specify filename of checkpoint(s) and create checkpoint callback

    path2Allcheckpoints = path2currDir + 'checkpoints/'
    path2run = path2Allcheckpoints + run_directory

    # make this directory if doesnt exist yet
    make_checkpoint_run_folder(path2checkpoints=path2Allcheckpoints, path2run=path2run)

    # construct filename of checkpoints
    filename = f"{task}_loss_{loss}_" + "epoch{epoch:05d}" #+ f"{monitor_name}=" + "{" + f"{monitor}" + ":.3f}"
    if subnets:
        metrics = [(a, 'val/full/' + a + '/mean') for a in ['error', 'mcc', 'f1', 'se', 'ae']]
    else:
        metrics = [(a, 'val/' + a + '/mean') for a in ['error', 'mcc', 'f1', 'se', 'ae']]

    for metric_name, logged_name in metrics:
        filename += f"_{metric_name}" + "{" + f"{logged_name}" + ":.7f}"
    #if 'regress' in task:
    #    logged_name = f"train/{loss}_epoch"
    #    metric_name = f"Train{loss}"
    #    filename += "_" + metric_name + "{" + f"{logged_name}" + ":.7f}"
    filename += f"_seed{rand_seed}_" + 'date&time' + str(datetime.now()).replace(" ", "_")[5:-7].replace("/", "-")

    if 'real' in which_exp or 'pseudo' in which_exp:
        filename = f"{which_exp}-" + filename

    if misc is not None:
        filename = f"{misc}-" + filename

    assert'=' not in filename, f" '=' seems to mess up loading"
    checkpoint_callback_args = {'monitor': monitor,
                                'dirpath': path2run,
                                'verbose': True,
                                'save_last': False,
                                'save_top_k': 1,
                                'auto_insert_metric_name': False,
                                'filename': filename, # <= this will be changed for each split!
                                'mode': mode,
                                'save_on_train_epoch_end': False}

    return checkpoint_callback_args


if __name__ == "__main__":
    print('hi')
