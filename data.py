import os
import re

import numpy as np


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"
    elif "cluster" in dataset:
        reg_exp = f"{str(sid).zfill(2)}.npz"
    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)

    return subject_files


def load_data(subject_files, data_from_cluster=False):
    """Load data from subject files."""

    signals = []
    labels = []
    cluster_labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            if data_from_cluster:
                y_cluster = f['y_cluster']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)
            if data_from_cluster:
                y_cluster = y_cluster.astype(np.int32)

            signals.append(x)
            labels.append(y)
            if data_from_cluster:
                cluster_labels.append(y_cluster)

    return signals, labels, sampling_rate, cluster_labels


def my_get_subject_files(files):
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    reg_exp = f"S[C|T][4|7][0-9][0-9][a-zA-Z0-9]+\.npz$"

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)

    return subject_files
