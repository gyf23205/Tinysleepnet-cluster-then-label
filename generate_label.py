import argparse
import glob
import importlib
import os
import shutil
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from data import load_data, my_get_subject_files
# from model import TinySleepNet
from minibatching import (iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import print_n_samples_each_class
from extract_feature import extract_feature


def feature_extract(
    config_file,
    model_output_dir,
    data_output_dir,
    model_based=False
):

    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory, model is not writing anything to this dir in this function
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    subject_files = glob.glob(os.path.join(config["feature_extract_file_dir"], "*.npz"))
    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    # model = TinySleepNet(
    #     config=config,
    #     output_dir=model_output_dir,
    #     use_rnn=True,
    #     testing=True,
    #     use_best=True,
    #     feature_extract=True
    # )
    test_files = my_get_subject_files(files=subject_files)
    test_x, test_y, fs, _ = load_data(test_files, data_from_cluster=False)
    # Print test set
    print_n_samples_each_class(np.hstack(test_y))
    logits = extract_feature(test_x, fs)
    labels = []
    for i in range(len(test_y)):
        for j in test_y[i]:
            labels.append(j)
    ground_truth = labels
    # logits, labels, ground_truth = pred_logits(config, test_x, test_y, model)
    generated_labels = []

    class_num = 5
    # clustering
    if model_based:
        cluster_train_subject_files = glob.glob(os.path.join(config["feature_extract_train_file_dir"], "*.npz"))
        train_files = my_get_subject_files(files=cluster_train_subject_files)
        cluster_train_X, cluster_train_y, fs1, _ = load_data(train_files, data_from_cluster=False)
        cluster_train_logits=extract_feature(cluster_train_X, fs1)
        # cluster_train_logits, _, _ = pred_logits(config, cluster_train_X, cluster_train_y, model)
        cluster_label = clustering(cluster_train_logits, logits, class_num, True)
    else:
        cluster_label = clustering(None, logits, class_num, False)
    label_map = np.zeros((class_num, class_num))
    label_map_true = np.zeros((class_num, class_num))
    for i in range(cluster_label.shape[0]):
        label_map[cluster_label[i], labels[i]] += 1
    print(label_map)
    print()
    for i in range(cluster_label.shape[0]):
        label_map_true[cluster_label[i], ground_truth[i]] += 1
    print(label_map_true)
    # correct labels, make data in the same cluster have the same label
    corrected_label = -1 * np.ones((label_map.shape[0],))
    ratio = np.zeros((class_num, class_num))
    for i in range(label_map.shape[0]):
        cluster_sum = np.sum(label_map[i])
        ratio[i] = label_map[i]/cluster_sum
    print(ratio)
    for i in range(class_num):
        # corrected_label[np.where(ratio[:, i] == np.amax(ratio[:, i]))[0][0]] = i
        corrected_label[i] = np.where(ratio[i] == np.amax(ratio[i]))[0][0]
    print(corrected_label)
    for i in range(cluster_label.shape[0]):
        generated_labels.append(int((corrected_label[cluster_label[i]])))
    assert len(labels) == len(generated_labels)
    # split data into n_folds files
    n_folds = 6
    len_per_file = len(labels)//n_folds
    flatten_x = []
    for i in range(len(test_x)):
        for j in range(test_x[i].shape[0]):
            flatten_x.append(test_x[i][j])
    new_x = []
    new_cluster_label = []
    new_true_label = []
    start = 0
    correct_num = 0
    total_num = 0
    for i in range(n_folds):
        x = []
        y = []
        y_cluster = []
        if start+len_per_file <= len(flatten_x):
            end = start+len_per_file
        else:
            end = len(flatten_x)
        x.extend(flatten_x[start:end])
        y.extend(ground_truth[start:end])
        y_cluster.extend((generated_labels[start:end]))
        start += len_per_file
        new_x.append(np.array(x))
        new_true_label.append(np.array(y))
        new_cluster_label.append((np.array(y_cluster)))
        correct_num +=np.sum(np.array(y) == np.array(y_cluster))
        total_num +=len(x)
    print(correct_num)
    print(total_num)
    print(correct_num/total_num)
    # Output dir
    # if not os.path.exists(data_output_dir):
    #     os.makedirs(data_output_dir)
    # else:
    #     shutil.rmtree(data_output_dir)
    #     os.makedirs(data_output_dir)
    # for i in range(len(new_true_label)):
    #     # Save
    #     filename = f"{str(i).zfill(2)}.npz"
    #     save_dict = {
    #         "x": new_x[i].astype(np.float32),
    #         "y": new_true_label[i].astype(np.int32),
    #         "y_cluster": new_cluster_label[i].astype(np.int32),
    #         "fs": 100
    #     }
    #     np.savez(os.path.join(data_output_dir, filename), **save_dict)
    return


def pred_logits(config, X, y, model):
    """Given signal, generate labels using tinysleepnet"""
    logits = []
    labels = []
    ground_truth = []
    if config["model"] == "model-origin":
        for night_idx, night_data in enumerate(zip(X, y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_seq_minibatches(
                night_x,
                night_y,
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
            )
            # Evaluate
            test_outs = model.evaluate(test_minibatch_fn)
            logits.extend(test_outs["test/logits"])
            labels.extend(test_outs["test/preds"])
            ground_truth.extend(test_outs["test/trues"])
    else:
        for night_idx, night_data in enumerate(zip(X, y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            if (config.get('augment_signal') is not None) and config['augment_signal']:
                # Evaluate
                test_outs = model.evaluate_aug(test_minibatch_fn)
            else:
                # Evaluate
                test_outs = model.evaluate(test_minibatch_fn)
            logits.extend(test_outs["test/logits"])
            labels.extend(test_outs["test/preds"])
            ground_truth.extend(test_outs["test/trues"])
    tf.reset_default_graph()
    return logits, labels, ground_truth


def clustering(train_X, pred_X, class_num, model_based):
    if model_based:
        train_X = np.squeeze(train_X)
        gm = GaussianMixture(n_components=class_num, random_state=0).fit(train_X)
        cluster_label = gm.predict(np.squeeze(pred_X))
    else:
        cluster_label = KMeans(n_clusters=class_num, random_state=1).fit(pred_X).labels_
    return cluster_label


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", type=str, required=True)
    # parser.add_argument("--model_based", type=ast.literal_eval, required=True)
    # parser.add_argument("--model_output_dir", type=str, default="./out_sleepedf/feature_extract")
    # parser.add_argument("--data_output_dir", type=str, default="./my_data/labels_from_cluster_data16")
    # args = parser.parse_args()
    # # --config_file config/cluster.py
    # # --model_output_dir out_sleepedf/feature_extract
    # feature_extract(config_file=args.config_file,
    #                 model_output_dir=args.model_output_dir,
    #                 data_output_dir=args.data_output_dir,
    #                 model_based=args.model_based)
    feature_extract(config_file="config/cluster.py",
                    model_output_dir="./out_sleepedf/feature_extract",
                    data_output_dir="./my_data/labels_from_cluster_data16",
                    model_based=True)





