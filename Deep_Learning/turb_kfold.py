# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import sys
from sklearn import preprocessing
from resnet import ResNet1D
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torchsummary import summary
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sn

sys.path.insert(1, "../")

from datasets import turbAugOnlyDataset
import copy
from functools import partial

# Hyperparams
WINDOW_SIZE = 15  # the size of each data segment
SEED = 42
BATCH_SIZE = 32

# this is the number of epochs per fold, but because data is already batched,
#   when larger than 1, training takes a long time
EPOCHS = 3

SPLITS = 5

# Paths to data files
fdom_raw_data = "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
stage_raw_data = "../Data/converted_data/julian_format/stage_10.1.11-1.1.19.csv"
turb_raw_data = (
    "../Data/converted_data/julian_format/turbidity_raw_10.1.2011_9.4.2020.csv"
)

turb_labeled = "../Data/labeled_data/ground_truths/turb/turb_all_julian_0k-300k.csv"

fdom_raw_augmented = "../Data/augmented_data/turb/unlabeled/unlabeled_fdom.csv"
turb_labeled_augmented = "../Data/augmented_data/turb/labeled/labeled_turb_peaks.csv"

turb_augmented_raw_data = "../Data/augmented_data/turb/unlabeled/unlabeled_turb.csv"

stage_augmented_data_fn = "../Data/augmented_data/turb/unlabeled/unlabeled_stage.csv"

turb_fpt_lookup_path = "../Data/augmented_data/turb/fpt_lookup.csv"

# util functions
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            print(f"reset trainable params of layer = {layer}")
            layer.reset_parameters()


def collate_fn_pad(batch, device):
    """
    Pads batch of variable length
    """

    label_list, sample_list, lengths = [], [], []

    for (sample, label) in batch:
        label_list.append(label)
        # convert sample to tensor
        sample = torch.tensor(
            sample, dtype=torch.float64
        ).T  # tranpose to send in data, pad_sequences won't accept original

        # append to lengths
        lengths.append(sample.shape[0])

        sample_list.append(sample)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    sample_list = torch.nn.utils.rnn.pad_sequence(
        sample_list, batch_first=True, padding_value=0
    )

    # re-tranpose list, so we go back to a 4 channel dataset
    sample_list = sample_list.transpose(1, 2)

    lengths = torch.tensor(lengths, dtype=torch.long)

    return [sample_list.to(device), label_list.to(device), lengths]


def main():
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    classes = ["NAP", "FPT", "PP", "SKP"]
    le = preprocessing.LabelEncoder()

    targets = le.fit_transform(classes)

    dataset = turbAugOnlyDataset(
        le,
        fdom_raw_augmented,
        stage_augmented_data_fn,
        turb_augmented_raw_data,
        turb_labeled_augmented,
        turb_fpt_lookup_path,
        WINDOW_SIZE,
    )

    torch.manual_seed(42)
    results = {}

    tss = TimeSeriesSplit(SPLITS)

    criterion = nn.CrossEntropyLoss().to(device)

    # K-fold training
    conf_matrices = {}
    accumulated_metrics = {}

    for fold, (train_ids, test_ids) in enumerate(tss.split(dataset)):
        print(f"FOLD {fold}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=train_subsampler,
            collate_fn=partial(collate_fn_pad, device=device),
        )

        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=test_subsampler,
            collate_fn=partial(collate_fn_pad, device=device),
        )

        # init model
        model = ResNet1D(
            in_channels=4,
            base_filters=64,
            kernel_size=16,
            stride=2,
            n_block=48,
            groups=1,  # check this
            n_classes=len(classes),
            downsample_gap=6,
            increasefilter_gap=12,
            verbose=False,
        ).to(device)

        # set model to use float instead of doubles to prevent errors
        model = model.float()

        # init optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(0, EPOCHS):
            print(f"Starting epoch {epoch + 1}")

            current_loss = 0

            # prog bar
            prog_bar = tqdm(trainloader, desc="Training", leave=False)
            for i, data in enumerate(prog_bar):
                x = data[0].to(device)
                y = data[1].squeeze().to(device)

                if i == len(prog_bar) - 1:
                    break

                optimizer.zero_grad()

                pred = model(x.float())
                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()

                # print stats
                current_loss += loss.item()
                if i % 500 == 499:
                    print(
                        "Loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 500)
                    )
                    current_loss = 0.0

        # completed training, now test
        print(f"Training for fold {fold} has completed, now testing")

        # save best params
        save_path = f"./results/models/turb/kfold/may-9-model-fold={fold}.pth"
        torch.save(model.state_dict(), save_path)

        total, correct = 0, 0

        # for checking correct and incorrect preds
        y_true = []
        y_pred = []

        prog_bar = tqdm(testloader, desc="Testing", leave=False)
        with torch.no_grad():
            for i, data in enumerate(prog_bar):
                x = data[0].to(device)
                y = data[1].squeeze().to(device)

                outputs = model(x.float())

                _, preds = torch.max(outputs, 1)

                for label, prediction in zip(y, preds):
                    # convert label and prediction to current vals
                    label = le.inverse_transform([label])[0]
                    prediction = le.inverse_transform([prediction])[0]

                    # for confusion matrices
                    y_pred.append(prediction)
                    y_true.append(label)

                    if label == prediction:
                        correct += 1
                    total += 1

            # Print rough general accuracy
            print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
            print("--------------------------------")
            results[fold] = 100.0 * (correct / total)

            # make classification report
            acc_report = classification_report(y_true, y_pred)
            print(acc_report)

            # get acc score
            acc_score = accuracy_score(y_true, y_pred)

            bal_acc = balanced_accuracy_score(y_true, y_pred)

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted",
            )

            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
            )

            # make conf matrix
            matrix = confusion_matrix(y_true, y_pred, labels=classes)

            # save conf matrix
            conf_matrices[fold] = copy.deepcopy(matrix)

            # save accumulated metrics
            accumulated_metrics[fold] = {
                "f1": f1,
                "acc": acc_score,
                "ba": bal_acc,
                "precision": precision,
            }

    # Print fold results
    print("\n")
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {SPLITS} FOLDS")
    print("--------------------------------")
    sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        sum += value
    print(f"Average: {sum/len(results.items())} %")

    # save accumulated metrics
    mean_f1 = 0
    mean_ba = 0
    mean_precision = 0
    mean_acc = 0

    for key in accumulated_metrics:
        metrics = accumulated_metrics[key]

        mean_f1 += metrics["f1"]
        mean_ba += metrics["ba"]
        mean_precision += metrics["precision"]
        mean_acc += metrics["acc"]

    print("Mean Test F1: ", mean_f1 / len(accumulated_metrics))
    print("Mean Test BA: ", mean_ba / len(accumulated_metrics))
    print("Mean Test Acc: ", mean_acc / len(accumulated_metrics))
    print("Mean Test Precision: ", mean_precision / len(accumulated_metrics))

    # make mean confusion matrix
    mean_cfmx = np.zeros((len(classes), len(classes)))
    for key in conf_matrices.keys():
        mean_cfmx += conf_matrices[key]

    mean_cfmx = mean_cfmx / len(conf_matrices)

    plt.figure(figsize=(10, 7))
    plt.title(label="Turbidity Peak Detection Ratio Confusion Matrix KFold")

    sn.set(font_scale=1.5)

    plot = sn.heatmap(
        pd.DataFrame(
            mean_cfmx.astype("float") / mean_cfmx.sum(axis=1)[:, np.newaxis],
            index=classes,
            columns=classes,
        ),
        annot=True,
        annot_kws={"size": 16},
    )

    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")
    plt.show()

    plot.get_figure().savefig(
        "./results/graphics/turb/kfold/may-9-conf-ratio-balanced-test.png"
    )

    plt.figure(figsize=(10, 7))
    plt.title(label="Turbidity Peak Detection Totals Confusion Matrix KFold")

    sn.set(font_scale=1.5)

    plot = sn.heatmap(
        pd.DataFrame(
            mean_cfmx,
            index=classes,
            columns=classes,
        ),
        annot=True,
        annot_kws={"size": 16},
    )

    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")
    plt.show()

    plot.get_figure().savefig(
        "./results/graphics/turb/kfold/may-9-conf-totals-balanced-test.png"
    )


main()
