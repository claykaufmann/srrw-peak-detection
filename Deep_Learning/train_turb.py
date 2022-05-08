# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, train_test_split
import sys
from sklearn import preprocessing
from resnet import ResNet1D
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torchsummary import summary
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sn

sys.path.insert(1, "../")

from datasets import turbidityDataset, turbAugOnlyDataset, collate_fn_pad
from functools import partial

# Hyperparams
WINDOW_SIZE = 15  # the size of each data segment
TEST_SIZE = 0.10
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

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


def train(model, optimizer, loss_fn, dataloader, device):
    """
    train the model
    """

    model.train()

    total_loss = 0

    with tqdm(dataloader, unit="batch") as prog_bar:
        for i, batch in enumerate(prog_bar):
            if i == len(prog_bar) - 1:
                break

            x = batch[0].to(device)

            # squeeze y to flatten predictions into 1d tensor
            y = batch[1].squeeze().to(device)

            if x.shape[1] != 4:
                print(x.shape)

            pred = model(x.float())

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

            # update progress bar
            prog_bar.set_postfix(loss=loss.item())
            prog_bar.update()

    return total_loss / len(dataloader)


def validation(model, loss_fn, dataloader, device):
    """
    val loop
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as prog_bar:
            for i, batch in enumerate(prog_bar):
                x = batch[0].to(device)

                y = batch[1].squeeze().to(device)

                preds = model(x.float())
                loss = loss_fn(preds, y)

                total_loss += loss.detach().item()

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update()

    return total_loss / len(dataloader)


def fit(
    model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device, scheduler
):
    """
    fit the model
    """
    train_loss_list, validation_loss_list = [], []

    print("Fitting model...")

    for epoch in range(epochs):
        print(f"---------------- EPOCH {epoch + 1} ----------------")

        train_loss = train(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]

        validation_loss = validation(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]

        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {validation_loss}")
        print()

        # step scheduler
        scheduler.step()

    return train_loss_list, validation_loss_list


def test(model, dataloader, device, le):
    """
    test the model
    """
    model.load_state_dict(torch.load("./results/models/turb/raw/may-sixth.pth"))
    model.eval()

    y_pred = []
    y_true = []

    prog_bar = tqdm(dataloader, desc="Testing", leave=False)
    with torch.no_grad():
        for i, batch in enumerate(prog_bar):
            x = batch[0].to(device)

            y = batch[1].squeeze().to(device)

            outs = model(x.float())

            _, preds = torch.max(outs, 1)

            for label, prediction in zip(y, preds):
                # convert label and prediction to current vals
                label = le.inverse_transform([label])[0]
                prediction = le.inverse_transform([prediction])[0]

                y_pred.append(prediction)
                y_true.append(label)

    return y_pred, y_true


def main():
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    classes = ["NAP", "FPT", "PP", "SKP"]
    le = preprocessing.LabelEncoder()

    targets = le.fit_transform(classes)

    train_dataset = turbAugOnlyDataset(
        le,
        fdom_raw_augmented,
        stage_augmented_data_fn,
        turb_augmented_raw_data,
        turb_labeled_augmented,
        turb_fpt_lookup_path,
        WINDOW_SIZE,
    )

    test_dataset = turbidityDataset(
        le,
        fdom_raw_data,
        stage_raw_data,
        turb_raw_data,
        turb_labeled,
        window_size=WINDOW_SIZE,
    )

    # split training into train and validation
    train_size = int(0.85 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size]
    )

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn_pad, device=device),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn_pad, device=device),
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    model = model.float()

    # Optimizer/criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.1)

    criterion = nn.CrossEntropyLoss().to(device)

    # call fit function
    train_loss_list, validation_loss_list = fit(
        model,
        optimizer,
        criterion,
        trainloader,
        val_loader,
        EPOCHS,
        device,
        lr_scheduler,
    )

    torch.save(model.state_dict(), "./results/models/turb/raw/may-sixth.pth")

    # TEST MODEL
    y_pred, y_true = test(model, testloader, device, le)

    # build conf matrix
    conf = confusion_matrix(y_true, y_pred, labels=classes)

    # review the classnames here
    df_cm = pd.DataFrame(
        conf / conf.sum(axis=1)[:, np.newaxis],
        index=[i for i in classes],
        columns=[i for i in classes],
    )

    # classification report
    acc_report = classification_report(y_true, y_pred)
    print(acc_report)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced accuracy: {bal_acc}")

    # display conf matrix
    plt.figure(figsize=(12, 7))

    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")
    plt.title(label="Turbidity Peak Detection Ratio Confusion Matrix")

    plot = sn.heatmap(df_cm, annot=True)

    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")
    plt.title(label="Turbidity Peak Detection Ratio Confusion Matrix")
    plt.show()

    # might be a better way to save?
    # plt.imsave("./results/graphics/turb/raw/may-6.png")
    plot.get_figure().savefig("./results/graphics/turb/raw/may-6.png")


main()
