import tqdm
import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from model import LSTM

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)


def encode_data(data, vocab_to_index, seq_len, actions_to_index, targets_to_index):
    n_episodes = len(data)

    x = np.zeros((n_episodes, seq_len, 1), dtype=np.int32)
    y = np.zeros((n_episodes, 2), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0

    for instruction, label in data:
        instruction = preprocess_string(instruction)
        action, target = label
        x[idx][0] = vocab_to_index["<start>"]
        jdx = 1
        for word in instruction.split():
            if len(word) > 0:
                x[idx][jdx][0] = (
                    vocab_to_index[word]
                    if word in vocab_to_index
                    else vocab_to_index["<unk>"]
                )
                n_unks += 1 if x[idx][jdx][0] == vocab_to_index["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx][0] = vocab_to_index["<end>"]
        y[idx][0] = actions_to_index[action]
        y[idx][1] = targets_to_index[target]
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(vocab_to_index))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, y


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validation set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    with open(args.in_data_fn) as f:
        data = json.load(f)
        train_data = data["train"]
        validation_data = data["valid_seen"]

    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data)
    (
        actions_to_index,
        index_to_actions,
        targets_to_index,
        index_to_targets,
    ) = build_output_tables(train_data)
    train_data = [
        instruction for instruction_set in train_data for instruction in instruction_set
    ]
    train_np_x, train_np_y = encode_data(
        data=train_data,
        vocab_to_index=vocab_to_index,
        seq_len=len_cutoff,
        actions_to_index=actions_to_index,
        targets_to_index=targets_to_index,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_np_x), torch.from_numpy(train_np_y)
    )

    validation_data = [
        instruction
        for instruction_set in validation_data
        for instruction in instruction_set
    ]
    validation_np_x, validation_np_y = encode_data(
        data=validation_data,
        vocab_to_index=vocab_to_index,
        seq_len=len_cutoff,
        actions_to_index=actions_to_index,
        targets_to_index=targets_to_index,
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_np_x), torch.from_numpy(validation_np_y)
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    validation_loader = DataLoader(
        validation_dataset, shuffle=True, batch_size=args.batch_size
    )

    num_actions = len(index_to_actions)
    num_targets = len(index_to_targets)
    vocab_size = len(vocab_to_index)

    return (
        train_loader,
        validation_loader,
        len_cutoff,
        num_actions,
        num_targets,
        vocab_size,
    )


def setup_model(args, len_cutoff, num_actions, num_targets, vocab_size, embedding_dim):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = LSTM(
        num_classes=2,
        input_size=1,
        hidden_size=256,
        num_layers=1,
        seq_length=len_cutoff,
        num_actions=num_actions,
        num_targets=num_targets,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.float(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.float(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().detach().numpy())
        target_preds.extend(target_preds_.cpu().detach().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_action_acc_list = list()
    train_action_loss_list = list()
    train_target_acc_list = list()
    train_target_loss_list = list()
    validation_action_acc_list = list()
    validation_action_loss_list = list()
    validation_target_acc_list = list()
    validation_target_loss_list = list()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        train_action_acc_list.append(train_action_acc)
        train_action_loss_list.append(train_action_loss)
        train_target_acc_list.append(train_target_acc)
        train_target_loss_list.append(train_target_loss)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

            validation_action_acc_list.append(val_action_acc)
            validation_action_loss_list.append(val_action_loss)
            validation_target_acc_list.append(val_target_acc)
            validation_target_loss_list.append(val_target_loss)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    fig, axes = plt.subplots(4, 2)
    axes[0, 0].plot(train_action_acc_list, [i for i in range(len(train_action_acc_list))])
    axes[0, 0].set_title("Training Accuracy Actions")
    axes[0, 1].plot(train_action_loss_list, [i for i in range(len(train_action_loss_list))])
    axes[0, 1].set_title("Training Loss Actions")
    axes[1, 0].plot(train_target_acc_list, [i for i in range(len(train_target_acc_list))])
    axes[1, 0].set_title("Training Accuracy Targets")
    axes[1, 1].plot(train_target_loss_list, [i for i in range(len(train_target_loss_list))])
    axes[1, 1].set_title("Training Loss Targets")
    axes[2, 0].plot(validation_action_acc_list, [i for i in range(len(validation_action_acc_list))])
    axes[2, 0].set_title("Validation Accuracy Actions")
    axes[2, 1].plot(validation_action_loss_list, [i for i in range(len(validation_action_loss_list))])
    axes[2, 1].set_title("Validation Loss Actions")
    axes[3, 0].plot(validation_target_acc_list, [i for i in range(len(validation_target_acc_list))])
    axes[3, 0].set_title("Validation Accuracy Targets")
    axes[3, 1].plot(validation_target_loss_list, [i for i in range(len(validation_target_loss_list))])
    axes[3, 1].set_title("Validation Loss Targets")

    plt.show()


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    # train_loader, val_loader, maps = setup_dataloader(args)
    # loaders = {"train": train_loader, "val": val_loader}

    (
        train_loader,
        val_loader,
        len_cutoff,
        num_actions,
        num_targets,
        vocab_size,
    ) = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    # model = setup_model(args, maps, device)
    # print(model)
    model = setup_model(
        args, len_cutoff, num_actions, num_targets, vocab_size, args.embedding_dim
    )
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=5,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=32, help="embedding dimension"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
