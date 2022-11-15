import tqdm
import torch
import argparse
import numpy as np
import json
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
)

from model import EncoderDecoder


def encode_data(
    data,
    vocab_to_index,
    instruction_cutoff_len,
    label_seq_len,
    actions_to_index,
    targets_to_index,
):
    n_episodes = len(data)

    x = np.zeros((n_episodes, instruction_cutoff_len, 1), dtype=np.int32)
    y = np.zeros((n_episodes, label_seq_len, 2), dtype=np.int32)  # label_seq_len is N

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0

    for episode in data:
        # print(f"episode: {episode}")
        # print(f"episode[0]: {episode[0]}")
        instructions_concat = " ".join(
            [instruction_set[0] for instruction_set in episode]
        )
        actions_targets = [instruction_set[1] for instruction_set in episode]

        instruction = preprocess_string(instructions_concat)
        # action, target = label
        x[idx][0] = vocab_to_index["<start>"]
        jdx = 1

        # encoding instructions for whole episode
        for (
            word
        ) in instruction.split():  # do we need to worry about adding padding here?
            if len(word) > 0:
                x[idx][jdx][0] = (
                    vocab_to_index[word]
                    if word in vocab_to_index
                    else vocab_to_index["<unk>"]
                )
                n_unks += 1 if x[idx][jdx][0] == vocab_to_index["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == instruction_cutoff_len - 1:
                    n_early_cutoff += 1
                    break

        x[idx][jdx][0] = vocab_to_index["<end>"]

        if jdx < instruction_cutoff_len:
            for i in range(jdx, instruction_cutoff_len):
                x[idx][i][0] = vocab_to_index["<pad>"]

        # encoding labels
        for i, (action, target) in enumerate(actions_targets):
            # do we need to add padding to action, target?
            if i == label_seq_len:
                break

            y[idx][i][0] = actions_to_index[action]
            y[idx][i][1] = targets_to_index[target]

            if i == len(actions_targets) - 1 and i + 1 < label_seq_len:
                y[idx][i + 1][0] = actions_to_index["<EOS>"]
                y[idx][i + 1][1] = targets_to_index["<EOS>"]

        # y[idx][0] = actions_to_index[action]
        # y[idx][1] = targets_to_index[target]
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(vocab_to_index))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, instruction_cutoff_len)
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
    # train set and validataion set and create respective
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
        avg_num_labels,
    ) = build_output_tables(train_data)
    train_data = [episode for episode in train_data]
    # print(f"train_data[0]: {train_data[0]}")
    train_np_x, train_np_y = encode_data(
        data=train_data,
        vocab_to_index=vocab_to_index,
        instruction_cutoff_len=len_cutoff * 4,
        label_seq_len=avg_num_labels,
        actions_to_index=actions_to_index,
        targets_to_index=targets_to_index,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_np_x), torch.from_numpy(train_np_y)
    )

    validation_data = [episode for episode in validation_data]
    validation_np_x, validation_np_y = encode_data(
        data=validation_data,
        vocab_to_index=vocab_to_index,
        instruction_cutoff_len=len_cutoff * 4,
        label_seq_len=avg_num_labels,
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

    num_actions = len(index_to_actions) - 3
    num_targets = len(index_to_targets) - 3
    vocab_size = len(vocab_to_index)

    return (
        train_loader,
        validation_loader,
        len_cutoff,
        avg_num_labels,
        num_actions,
        num_targets,
        vocab_size,
    )


def setup_model(args, vocab_size, num_actions, num_targets, num_predictions):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = EncoderDecoder(
        vocab_size=vocab_size,
        encoder_embedding_dim=64,
        encoder_hidden_size=128,
        encoder_num_layers=1,
        decoder_hidden_size=128,
        decoder_num_layers=1,
        num_actions=num_actions,
        num_targets=num_targets,
        batch_first=True,
        num_predictions=num_predictions,
    )
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
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
    num_actions,
    num_targets,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs, labels, teacher_forcing=True)
        actions_out = torch.tensor(actions_out)
        targets_out = torch.tensor(targets_out)

        # print(f"actions_out.shape: {actions_out.shape}")
        # print(f"targets_out.shape: {targets_out.shape}")

        # print(f"len(actions_out): {len(actions_out)}")
        # print(f"len(targets_out): {len(targets_out)}")

        # print(f"actions_out[0].shape: {actions_out[0].shape}")

        # loss = criterion(output.squeeze(), labels[:, 0].long())

        action_targets = labels[:, :, 0]
        target_targets = labels[:, :, 1]

        # print(f"labels.shape: {labels.shape}")
        #
        # print(f"action_targets.shape: {action_targets.shape}")

        true_actions_one_hots = torch.nn.functional.one_hot(
            action_targets.long(), num_classes=num_actions + 3
        ).float()
        true_targets_one_hots = torch.nn.functional.one_hot(
            target_targets.long(), num_classes=num_targets + 3
        ).float()

        true_actions_one_hots.requires_grad = True
        true_targets_one_hots.requires_grad = True

        action_loss = action_criterion(actions_out.float(), true_actions_one_hots)
        target_loss = target_criterion(targets_out.float(), true_targets_one_hots)

        # print(f"action_loss.shape: {action_loss.shape}")
        # print(f"target_loss.shape: {target_loss.shape}")
        #
        # print(f"action_loss: {action_loss}")

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        # em = output == labels
        # prefix_em = prefix_em(output, labels)
        acc = 0.0

        # logging
        # epoch_loss += loss.item()
        epoch_loss += loss
        # epoch_acc += acc.item()

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, device, num_actions, num_targets):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
            num_actions=num_actions,
            num_targets=num_targets
        )

    return val_loss, val_acc


def train(
    args,
    model,
    loaders,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    num_actions,
    num_targets,
):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
            num_actions,
            num_targets,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
                num_actions,
                num_targets
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    (
        train_loader,
        val_loader,
        len_cutoff,
        avg_num_labels,
        num_actions,
        num_targets,
        vocab_size,
    ) = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(
        args, vocab_size=vocab_size, num_actions=num_actions, num_targets=num_targets, num_predictions=avg_num_labels
    )
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
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
            args,
            model,
            loaders,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            num_actions=num_actions,
            num_targets=num_targets,
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
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
