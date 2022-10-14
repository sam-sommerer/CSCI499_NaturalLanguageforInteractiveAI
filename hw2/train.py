import argparse
import os
import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from eval_utils import multiclass_multilabel_accuracy_score, downstream_validation
from model import SkipgramModel
import utils
import data_utils


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # print(f"encoded_sentences: {encoded_sentences}")
    # print(f"encoded_sentences.shape: {encoded_sentences.shape}")
    # print(f"lens: {lens}")
    # print(f"lens.shape: {lens.shape}")

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    context_size = args.context_size
    input_words = []
    contexts = []

    for sentence, length in zip(encoded_sentences, lens.squeeze()):
        for i in range(length - ((2 * context_size) + 1) + 1):
            curr_context = sentence[i : i + ((2 * context_size) + 1)]

            input_word = curr_context[context_size + 1]
            curr_context = np.delete(curr_context, context_size + 1)

            input_words.append(input_word)
            contexts.append(curr_context)

    x_train, x_val, y_train, y_val = train_test_split(
        input_words,
        contexts,
        test_size=args.test_split / 100,
        random_state=args.seed_value,
    )

    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_data = TensorDataset(torch.tensor(x_val), torch.tensor(torch.tensor(y_val)))

    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader, index_to_vocab


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = SkipgramModel(vocab_size=args.vocab_size, embedding_dim=args.embedding_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions.
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        # pred_logits = model(inputs, labels)
        pred_logits = model(inputs)
        topk_indices = torch.topk(pred_logits, 2, sorted=False).indices

        #  Both of these should be of shape args.batch_size, args.vocab_size
        multihot_pred_vectors = utils.convert_indices_to_multihot(
            topk_indices, pred_logits.size()
        )
        multihot_label_vectors = utils.convert_indices_to_multihot(
            labels, pred_logits.size()
        )

        multihot_pred_vectors.requires_grad_()
        multihot_label_vectors.requires_grad_()

        # calculate prediction loss
        loss = criterion(
            multihot_pred_vectors.squeeze(), multihot_label_vectors.squeeze()
        )

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        # preds = pred_logits.argmax(-1)
        preds = topk_indices
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    # acc = accuracy_score(
    #     pred_labels, target_labels
    # )  # may need to change this/implement new accuracy function
    acc = multiclass_multilabel_accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    # device = utils.get_device(args.force_cpu)
    # device = torch.device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)  # i2v is index_to_vocab
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="where to save training outputs",
        default="output_dir",
    )
    parser.add_argument(
        "--data_dir", type=str, help="where the book dataset is stored", default="books"
    )
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn",
        type=str,
        help="filepath to the analogies json file",
        default="analogies_v3000_1309.json",
    )
    parser.add_argument(
        "--word_vector_fn",
        type=str,
        help="filepath to store the learned word vectors",
        default="learned_word_vectors.txt",
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    parser.add_argument(
        "--context_size",
        default=2,
        type=int,
        help="context size",
    )

    parser.add_argument(
        "--seed_value",
        default=42,
        type=int,
        help="random seed",
    )

    parser.add_argument(
        "--test_split",
        default=25,
        type=int,
        help="percentage of dataset to put in test split",
    )

    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
        help="dimension of embedding vectors",
    )

    args = parser.parse_args()
    main(args)
