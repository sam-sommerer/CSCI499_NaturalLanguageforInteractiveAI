import pickle
import matplotlib.pyplot as plt


pickle_output_dir = "pickles/"
train_losses_fn = pickle_output_dir + "train_losses.pkl"
train_accs_fn = pickle_output_dir + "train_accs.pkl"
val_losses_fn = pickle_output_dir + "val_losses.pkl"
val_accs_fn = pickle_output_dir + "val_accs.pkl"

with open(train_losses_fn, "rb") as f:
    train_losses = pickle.load(f)

with open(train_accs_fn, "rb") as f:
    train_accs = pickle.load(f)

with open(val_losses_fn, "rb") as f:
    val_losses = pickle.load(f)

with open(val_accs_fn, "rb") as f:
    val_accs = pickle.load(f)


fig, axes = plt.subplots(2, 2)
axes[0, 0].plot([i for i in range(len(train_losses))], train_losses)
axes[0, 0].set_title("Training Loss")
axes[0, 1].plot([i for i in range(len(train_accs))], train_accs)
axes[0, 1].set_title("Training Accuracy")
axes[1, 0].plot([i for i in range(len(val_losses))], val_losses)
axes[1, 0].set_title("Validation Loss")
axes[1, 1].plot([i for i in range(len(val_accs))], val_accs)
axes[1, 1].set_title("Validation Accuracy")


plt.show()