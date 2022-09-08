# Project 1 Report
Name: Samuel Sommerer

USC ID: 6032951260

## Implementation Choices
My model consists of embedding layer, an LSTM with 1 layer, a fully-connected layer, and finally two
fully-connected output layers for predicting actions and targets.

I am fully aware that adding more layers to my LSTM would improve model performance. However,
Training my model on my CPU takes >5 hours even with an LSTM with just 1 layer, so for the sake of
speed/efficiency of testing I opted to stick with the 1 layer.

## Hyperparameters
- Batch Size: 1000
- Number of Epochs: 100
- Loss Functions: Cross-entropy loss
- Optimizer: Adam

Model Architecture:
```commandline
MODEL(
  (lstm): LSTM(32, 256, batch_first=True)
  (fc_1): Linear(in_features=256, out_features=128, bias=True)
  (fc_actions): Linear(in_features=128, out_features=8, bias=True)
  (fc_targets): Linear(in_features=128, out_features=80, bias=True)
  (embedding): Embedding(1000, 32, padding_idx=0)
  (relu): ReLU()
  (softmax): Softmax(dim=1)
)
```

I settled on a batch size of 1000 arbitrarily. Even with a batch size this large, the model takes >5 hours to train.
I tried training the model on smaller batch sizes and those models trained too slowly. 
I chose 100 epochs because it was the default. More than 100 epochs would take too long to train, 
and less than 100 epochs would mean lower training/validation accuracy. I chose cross-entropy as
my loss function because that's the only loss function we discussed in lecture. I chose the Adam
optimizer because it's generally recognized as the best optimizer.

The model architecture I chose pretty much arbitrarily. If I had better computing resources, I'd
add more layers to the LSTM to see if that would boost model performance. I'd also try
adding/removing fully-connected layers and varying the number of nodes in each fully-connected
layer. I'd keep the ReLU activation function because that's an industry standard at this point.
I'd also keep softmax on my last fully-connected layers to normalize my logits.

## Model Performance

Training performance: 
```commandline
train action loss : 855.6660268306732 | train target loss: 2055.868987083435
train action acc : 0.48017341962344606 | train target acc: 0.09326341148128563
```

Validation performance:
```commandline
val action loss : 35.97212517261505 | val target loss: 86.29418277740479
val action acc : 0.475403266205791 | val target acc: 0.08856827973148983
```

It's confusing to me why training loss is so much higher than validation loss. You'd expect it to 
be the other way around. Additionally, in both the training and validation sets, the action
accuracy is much higher than the target accuracy.

These metrics indicate that our model is severely underfitting our data. To remedy this, I'd
increase model complexity by adding more layers and adding more nodes per layer. I'd also swap
out my existing embedding layer with `word2vec` or some other pre-trained word embedding.