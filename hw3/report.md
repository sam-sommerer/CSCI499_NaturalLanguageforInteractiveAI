# Implementation Choices

## Model Hyperparameters
I essentially chose all hyperparameters arbitrarily, but with a focus on keeping my model training time reasonably short.
For instance, the choice to make the encoder's `embedding_dim` 64 dimensions was arbitrary. 64 is a power of 2 and small
enough that training time won't be too long.

## Encoder/Decoder Implementation
I implemented both my encoder and decoder using an LSTM. This also was an essentially arbitrary choice. I could have used an RNN or a GRU,
but since this task is the same as Homework 1 and I used an LSTM in Homework 1, I figured I might as well use the same model for this
assignment.

For the attention version of the model, I implemented flat attention. The only reason I chose to do this was because it was easier. I imagine
implementing hierarchical attention could improve model performance, but I'm guessing it wouldn't improve flat attention's poor performance
enough to justify reimplementing the entire model.

## Miscellaneous
When calculating accuracy and prefix em, I tried to use pytorch's built-in functions/operations as much as possible to speed up computation.

# Performance/Metrics

For specific numbers, please open `default1.png` (base encoder decoder model) and `attention1.png`.

## Notes
I decided not to implement longest common subsequence as a metric because to do so efficiently would require dynamic programming, and that seems
like too much programming overhead for very little return in terms of information gained about model performance. This is especially true given 
that model accuracy tends to be below 4% and thus one would guess there would be little if any common subsequences at all between predicted and
ground truth labels.

Additionally, even though it was a provided metric, prefix em seems too harsh of a metric. That's why we see it stay basically at zero across the
board.

Finally, I calculated accuracy and prefix em individually for actions and targets as well as jointly to see if the model is learning one versus
the other better for some reason.

## Accuracy

### Training
For the base encoder decoder model, the action accuracy is signficantly better than the target accuracy. This is not the case for the attention
model, where the target accuracy is about 10% more accurate than action accuracy.

The joint accuracy of the attention model during training is orders of magnitude better than the base encoder decoder model, indicating that 
added attention layers may have successfully helped the model learn to attend to relevant parts of the encoder's output!

### Validation
The above analysis also holds true when it comes to validation accuracy. In both the base model and the attention model, however, joint validation
accuracy is higher than joint training accuracy. This may be due to chance, but it's worth noting.

## Prefix EM

### Training
The base encoder decoder model's action prefix em was extremely poor, peaking at slightly above `6 * 1e-5`. Target prefix em was orders of magnitude better but still very poor, peaking around `0.00085`. The attention model achieved `0` on action prefix em across all epochs, but
achieved a high of around `0.00380` target prefix em, much higher than the base model. These results suggest that correctly learning the
first few targets given an instruction sequence is easier than learning the action.

Due to the `0` on the action prefix em, the attention model achieved `0` joint prefix em, obviously scoring lower than the base model.

### Validation
The analysis above also holds true for validation prefix em scores. Interestingly, the attention model also scored `0` across all epochs
during validation. Combined with the `0` from the training data, this indicates that perhaps the attention model is having trouble
attending to the action parts of the input.

## Loss
Across both training and validation, the losses of both the base encoder decoder model and the attention model essentially perform the same.

