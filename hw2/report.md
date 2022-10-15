# Report Homework 2

## Implementation Choices

I chose the Adam optimizer because it's industry standard. I converted the labels and model output from a set of indices
to a multihot vector so that BCEWithLogitsLoss can work with them. I then implemented my own accuracy function to
calculate accuracy of the multihot prediction vectors compared to the multihot label vectors. Sci-kit learn's accuracy
score function doesn't support multiclass multi-output accuracy scoring.

I chose an embedding dimension of 128 arbitrarily, it's a power of 2 and common number to choose. I chose to use 30
epochs because it was the default and because any larger number would require too much time to train. It would be harder
to test hyperparameters using more epochs.

## Bonus

I implemented a skip-gram model.

## Analysis

In vitro tasks: The in vitro task here is learning the context vector of an input word given the input word. This is
measured by an accuracy score comparing the true multihot vector of the context versus the predicted multihot vector.

In vivo tasks: The in vivo task here is the analogy task we talked about in class. We investigate whether or not the
embeddings that we learn from the skip-gram model lead to the close distance of words in the embedding space that we 
think should be close to each other. The distance between king and queen should be similar to man and woman. We use
MMR (mean reciprocal rank). A problem with this is that MRR only considers the first relevant item in its scoring
calculations, not taking into account whether the number of relevant embeddings/words that may have been found. This
consequently may over-penalize an embedding space that has a single irrelevant word close to the target and over-reward
embedding spaces that place a single relevant word close to the target.

