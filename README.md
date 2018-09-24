# Word2Vec Experiment

Some quick "Getting Started" tools for a Word2Vec experiment.

## train.py

Train a Word2Vec model on some input corpus. Expects entire corpus to be concatenated into one file with one sentence per line.

Takes two arguments - input corpus file and output model file

## interact.py

Load a model and allow the user to query for terms that are similar to input. Type "\quit" to exit the program.

Takes one argument: The model from which similarities are derived

## evaluate.py

Loads two models and computes the amount of agreement between their representation of terms using Average Jaccard Similarity

Takes two arguments: Two Word2Vec models to be used for evaluation
