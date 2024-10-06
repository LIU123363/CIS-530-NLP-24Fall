# Starter Code 

## Files
The starter code contains 3 python scripts.
* `pos_tagger.py`: You will have to complete this file with the POS Tagger. It also provides evaluation code for your models so that you can focus on estimation and inference parts of the assignment. 
*  `constants.py`: This file contains any constants needed for the assignment. You are not required to use it, but it might be helpful if you want to run multiple experiments with different settings. Alternatively, you could pass command line arguments using `argparse` (see [documentation](https://docs.python.org/3/library/argparse.html)). Note, if you think you should add more constants/hyperparameters feel free to do so. 
* `utils.py`:  This file contains utility functions that are used in `pos_tagger.py`. You can add more helper functions here for better organization of your code. 

## Provided Functions

### POS TAGGER

In `pos_tagger.py` you will find the following functions/methods:
* `evaluate(data, model)`: The function takes as input a data tuple that can be computed from a file using the `utils.load_data` method, and  `POSTagger` model. The goal of the function is to evaluate the POS model on some sentences and gold tags. It computs a few different accuracies: whole-sentence accuracy, per-token accuracy, unkown token accuracy. It also saves a confusion matrix of the input data as `cm.png`. You can refactor the function as you wish, it is just provided as a helpful tool to save you time from writing evaluation code. 
* `POSTagger`: This is the main class which contains your POS Tagger. You will have to implement its methods, as per the write up. Look at the comments to understand what each function does.

### Utils
In `utils.py` you will find a collection of helper functions that are used in `pos_tagger.py`. 
• `infer_sentences(model, sentences, start)`: This function is used to parallelize the inference of a model. It takes as input a `POSTagger` model, the subset of  `sentences` that a single process infers, as well as the `start` index of the input sentence list, i.e. if the evaluation set `sentences` contains 462 sentences, and this particular process infers `sentences[20:50]`, then the function will be called as follows: `infer_sentences(pos_tagger, sentences[20:50], 20)`.
* `compute_prob(model, sentences, tags, start)`: This function is used to compute the probability of the tags given the sentences and the model. Similarly to the `infer_sentences` function, it is used to parallelize the computation, and as such the `sentences` and `tags` arguments are a subset of the original sets, with `start` denoting the start index. 
* `indices(lst, element)`: This helper function returns all indices in which `element` appears in `lst`.
* `load_data(sentence_file, tag_file=None)`: Given a `sentence_file` and an optional `tag_file` this function returns a list of sentences, and tags if `tag_file` is provided. The following hyperaparameters from `constants.py` are relevant for this function:
    - `CAPITALIZATION`: If set to `False`, then all tokens are converted to lowercase, otherwise they maintain default capitalization.
    - `STOP_WORD`: If set to `True` then the token `<STOP>` is appended at the end of each stentence with corresponding tag `<STOP>`.
* `confusion_matrix(tag2idx,idx2tag, pred, gt, fname)`: This function saves a confusion matrix for the given predictions list `pred` and ground truth list `gt` at file `fname`. The arguments `tag2idx` and `idx2tag` are dictionaries that map tags to their corresponding index, and vice-versa. We provide code for their computation in `POSTagger.train`.
