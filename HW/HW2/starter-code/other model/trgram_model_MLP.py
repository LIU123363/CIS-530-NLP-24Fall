from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
from collections import Counter, defaultdict
import heapq
import copy
import re
import string
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from tagger_constants import *

""" Contains the part of speech tagger class. """


def evaluate(data, model, method):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is.

    As per the write-up, you may find it faster to use multiprocessing (code included).

    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n // processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i: None for i in range(n)}
    probabilities = {i: None for i in range(n)}

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i + k], i, method]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time() - start) / 60} minutes.")

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i + k], tags[i:i + k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time() - start) / 60} minutes.")

    token_acc = sum(
        [1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if
                         tags[i][j] == predictions[i][j] and sentences[i][
                             j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx + 1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc / num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values()) / n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))

    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc / num_whole_sent, token_acc, sum(probabilities.values()) / n



class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary."""
        self.data = None
        self.unigram_probs = None
        self.bigram_probs = None
        self.trigram_probs = None
        self.lexical_probs = None
        self.all_tags = []
        self.tag2idx = {}
        self.idx2tag = {}
        self.word2idx = {}
        self.idx2word = {}
        # MLP model and encoders
        self.mlp_model = None
        self.label_encoder = None
        self.vectorizer = None

    def get_unigrams(self):
        N = len(self.all_tags)
        self.unigram_counts = np.zeros(N)
        for tags in self.data[1]:
            for tag in tags:
                self.unigram_counts[self.tag2idx[tag]] += 1
        total_counts = np.sum(self.unigram_counts)
        m, V = 1, N
        self.unigram_probs = (self.unigram_counts + m) / (total_counts + m * V)

    def get_bigrams(self):
        N = len(self.all_tags)
        self.bigram_counts = np.zeros((N, N))
        for tags in self.data[1]:
            for i in range(len(tags) - 1):
                idx_tag1 = self.tag2idx[tags[i]]
                idx_tag2 = self.tag2idx[tags[i + 1]]
                self.bigram_counts[idx_tag1, idx_tag2] += 1

        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.bigram_probs = (self.bigram_counts + k) / (
                self.bigram_counts.sum(axis=1, keepdims=True) + k * N)

        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.9
                lambda_2 = 0.1
            else:
                lambda_1, lambda_2 = LAMBDAS

            self.bigram_probs = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    bigram_count = self.bigram_counts[i, j]
                    unigram_count = self.unigram_counts[j]
                    total_bigrams = self.bigram_counts.sum(axis=1)[i]
                    total_unigrams = self.unigram_counts.sum()
                    prob_bigram = (bigram_count + EPSILON) / (total_bigrams + EPSILON)
                    prob_unigram = (unigram_count + EPSILON) / (total_unigrams + EPSILON)
                    self.bigram_probs[i, j] = lambda_1 * prob_bigram + lambda_2 * prob_unigram

    def get_trigrams(self):
        N = len(self.all_tags)
        self.trigram_counts = np.zeros((N, N, N))
        for tags in self.data[1]:
            for i in range(len(tags) - 2):
                idx_tag1 = self.tag2idx[tags[i]]
                idx_tag2 = self.tag2idx[tags[i + 1]]
                idx_tag3 = self.tag2idx[tags[i + 2]]
                self.trigram_counts[idx_tag1, idx_tag2, idx_tag3] += 1

        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.trigram_probs = (self.trigram_counts + k) / (
                self.trigram_counts.sum(axis=2, keepdims=True) + k * N)

        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.8
                lambda_2 = 0.1
                lambda_3 = 0.1
            else:
                lambda_1, lambda_2, lambda_3 = LAMBDAS

            self.trigram_probs = np.zeros((N, N, N))
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        trigram_count = self.trigram_counts[i, j, k]
                        bigram_count = self.bigram_counts[j, k]
                        unigram_count = self.unigram_counts[k]
                        total_trigrams = self.trigram_counts.sum(axis=2)[i, j]
                        total_bigrams = self.bigram_counts.sum(axis=1)[j]
                        total_unigrams = self.unigram_counts.sum()

                        prob_trigram = (trigram_count + EPSILON) / (total_trigrams + EPSILON)
                        prob_bigram = (bigram_count + EPSILON) / (total_bigrams + EPSILON)
                        prob_unigram = (unigram_count + EPSILON) / (total_unigrams + EPSILON)

                        self.trigram_probs[i, j, k] = (
                            lambda_1 * prob_trigram
                            + lambda_2 * prob_bigram
                            + lambda_3 * prob_unigram
                        )

    def extract_features(self, word):
        features = {}
        features['is_first_upper'] = word[0].isupper()
        features['is_all_upper'] = word.isupper()
        features['is_all_lower'] = word.islower()
        features['is_digit'] = word.isdigit()
        features['contains_digit'] = any(char.isdigit() for char in word)
        features['word_length'] = len(word)
        features['is_punct'] = word in string.punctuation

        # Suffixes
        for m in range(1, 4):
            if len(word) >= m:
                suffix = word[-m:]
                features[f'suffix_{m}'] = suffix
            else:
                features[f'suffix_{m}'] = '<NONE>'

        # Prefixes
        for m in range(1, 4):
            if len(word) >= m:
                prefix = word[:m]
                features[f'prefix_{m}'] = prefix
            else:
                features[f'prefix_{m}'] = '<NONE>'

        return features

    def get_emissions(self, threshold=None):
        word_counts = defaultdict(int)
        for words in self.data[0]:
            for word in words:
                word_counts[word] += 1

        # Initialize word2idx and idx2word for known words
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        idx = 1
        for word, count in word_counts.items():
            if count >= UNK_C:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        N_tag = len(self.all_tags)
        N_word = len(self.word2idx)
        self.lexical_counts = np.zeros((N_tag, N_word))

        # Initialize data for training MLP
        X_train = []
        y_train = []

        for words, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(words, tags):
                idx_tag = self.tag2idx[tag]
                features = self.extract_features(word)
                X_train.append(features)
                y_train.append(tag)

                if word in self.word2idx:
                    idx_word = self.word2idx[word]
                    self.lexical_counts[idx_tag, idx_word] += 1

        # Compute emission probabilities for known words
        if SMOOTHING == LAPLACE:
            m, V = 1, N_word
            self.lexical_probs = (self.lexical_counts + m) / (
                self.lexical_counts.sum(axis=1, keepdims=True) + m * V)
        elif SMOOTHING == INTERPOLATION:
            # Implement interpolation smoothing if needed
            pass

        # Convert features to feature vectors
        self.vectorizer = DictVectorizer(sparse=False)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)

        # Train MLP model
        if X_train and y_train:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_train)

            # Initialize and train MLPClassifier
            self.mlp_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300)
            self.mlp_model.fit(X_train_vectorized, y_encoded)

    def train(self, data, emission_threshold=None):
        """Trains the model by computing transition and emission probabilities."""
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Compute probabilities
        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()
        self.get_emissions(emission_threshold)

    def inference(self, method, sequence):
        if method == 'viterbi':
            return self.viterbi(sequence)
        elif method == 'beam':
            return self.beam_search(sequence, k=5)
        elif method == 'greedy':
            return self.greedy_decoding(sequence)
        else:
            raise ValueError("Unknown decoding method.")

    def greedy_decoding(self, sequence):
        tag_pred = []
        N_tag = len(self.all_tags)
        min_prob = 1e-10
        for i, word in enumerate(sequence):
            idx_word = self.word2idx.get(word, -1)

            features = self.extract_features(word)
            X_test = self.vectorizer.transform([features])

            if self.mlp_model:
                mlp_emission_probs = np.zeros(N_tag)
                probs = self.mlp_model.predict_proba(X_test)[0]
                for j, tag in enumerate(self.label_encoder.classes_):
                    idx = self.tag2idx[tag]
                    mlp_emission_probs[idx] = probs[j]
            else:
                mlp_emission_probs = np.ones(N_tag) / N_tag

            if idx_word != -1:
                # Known word
                lexical_emission_probs = self.lexical_probs[:, idx_word]
            else:
                # Unknown word
                lexical_emission_probs = np.zeros(N_tag)

            # Combine the emission probabilities
            lambda_emission = 0.5  # Adjust as needed
            emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs

            # Avoid zero probabilities
            emission_probs = np.maximum(emission_probs, min_prob)
            emission_probs /= emission_probs.sum()

            prob_cur = float('-inf')
            tag_cur = None
            for idx in range(N_tag):
                emission_prob = emission_probs[idx]
                if i == 0:
                    transition_prob = max(self.unigram_probs[idx], min_prob)
                elif i == 1:
                    prev_tag_idx = self.tag2idx[tag_pred[-1]]
                    transition_prob = max(self.bigram_probs[prev_tag_idx, idx], min_prob)
                else:
                    prev_tag_idx1 = self.tag2idx[tag_pred[-2]]
                    prev_tag_idx2 = self.tag2idx[tag_pred[-1]]
                    transition_prob = max(self.trigram_probs[prev_tag_idx1, prev_tag_idx2, idx], min_prob)

                total_log_prob = np.log(emission_prob) + np.log(transition_prob)
                if total_log_prob > prob_cur:
                    prob_cur = total_log_prob
                    tag_cur = self.idx2tag[idx]
            tag_pred.append(tag_cur)
        return tag_pred

    def beam_search(self, sequence, k):
        """Tags a sequence with part of speech tags using beam search."""
        N_word = len(sequence)
        N_tag = len(self.all_tags)
        min_prob = 1e-10  # Prevent log(0)

        # Initialize beam
        beam = []
        word = sequence[0]
        idx_word = self.word2idx.get(word, -1)

        features = self.extract_features(word)
        X_test = self.vectorizer.transform([features])

        if self.mlp_model:
            mlp_emission_probs = np.zeros(N_tag)
            probs = self.mlp_model.predict_proba(X_test)[0]
            for j, tag in enumerate(self.label_encoder.classes_):
                idx = self.tag2idx[tag]
                mlp_emission_probs[idx] = probs[j]
        else:
            mlp_emission_probs = np.ones(N_tag) / N_tag

        if idx_word != -1:
            # Known word
            lexical_emission_probs = self.lexical_probs[:, idx_word]
        else:
            # Unknown word
            lexical_emission_probs = np.zeros(N_tag)

        # Combine emission probabilities
        lambda_emission = 0.5  # Adjust as needed
        emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs
        emission_probs = np.maximum(emission_probs, min_prob)
        emission_probs /= emission_probs.sum()

        for i in range(N_tag):
            emission_prob = emission_probs[i]
            transition_prob = max(self.unigram_probs[i], min_prob)
            total_log_prob = np.log(emission_prob) + np.log(transition_prob)
            path = [i]
            heapq.heappush(beam, (-total_log_prob, path))

        beam = heapq.nsmallest(k, beam)

        for t in range(1, N_word):
            candidates = []
            word = sequence[t]
            idx_word = self.word2idx.get(word, -1)

            features = self.extract_features(word)
            X_test = self.vectorizer.transform([features])

            if self.mlp_model:
                mlp_emission_probs = np.zeros(N_tag)
                probs = self.mlp_model.predict_proba(X_test)[0]
                for j, tag in enumerate(self.label_encoder.classes_):
                    idx = self.tag2idx[tag]
                    mlp_emission_probs[idx] = probs[j]
            else:
                mlp_emission_probs = np.ones(N_tag) / N_tag

            if idx_word != -1:
                # Known word
                lexical_emission_probs = self.lexical_probs[:, idx_word]
            else:
                # Unknown word
                lexical_emission_probs = np.zeros(N_tag)

            # Combine emission probabilities
            emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs
            emission_probs = np.maximum(emission_probs, min_prob)
            emission_probs /= emission_probs.sum()

            for neg_log_prob, path in beam:
                for j in range(N_tag):
                    emission_prob = emission_probs[j]
                    if emission_prob == 0:
                        continue
                    if t == 1:
                        prev_tag_idx = path[-1]
                        transition_prob = max(self.bigram_probs[prev_tag_idx, j], min_prob)
                    else:
                        prev_tag_idx1 = path[-2]
                        prev_tag_idx2 = path[-1]
                        transition_prob = max(self.trigram_probs[prev_tag_idx1, prev_tag_idx2, j], min_prob)
                    if transition_prob == 0:
                        continue
                    total_log_prob = -neg_log_prob + np.log(transition_prob) + np.log(emission_prob)
                    heapq.heappush(candidates, (-total_log_prob, path + [j]))
            beam = heapq.nsmallest(k, candidates)

        if beam:
            _, tag_pred_idx = min(beam)
            tag_pred = [self.idx2tag[idx] for idx in tag_pred_idx]
            return tag_pred
        else:
            # If no path found, return the most probable tag sequence
            return [self.idx2tag[np.argmax(self.unigram_probs)]] * N_word

    def viterbi(self, sequence):
        """Tags a sequence with part of speech tags using the Viterbi algorithm."""
        N_word = len(sequence)
        N_tag = len(self.all_tags)
        min_prob = 1e-10  # Prevent log(0)

        # Initialize
        pi = np.full((N_word, N_tag, N_tag), float('-inf'))
        backpointer = np.zeros((N_word, N_tag, N_tag), dtype=int)

        # First word
        word = sequence[0]
        idx_word = self.word2idx.get(word, -1)

        features = self.extract_features(word)
        X_test = self.vectorizer.transform([features])

        if self.mlp_model:
            mlp_emission_probs = np.zeros(N_tag)
            probs = self.mlp_model.predict_proba(X_test)[0]
            for j, tag in enumerate(self.label_encoder.classes_):
                idx = self.tag2idx[tag]
                mlp_emission_probs[idx] = probs[j]
        else:
            mlp_emission_probs = np.ones(N_tag) / N_tag

        if idx_word != -1:
            # Known word
            lexical_emission_probs = self.lexical_probs[:, idx_word]
        else:
            # Unknown word
            lexical_emission_probs = np.zeros(N_tag)

        # Combine emission probabilities
        lambda_emission = 0.5  # Adjust as needed
        emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs
        emission_probs = np.maximum(emission_probs, min_prob)
        emission_probs /= emission_probs.sum()

        for u in range(N_tag):
            emission_prob = emission_probs[u]
            pi[0, 0, u] = np.log(self.unigram_probs[u]) + np.log(emission_prob)
            backpointer[0, 0, u] = 0

        # Second word
        if N_word > 1:
            word = sequence[1]
            idx_word = self.word2idx.get(word, -1)

            features = self.extract_features(word)
            X_test = self.vectorizer.transform([features])

            if self.mlp_model:
                mlp_emission_probs = np.zeros(N_tag)
                probs = self.mlp_model.predict_proba(X_test)[0]
                for j, tag in enumerate(self.label_encoder.classes_):
                    idx = self.tag2idx[tag]
                    mlp_emission_probs[idx] = probs[j]
            else:
                mlp_emission_probs = np.ones(N_tag) / N_tag

            if idx_word != -1:
                # Known word
                lexical_emission_probs = self.lexical_probs[:, idx_word]
            else:
                # Unknown word
                lexical_emission_probs = np.zeros(N_tag)

            # Combine emission probabilities
            emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs
            emission_probs = np.maximum(emission_probs, min_prob)
            emission_probs /= emission_probs.sum()

            for u in range(N_tag):
                for v in range(N_tag):
                    emission_prob = emission_probs[v]
                    transition_prob = max(self.bigram_probs[u, v], min_prob)
                    pi[1, u, v] = pi[0, 0, u] + np.log(transition_prob) + np.log(emission_prob)
                    backpointer[1, u, v] = 0

        # Recursion
        for t in range(2, N_word):
            word = sequence[t]
            idx_word = self.word2idx.get(word, -1)

            features = self.extract_features(word)
            X_test = self.vectorizer.transform([features])

            if self.mlp_model:
                mlp_emission_probs = np.zeros(N_tag)
                probs = self.mlp_model.predict_proba(X_test)[0]
                for j, tag in enumerate(self.label_encoder.classes_):
                    idx = self.tag2idx[tag]
                    mlp_emission_probs[idx] = probs[j]
            else:
                mlp_emission_probs = np.ones(N_tag) / N_tag

            if idx_word != -1:
                # Known word
                lexical_emission_probs = self.lexical_probs[:, idx_word]
            else:
                # Unknown word
                lexical_emission_probs = np.zeros(N_tag)

            # Combine emission probabilities
            emission_probs = lambda_emission * lexical_emission_probs + (1 - lambda_emission) * mlp_emission_probs
            emission_probs = np.maximum(emission_probs, min_prob)
            emission_probs /= emission_probs.sum()

            for u in range(N_tag):
                for v in range(N_tag):
                    emission_prob = emission_probs[v]
                    max_prob = float('-inf')
                    best_w = 0
                    for w in range(N_tag):
                        transition_prob = max(self.trigram_probs[w, u, v], min_prob)
                        prob = pi[t - 1, w, u] + np.log(transition_prob) + np.log(emission_prob)
                        if prob > max_prob:
                            max_prob = prob
                            best_w = w
                    pi[t, u, v] = max_prob
                    backpointer[t, u, v] = best_w

        # Termination
        max_prob = float('-inf')
        best_u, best_v = 0, 0
        for u in range(N_tag):
            for v in range(N_tag):
                if pi[N_word - 1, u, v] > max_prob:
                    max_prob = pi[N_word - 1, u, v]
                    best_u, best_v = u, v

        # Backtrace
        tags_idx = [0] * N_word
        tags_idx[N_word - 1] = best_v
        tags_idx[N_word - 2] = best_u
        for t in range(N_word - 3, -1, -1):
            tags_idx[t] = backpointer[t + 2, tags_idx[t + 1], tags_idx[t + 2]]

        tag_pred = [self.idx2tag[idx] for idx in tags_idx]
        return tag_pred



if __name__ == "__main__":
#######################################################################################
    pos_tagger = POSTagger()
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    emission_threshold = 2
    pos_tagger.train(train_data, emission_threshold)
#######################################################################################


    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    method = 'beam'
    evaluate(dev_data, pos_tagger, method)

    # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     pred_tags = pos_tagger.inference(method, sentence)  # You can choose 'viterbi', 'beam', 'greedy'
    #     test_predictions.append(pred_tags)
    #
    # # Write predictions to a file
    # with
# open('test_predictions.txt', 'w') as f:
    #     for tags in test_predictions:
    #         f.write(' '.join(tags) + '\n')
