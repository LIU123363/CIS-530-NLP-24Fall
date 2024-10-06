from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
from collections import Counter, defaultdict
import heapq
import copy
from tagger_constants import *

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

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



import numpy as np
from collections import defaultdict, Counter
import heapq
from tagger_constants import *
import copy

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
        # unknown words parameter
        self.suffix_tag_probs = {}
        self.unknown_tag_probs = None

    def good_turing_adjust_counts(self, counts):
        """
        Apply Good-Turing smoothing to counts.
        counts: numpy array of counts (could be 1D, 2D, or 3D)
        Returns adjusted counts.
        """
        flat_counts = counts.flatten()
        unique_counts, counts_of_counts = np.unique(flat_counts, return_counts=True)
        N_c = dict(zip(unique_counts, counts_of_counts))
        max_count = int(np.max(unique_counts))

        adjusted_flat_counts = np.zeros_like(flat_counts, dtype=float)
        total_types = np.sum(counts > 0)

        # Compute the total number of possible n-grams
        if counts.ndim == 1:
            total_possible = counts.shape[0]
        elif counts.ndim == 2:
            total_possible = counts.shape[0] * counts.shape[1]
        elif counts.ndim == 3:
            total_possible = counts.shape[0] * counts.shape[1] * counts.shape[2]

        N_0 = total_possible - total_types
        N_c[0] = N_0  # Number of unseen n-grams

        for idx, c in enumerate(flat_counts):
            c = int(c)
            N_c_c = N_c.get(c, 0)
            N_c_c1 = N_c.get(c + 1, 0)
            if c < max_count:
                if N_c_c > 0:
                    c_star = ((c + 1) * N_c_c1) / N_c_c
                else:
                    c_star = 0
            else:
                c_star = c
            adjusted_flat_counts[idx] = c_star

        adjusted_counts = adjusted_flat_counts.reshape(counts.shape)
        return adjusted_counts

    def get_unigrams(self):
        """
        Computes unigrams with smoothing.
        """
        N = len(self.all_tags)
        self.unigram_counts = np.zeros(N)
        for tags in self.data[1]:
            for tag in tags:
                self.unigram_counts[self.tag2idx[tag]] += 1
        total_counts = np.sum(self.unigram_counts)

        if SMOOTHING == LAPLACE:
            m, V = 1, N
            self.unigram_probs = (self.unigram_counts + m) / (total_counts + m * V)
        elif SMOOTHING == INTERPOLATION:
            m, V = 1, N
            self.unigram_probs = (self.unigram_counts + m) / (total_counts + m * V)
        elif SMOOTHING == GOODTURING:
            # Good-Turing smoothing for unigrams
            counts = self.unigram_counts
            adjusted_counts = self.good_turing_adjust_counts(counts)
            adjusted_total = np.sum(adjusted_counts)
            self.unigram_probs = adjusted_counts / adjusted_total

    def get_bigrams(self):
        """
        Computes bigrams with smoothing.
        """
        N = len(self.all_tags)
        self.bigram_counts = np.zeros((N, N))
        for tags in self.data[1]:
            for i in range(len(tags) - 1):
                idx_tag1 = self.tag2idx[tags[i]]
                idx_tag2 = self.tag2idx[tags[i + 1]]
                self.bigram_counts[idx_tag1, idx_tag2] += 1

        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.bigram_probs = (self.bigram_counts + k) / (self.bigram_counts.sum(axis=1, keepdims=True) + k * N)
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
        elif SMOOTHING == GOODTURING:
            # Good-Turing smoothing for bigrams
            counts = self.bigram_counts
            adjusted_counts = self.good_turing_adjust_counts(counts)
            row_sums = adjusted_counts.sum(axis=1, keepdims=True)
            # To handle zero row sums
            row_sums[row_sums == 0] = 1
            self.bigram_probs = adjusted_counts / row_sums

    def get_trigrams(self):
        """
        Computes trigrams with smoothing.
        """
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
        elif SMOOTHING == GOODTURING:
            # Good-Turing smoothing for trigrams
            counts = self.trigram_counts
            adjusted_counts = self.good_turing_adjust_counts(counts)
            sum_over_k = adjusted_counts.sum(axis=2, keepdims=True)
            sum_over_k[sum_over_k == 0] = 1  # Avoid division by zero
            self.trigram_probs = adjusted_counts / sum_over_k

    def get_emissions(self, threshold=None):
        """
        Computes emission probabilities with TnT-style suffix handling for unknown words.
        """
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

        # Initialize suffix_tag_counts for unknown word
        suffix_tag_counts = defaultdict(lambda: np.zeros(N_tag))

        for words, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(words, tags):
                idx_tag = self.tag2idx[tag]
                if word in self.word2idx:
                    idx_word = self.word2idx[word]
                    self.lexical_counts[idx_tag, idx_word] += 1
                else:
                    # Collect suffix statistics for low-frequency words
                    for m in range(UNK_M, 0, -1):
                        if len(word) >= m:
                            suffix = word[-m:]
                            suffix_tag_counts[suffix][idx_tag] += 1

        # Compute the emission probabilities
        if SMOOTHING == LAPLACE:
            m, V = 1, N_word
            self.lexical_probs = (self.lexical_counts + m) / (
                self.lexical_counts.sum(axis=1, keepdims=True) + m * V)
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.9
                lambda_2 = 0.1
            else:
                lambda_1, lambda_2 = LAMBDAS

            total_words = sum(word_counts.values())
            word_unigram_probs = np.zeros(N_word)
            for word, idx in self.word2idx.items():
                word_unigram_probs[idx] = (word_counts[word] + EPSILON) / (total_words + EPSILON)

            self.lexical_probs = np.zeros((N_tag, N_word))
            for i in range(N_tag):
                for j in range(N_word):
                    prob_emission = (self.lexical_counts[i, j] + EPSILON) / (
                        self.lexical_counts.sum(axis=1)[i] + EPSILON)
                    prob_word_unigram = word_unigram_probs[j]
                    self.lexical_probs[i, j] = lambda_1 * prob_emission + lambda_2 * prob_word_unigram
        elif SMOOTHING == GOODTURING:
            # Good-Turing smoothing for emissions
            counts = self.lexical_counts
            adjusted_counts = self.good_turing_adjust_counts(counts)
            row_sums = adjusted_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.lexical_probs = adjusted_counts / row_sums

        # Convert suffix_tag_counts to probabilities
        self.suffix_tag_probs = {}
        for suffix, counts in suffix_tag_counts.items():
            total_counts = counts.sum()
            if total_counts > 0:
                self.suffix_tag_probs[suffix] = counts / total_counts
            else:
                self.suffix_tag_probs[suffix] = np.ones(N_tag) / N_tag

        # For unknown words, default to uniform distribution
        self.unknown_tag_probs = np.ones(N_tag) / N_tag

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

    def sequence_probability(self, sequence, tags):
        sequence_log_prob = 0.0
        N_tag = len(self.all_tags)
        min_prob = 1e-10
        for i, word in enumerate(sequence):
            idx_tag = self.tag2idx[tags[i]]

            # Get the index of the word
            idx_word = self.word2idx.get(word, -1)
            if idx_word == -1:
                # Unknown word
                suffix_probs = self.unknown_tag_probs
                for m in range(UNK_M, 0, -1):
                    if len(word) >= m:
                        suffix = word[-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break  # Use the longest matching suffix
                emission_prob = max(suffix_probs[idx_tag], min_prob)
            else:
                emission_prob = max(self.lexical_probs[idx_tag, idx_word], min_prob)

            if i == 0:
                transition_prob = max(self.unigram_probs[idx_tag], min_prob)
            elif i == 1:
                idx_pre_tag = self.tag2idx[tags[i - 1]]
                transition_prob = max(self.bigram_probs[idx_pre_tag, idx_tag], min_prob)
            else:
                idx_pre_tag1 = self.tag2idx[tags[i - 2]]
                idx_pre_tag2 = self.tag2idx[tags[i - 1]]
                transition_prob = max(self.trigram_probs[idx_pre_tag1, idx_pre_tag2, idx_tag], min_prob)

            sequence_log_prob += np.log(transition_prob) + np.log(emission_prob)
        return sequence_log_prob

    def inference(self, method, sequence):
        """Tags a sequence with part of speech tags."""
        if method == 'viterbi':
            return self.viterbi(sequence)
        elif method == 'beam':
            return self.beam_search(sequence, k=5)
        elif method == 'greedy':
            return self.greedy_decoding(sequence)
        else:
            raise ValueError("Unknown decoding method.")

    # The decoding methods (greedy_decoding, beam_search, viterbi) remain the same
    # except for the adjustment in the suffix matching loop direction

    def greedy_decoding(self, sequence):
        """Tags a sequence with part of speech tags using greedy decoding."""
        tag_pred = []
        N_tag = len(self.all_tags)
        min_prob = 1e-10  # Prevent log(0)
        for i, word in enumerate(sequence):
            idx_word = self.word2idx.get(word, -1)
            if idx_word == -1:
                # Handle unknown words
                suffix_probs = self.unknown_tag_probs
                for m in range(UNK_M, 0, -1):
                    if len(word) >= m:
                        suffix = word[-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break  # Use the longest matching suffix
                emission_probs = suffix_probs
            else:
                emission_probs = self.lexical_probs[:, idx_word]

            prob_cur = float('-inf')
            tag_cur = None
            for idx in range(N_tag):
                emission_prob = max(emission_probs[idx], min_prob)
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
        idx_word = self.word2idx.get(sequence[0], -1)
        if idx_word == -1:
            # Unknown word handling
            suffix_probs = self.unknown_tag_probs
            for m in range(UNK_M, 0, -1):
                if len(sequence[0]) >= m:
                    suffix = sequence[0][-m:]
                    if suffix in self.suffix_tag_probs:
                        suffix_probs = self.suffix_tag_probs[suffix]
                        break
            emission_probs = suffix_probs
        else:
            emission_probs = self.lexical_probs[:, idx_word]

        for i in range(N_tag):
            emission_prob = max(emission_probs[i], min_prob)
            transition_prob = max(self.unigram_probs[i], min_prob)
            total_log_prob = np.log(emission_prob) + np.log(transition_prob)
            path = [i]
            heapq.heappush(beam, (-total_log_prob, path))

        beam = heapq.nsmallest(k, beam)

        for t in range(1, N_word):
            candidates = []
            idx_word = self.word2idx.get(sequence[t], -1)
            if idx_word == -1:
                # Unknown word handling
                suffix_probs = self.unknown_tag_probs
                for m in range(UNK_M, 0, -1):
                    if len(sequence[t]) >= m:
                        suffix = sequence[t][-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break
                emission_probs = suffix_probs
            else:
                emission_probs = self.lexical_probs[:, idx_word]

            for neg_log_prob, path in beam:
                for j in range(N_tag):
                    emission_prob = max(emission_probs[j], min_prob)
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
            return [self.idx2tag[np.argmax(self.unigram_probs)]] * N_word

    def viterbi(self, sequence):
        N_word = len(sequence)
        N_tag = len(self.all_tags)
        min_prob = 1e-10  # Prevent log(0)

        # Initialize
        pi = np.full((N_word, N_tag, N_tag), float('-inf'))
        backpointer = np.zeros((N_word, N_tag, N_tag), dtype=int)

        # first word
        idx_word = self.word2idx.get(sequence[0], -1)
        if idx_word == -1:
            # Unknown word handling
            suffix_probs = self.unknown_tag_probs
            for m in range(UNK_M, 0, -1):
                if len(sequence[0]) >= m:
                    suffix = sequence[0][-m:]
                    if suffix in self.suffix_tag_probs:
                        suffix_probs = self.suffix_tag_probs[suffix]
                        break
            emission_probs = suffix_probs
        else:
            emission_probs = self.lexical_probs[:, idx_word]

        for u in range(N_tag):
            emission_prob = max(emission_probs[u], min_prob)
            pi[0, 0, u] = np.log(self.unigram_probs[u]) + np.log(emission_prob)
            backpointer[0, 0, u] = 0

        # second word
        if N_word > 1:
            idx_word = self.word2idx.get(sequence[1], -1)
            if idx_word == -1:
                # Unknown word handling
                suffix_probs = self.unknown_tag_probs
                for m in range(UNK_M, 0, -1):
                    if len(sequence[1]) >= m:
                        suffix = sequence[1][-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break
                emission_probs = suffix_probs
            else:
                emission_probs = self.lexical_probs[:, idx_word]

            for u in range(N_tag):
                for v in range(N_tag):
                    emission_prob = max(emission_probs[v], min_prob)
                    transition_prob = max(self.bigram_probs[u, v], min_prob)
                    pi[1, u, v] = pi[0, 0, u] + np.log(transition_prob) + np.log(emission_prob)
                    backpointer[1, u, v] = 0

        for t in range(2, N_word):
            idx_word = self.word2idx.get(sequence[t], -1)
            if idx_word == -1:
                # Unknown word handling
                suffix_probs = self.unknown_tag_probs
                for m in range(UNK_M, 0, -1):
                    if len(sequence[t]) >= m:
                        suffix = sequence[t][-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break
                emission_probs = suffix_probs
            else:
                emission_probs = self.lexical_probs[:, idx_word]

            for u in range(N_tag):
                for v in range(N_tag):
                    emission_prob = max(emission_probs[v], min_prob)
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

        max_prob = float('-inf')
        best_u, best_v = 0, 0
        for u in range(N_tag):
            for v in range(N_tag):
                if pi[N_word - 1, u, v] > max_prob:
                    max_prob = pi[N_word - 1, u, v]
                    best_u, best_v = u, v

        tags_idx = [0] * N_word
        tags_idx[N_word - 1] = best_v
        tags_idx[N_word - 2] = best_u
        for t in range(N_word - 3, -1, -1):
            tags_idx[t] = backpointer[t + 2, tags_idx[t + 1], tags_idx[t + 2]]

        tag_pred = [self.idx2tag[idx] for idx in tags_idx]
        return tag_pred





if __name__ == "__main__":
#######################################################################################
# debug closed
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
