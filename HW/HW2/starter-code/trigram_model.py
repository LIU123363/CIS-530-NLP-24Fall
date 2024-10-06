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

    def get_unigrams(self):
        """
        Computes unigrams.
        Tip. Map each tag to an integer and store the unigrams in a numpy array.
        """
        N = len(self.all_tags)
        self.unigram_counts = np.zeros(N)
        for tags in self.data[1]:
            for tag in tags:
                self.unigram_counts[self.tag2idx[tag]] += 1
        total_counts = np.sum(self.unigram_counts)
        m, V = 1, N
        self.unigram_probs = (self.unigram_counts + m) / (total_counts + m * V)

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

        # Add-k smoothing
        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.bigram_probs = (self.bigram_counts + k) / (self.bigram_counts.sum(axis=1, keepdims=True) + k * N)

        # Linear Interpolation
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.95
                lambda_2 = 0.05
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

        # Add-k smoothing
        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.trigram_probs = (self.trigram_counts + k) / (self.trigram_counts.sum(axis=2, keepdims=True) + k * N)

        # Linear Interpolation
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.9
                lambda_2 = 0.07
                lambda_3 = 0.03
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
                    # Low-frequency words
                    for m in range(1, UNK_M + 1):
                        if len(word) >= m:
                            suffix = word[-m:]
                            suffix_tag_counts[suffix][idx_tag] += 1

        # compute the emission prob
        if SMOOTHING == LAPLACE:
            m, V = 1, N_word
            self.lexical_probs = (self.lexical_counts + m) / (self.lexical_counts.sum(axis=1, keepdims=True) + m * V)
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 1
                lambda_2 = 0
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

        # convert suffix_tag_counts to prob
        self.suffix_tag_probs = {}
        for suffix, counts in suffix_tag_counts.items():
            total_counts = counts.sum()
            if total_counts > 0:
                self.suffix_tag_probs[suffix] = counts / total_counts
            else:
                self.suffix_tag_probs[suffix] = np.ones(N_tag) / N_tag

        # 对于未知词，默认使用均匀分布
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
            idx_word = self.word2idx.get(word, -1)
            if idx_word == -1:
                # unknown word
                suffix_probs = self.unknown_tag_probs
                for m in range(1, UNK_M + 1):
                    if len(word) >= m:
                        suffix = word[-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break  # use the longest suffix
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
            return self.beam_search(sequence, k=10)
        elif method == 'greedy':
            return self.greedy_decoding(sequence)
        else:
            raise ValueError("Unknown decoding method.")

    def greedy_decoding(self, sequence):
        """Tags a sequence with part of speech tags using greedy decoding."""
        tag_pred = []
        N_tag = len(self.all_tags)
        min_prob = 1e-10  # 防止取对数时出现 log(0)
        for i, word in enumerate(sequence):
            idx_word = self.word2idx.get(word, -1)
            if idx_word == -1:
                # 处理未知词
                suffix_probs = self.unknown_tag_probs
                for m in range(1, UNK_M + 1):
                    if len(word) >= m:
                        suffix = word[-m:]
                        if suffix in self.suffix_tag_probs:
                            suffix_probs = self.suffix_tag_probs[suffix]
                            break  # 使用最长匹配的后缀
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
        min_prob = 1e-10  # 防止取对数时出现 log(0)

        # Initialize beam
        beam = []
        idx_word = self.word2idx.get(sequence[0], -1)
        if idx_word == -1:
            # Unknown word handling
            suffix_probs = self.unknown_tag_probs
            for m in range(1, UNK_M + 1):
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
                for m in range(1, UNK_M + 1):
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
        min_prob = 1e-10  # 防止取对数时出现 log(0)

        # Initialize
        pi = np.full((N_word, N_tag, N_tag), float('-inf'))
        backpointer = np.zeros((N_word, N_tag, N_tag), dtype=int)

        # first word
        idx_word = self.word2idx.get(sequence[0], -1)
        if idx_word == -1:
            # Unknown word handling
            suffix_probs = self.unknown_tag_probs
            for m in range(1, UNK_M + 1):
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
                for m in range(1, UNK_M + 1):
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
                for m in range(1, UNK_M + 1):
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
    pos_tagger = POSTagger()
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_x = load_data("data/test_x.csv")

    emission_threshold = 2
    pos_tagger.train(train_data, emission_threshold)

    # evaluate model by dev_data
    method = 'beam'
    evaluate(dev_data, pos_tagger, method)

    # test prediction
    test_y = []
    for sentence in test_x:
        pred_tags = pos_tagger.inference(method, sentence)  # 可以选择 'viterbi'、'beam'、'greedy'
        test_y.append(pred_tags)

    flat_predictions = [tag for sent_tags in test_y for tag in sent_tags]
    ids = list(range(len(flat_predictions)))

    # write the result to the 'test_y.csv' file
    df = pd.DataFrame({'id': ids, 'tag': flat_predictions})
    df.to_csv('test_y.csv', index=False)
    print("Test predictions saved to test_y.csv.")
