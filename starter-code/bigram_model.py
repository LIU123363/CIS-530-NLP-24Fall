from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
from collections import Counter, defaultdict
import heapq
import copy

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
        """Initializes the tagger model parameters and anything else necessary. """
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

    def get_unigrams(self):
        """
        Computes unigrams.
        Tip. Map each tag to an integer and store the unigrams in a numpy array.
                unigrams[tag] = Prob(tag)
        """
        N = len(self.all_tags)  # number of tags
        self.unigram_counts = np.zeros(N)
        for tags in self.data[1]:
            for tag in tags:
                self.unigram_counts[self.tag2idx[tag]] += 1
        total_counts = np.sum(self.unigram_counts)
        # Laplace smoothing
        m, V = 1, N
        self.unigram_probs = (self.unigram_counts + m) / (total_counts + m * V)

    def get_bigrams(self):
        """
        Computes bigrams with smoothing.
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1).
        """
        N = len(self.all_tags)
        self.bigram_counts = np.zeros((N, N))
        for tags in self.data[1]:
            for i in range(len(tags) - 1):
                idx_tag1 = self.tag2idx[tags[i]]
                idx_tag2 = self.tag2idx[tags[i + 1]]
                self.bigram_counts[idx_tag1, idx_tag2] += 1

        # 使用拉普拉斯平滑（Add-k smoothing）
        if SMOOTHING == LAPLACE:
            k = LAPLACE_FACTOR
            self.bigram_probs = (self.bigram_counts + k) / (self.bigram_counts.sum(axis=1, keepdims=True) + k * N)

        # 使用线性插值平滑（Linear Interpolation）
        elif SMOOTHING == INTERPOLATION:
            if LAMBDAS is None:
                lambda_1 = 0.6
                lambda_2 = 0.4
            else:
                lambda_1, lambda_2 = LAMBDAS

            # 使用线性插值结合 unigram 和 bigram 概率
            self.bigram_probs = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    bigram_count = self.bigram_counts[i, j]
                    unigram_count = self.unigram_counts[j]
                    total_bigrams = self.bigram_counts.sum(axis=1)[i]
                    total_unigrams = self.unigram_counts.sum()

                    prob_bigram = (bigram_count + EPSILON) / (total_bigrams + EPSILON)
                    prob_unigram = (unigram_count + EPSILON) / (total_unigrams + EPSILON)

                    # 线性插值
                    self.bigram_probs[i, j] = lambda_1 * prob_bigram + lambda_2 * prob_unigram

    def get_emissions(self, threshold=None):
        """
        Computes emission probabilities.
        Tip. Map each tag to an integer and each word in the vocabulary to an integer.
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag)
        """
        word_counts = defaultdict(int)
        for words in self.data[0]:
            for word in words:
                word_counts[word] += 1

        # Fill the word2idx - no threshold
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        idx = 1
        for word, count in word_counts.items():
            if count >= threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        N_tag = len(self.all_tags)
        N_word = len(self.word2idx)
        self.lexical_counts = np.zeros((N_tag, N_word))
        for words, tags in zip(self.data[0], self.data[1]):
            for word, tag in zip(words, tags):
                idx_tag = self.tag2idx[tag]
                idx_word = self.word2idx.get(word, 0)  # Map rare words to '<UNK>'
                self.lexical_counts[idx_tag, idx_word] += 1

        if SMOOTHING == LAPLACE:
            # Laplace smoothing
            m, V = 1, N_word
            self.lexical_probs = (self.lexical_counts + m) / (self.lexical_counts.sum(axis=1, keepdims=True) + m * V)
        elif SMOOTHING == INTERPOLATION:
            # Linear Interpolation smoothing for emissions
            if LAMBDAS is None:
                lambda_1 = 0.7
                lambda_2 = 0.3
            else:
                lambda_1, lambda_2 = LAMBDAS

            # Calculating word-level unigram probabilities
            total_words = sum(word_counts.values())
            word_unigram_probs = np.zeros(N_word)
            for word, idx in self.word2idx.items():
                word_unigram_probs[idx] = (word_counts[word] + EPSILON) / (total_words + EPSILON)

            # Linearly interpolate emission probabilities
            self.lexical_probs = np.zeros((N_tag, N_word))
            for i in range(N_tag):
                for j in range(N_word):
                    prob_emission = (self.lexical_counts[i, j] + EPSILON) / (
                                self.lexical_counts.sum(axis=1)[i] + EPSILON)
                    prob_word_unigram = word_unigram_probs[j]
                    self.lexical_probs[i, j] = lambda_1 * prob_emission + lambda_2 * prob_word_unigram


    def train(self, data, emission_threshold=None):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.

        """
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Compute probabilities
        self.get_unigrams()
        self.get_bigrams()
        self.get_emissions(emission_threshold)

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""
        sequence_log_prob = 0.0
        for i, word in enumerate(sequence):
            idx_tag = self.tag2idx[tags[i]]
            idx_word = self.word2idx.get(word, self.word2idx.get('<UNK>', 0))
            emission_prob = self.lexical_probs[idx_tag, idx_word]
            emission_log_prob = np.log(emission_prob) if emission_prob > 0 else float('-inf')

            if i == 0:
                transition_prob = self.unigram_probs[idx_tag]
                transition_log_prob = np.log(transition_prob) if transition_prob > 0 else float('-inf')
            else:
                idx_pre_tag = self.tag2idx[tags[i - 1]]
                if SMOOTHING == LAPLACE:
                    transition_prob = self.bigram_probs[idx_pre_tag, idx_tag]
                elif SMOOTHING == INTERPOLATION:
                    # Linear interpolation smoothing
                    if LAMBDAS is None:
                        lambda_1 = 0.6
                        lambda_2 = 0.4
                    else:
                        lambda_1, lambda_2 = LAMBDAS

                    prob_bigram = self.bigram_probs[idx_pre_tag, idx_tag]
                    prob_unigram = self.unigram_probs[idx_tag]

                    # Linearly interpolate transition probabilities
                    transition_prob = lambda_1 * prob_bigram + lambda_2 * prob_unigram

                transition_log_prob = np.log(transition_prob) if transition_prob > 0 else float('-inf')
            sequence_log_prob += transition_log_prob + emission_log_prob
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
        for i, word in enumerate(sequence):
            idx_word = self.word2idx.get(word, self.word2idx.get('<UNK>', 0))
            tag_cur = None
            prob_cur = float('-inf')
            if i == 0:
                for idx in range(N_tag):
                    emission_prob = self.lexical_probs[idx, idx_word]
                    unigram_prob = self.unigram_probs[idx]
                    total_log_prob = np.log(emission_prob) + np.log(unigram_prob) if emission_prob > 0 and unigram_prob > 0 else float('-inf')
                    if total_log_prob > prob_cur:
                        tag_cur = self.idx2tag[idx]
                        prob_cur = total_log_prob
            else:
                prev_tag_idx = self.tag2idx[tag_pred[-1]]
                for idx in range(N_tag):
                    emission_prob = self.lexical_probs[idx, idx_word]
                    transition_prob = self.bigram_probs[prev_tag_idx, idx]
                    total_log_prob = np.log(emission_prob) + np.log(transition_prob) if emission_prob > 0 and transition_prob > 0 else float('-inf')
                    if total_log_prob > prob_cur:
                        tag_cur = self.idx2tag[idx]
                        prob_cur = total_log_prob
            tag_pred.append(tag_cur)
        return tag_pred

    def beam_search(self, sequence, k):
        """Tags a sequence with part of speech tags using beam search."""
        N_word = len(sequence)
        N_tag = len(self.all_tags)

        beam_path = []
        word_idx = self.word2idx.get(sequence[0], self.word2idx.get('<UNK>', 0))
        for i in range(N_tag):
            emission_prob = self.lexical_probs[i, word_idx]
            unigram_prob = self.unigram_probs[i]
            if emission_prob > 0 and unigram_prob > 0:
                total_log_prob = np.log(unigram_prob) + np.log(emission_prob)
                path = [self.idx2tag[i]]
                heapq.heappush(beam_path, (-total_log_prob, i, path))
        beam_path = heapq.nsmallest(k, beam_path)

        for t in range(1, N_word):
            candidates = []
            word_idx = self.word2idx.get(sequence[t], self.word2idx.get('<UNK>', 0))
            for neg_log_prob, prev_tag_idx, path in beam_path:
                for j in range(N_tag):
                    emission_prob = self.lexical_probs[j, word_idx]
                    transition_prob = self.bigram_probs[prev_tag_idx, j]
                    if emission_prob > 0 and transition_prob > 0:
                        total_log_prob = -neg_log_prob + np.log(transition_prob) + np.log(emission_prob)
                        heapq.heappush(candidates, (-total_log_prob, j, path + [self.idx2tag[j]]))
            beam_path = heapq.nsmallest(k, candidates)

        if beam_path:
            _, _, tag_pred = min(beam_path)
            return tag_pred
        else:
            return [self.idx2tag[0]] * N_word

    def viterbi(self, sequence):
        """Tags a sequence with part of speech tags using Viterbi algorithm."""
        N_word = len(sequence)
        N_tag = len(self.all_tags)
        pi = np.full((N_word, N_tag), float('-inf'))
        backpointer = np.zeros((N_word, N_tag), dtype=int)

        word_idx = self.word2idx.get(sequence[0], self.word2idx.get('<UNK>', 0))
        for i in range(N_tag):
            if self.unigram_probs[i] > 0 and self.lexical_probs[i, word_idx] > 0:
                pi[0, i] = np.log(self.unigram_probs[i]) + np.log(self.lexical_probs[i, word_idx])
            backpointer[0, i] = 0

        for t in range(1, N_word):
            word_idx = self.word2idx.get(sequence[t], self.word2idx.get('<UNK>', 0))
            for j in range(N_tag):
                max_prob = float('-inf')
                best_prev_tag = 0
                for i in range(N_tag):
                    prob = pi[t - 1, i] + np.log(self.bigram_probs[i, j]) + np.log(self.lexical_probs[j, word_idx]) if self.bigram_probs[i, j] > 0 and self.lexical_probs[j, word_idx] > 0 else float('-inf')
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = i
                pi[t, j] = max_prob
                backpointer[t, j] = best_prev_tag

        best_last_tag = np.argmax(pi[N_word - 1, :])
        tag_pred_idx = [best_last_tag]

        for t in range(N_word - 1, 0, -1):
            best_last_tag = backpointer[t, best_last_tag]
            tag_pred_idx.append(best_last_tag)

        tag_pred_idx.reverse()
        return [self.idx2tag[idx] for idx in tag_pred_idx]



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
