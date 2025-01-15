import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from .base_analyzer import BaseAnalyzer
from openood.utils import Config


class DelongsTest(BaseAnalyzer):
    def __init__(self, config: Config, analyzer_config: Config):
        super().__init__(config, analyzer_config)
        method = self.analyzer_config.method
        if method == 'fast':
            self.delongs_test = FastDelongsTest()
        elif method == 'simple':
            self.delongs_test = SimpleDelongsTest()
        else:
            raise ValueError(f"Unknown DeLong's test method: {method}")

    def analyze(self, true_labels, model1_scores, model2_scores):
        model_names = self.config.analyzer.model_names
        return self.delongs_test.analyze(true_labels, model1_scores,
                                         model2_scores, model_names)


# based on: https://github.com/yandexdataschool/roc_comparison
class FastDelongsTest:

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    @staticmethod
    def compute_midrank(x):
        """Computes midranks.
        Args:
           x - a 1D numpy array
        Returns:
           array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float32)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float32)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    @staticmethod
    def fastDeLong(predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
           predictions_sorted_transposed:
              a 2D numpy.array[n_classifiers, n_examples]
              sorted such as the examples with label "1" are first
           label_1_count: number of examples with label "1"
        Returns:
           (AUC value, DeLong covariance)
        Reference:
         @article{sun2014fast,
           title={Fast Implementation of DeLong's Algorithm for
                  Comparing the Areas Under Correlated Receiver
                  Operating Characteristic Curves},
           author={Xu Sun and Weichao Xu},
           journal={IEEE Signal Processing Letters},
           volume={21},
           number={11},
           pages={1389--1393},
           year={2014},
           publisher={IEEE}
         }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float32)
        ty = np.empty([k, n], dtype=np.float32)
        tz = np.empty([k, m + n], dtype=np.float32)
        for r in range(k):
            tx[r, :] = FastDelongsTest.compute_midrank(positive_examples[r, :])
            ty[r, :] = FastDelongsTest.compute_midrank(negative_examples[r, :])
            tz[r, :] = FastDelongsTest.compute_midrank(
                predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    @staticmethod
    def calc_log10_p_value(aucs, sigma):
        """Computes log10 of p-value.

        Args:
           aucs: 1D array of AUCs
           sigma: AUC DeLong covariances
        Returns:
           log10(p-value)
        """
        dl = np.array([[1, -1]])
        z = np.diff(aucs) / np.sqrt(np.dot(np.dot(dl, sigma), dl.T))
        log10_p_value = np.log10(2) + \
            norm.logsf(np.abs(z), loc=0, scale=1) / np.log(10)
        return log10_p_value.item()

    @staticmethod
    def compute_ground_truth_statistics(ground_truth):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    def analyze(self, true_labels, model1_scores, model2_scores, model_names):
        # Compute AUC for both models
        auc_model1 = roc_auc_score(true_labels, model1_scores)
        auc_model2 = roc_auc_score(true_labels, model2_scores)

        # Perform DeLong's test
        order, label_1_count = \
            self.compute_ground_truth_statistics(true_labels)
        predictions_sorted_transposed = \
            np.vstack((model1_scores, model2_scores))[:, order]
        aucs, delong_cov = self.fastDeLong(predictions_sorted_transposed,
                                           label_1_count)
        log10_p_value = self.calc_log10_p_value(aucs, delong_cov)
        p_value = 10**log10_p_value

        return {
            f'{model_names[0]} AUROC': auc_model1,
            f'{model_names[1]} AUROC': auc_model2,
            'log10(P-Value)': log10_p_value,
            'P-Value': p_value
        }


class SimpleDelongsTest:
    @staticmethod
    def compute_auc(scores, labels):
        """Compute the Area Under the Curve (AUC) using the rank-based Mann-
        Whitney statistic."""
        ranks = rankdata(scores)  # Rank the scores
        pos_ranks = np.sum(
            ranks[labels == 1])  # Sum of ranks for positive class
        n_pos = np.sum(labels == 1)  # Number of positive samples
        n_neg = np.sum(labels == 0)  # Number of negative samples
        auc = (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return auc

    @staticmethod
    def compute_v_statistics(scores, labels):
        """Compute the V-statistics for DeLong's test.

        Parameters:
        - scores: Array of scores from a single model.
        - labels: Array of true binary labels (0 or 1).

        Returns:
        - v_pos: Contributions of positive observations to the AUC.
        - v_neg: Contributions of negative observations to the AUC.
        """
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        # Compute the V-statistics
        v_pos = np.array([
            np.mean(pos_score > neg_scores) +
            0.5 * np.mean(pos_score == neg_scores) for pos_score in pos_scores
        ])
        v_neg = np.array([
            np.mean(pos_scores > neg_score) +
            0.5 * np.mean(pos_scores == neg_score) for neg_score in neg_scores
        ])

        return v_pos, v_neg

    @staticmethod
    def delong_covariance(v1_pos, v1_neg, v2_pos, v2_neg):
        """Compute the 2x2 covariance matrix of the AUC estimates for two
        models."""
        # Number of positive and negative samples
        n_pos = len(v1_pos)
        n_neg = len(v1_neg)

        # Variances for the first model
        var_auc1 = (np.var(v1_pos, ddof=1) / n_pos) + \
                   (np.var(v1_neg, ddof=1) / n_neg)

        # Variances for the second model
        var_auc2 = (np.var(v2_pos, ddof=1) / n_pos) + \
                   (np.var(v2_neg, ddof=1) / n_neg)

        # Covariance between the two models
        cov_auc1_auc2 = (np.cov(v1_pos, v2_pos, ddof=1)[0, 1] / n_pos) + \
                        (np.cov(v1_neg, v2_neg, ddof=1)[0, 1] / n_neg)

        return np.array([[var_auc1, cov_auc1_auc2], [cov_auc1_auc2, var_auc2]])

    def analyze(self, labels, scores1, scores2, model_names):
        """Perform DeLong's test to compare the AUCs of two models."""
        # Compute AUCs using rank-based Mann-Whitney statistic
        auc1 = self.compute_auc(scores1, labels)
        auc2 = self.compute_auc(scores2, labels)

        # Compute V-statistics for both models
        v1_pos, v1_neg = self.compute_v_statistics(scores1, labels)
        v2_pos, v2_neg = self.compute_v_statistics(scores2, labels)

        # Compute covariance matrix
        cov_matrix = self.delong_covariance(v1_pos, v1_neg, v2_pos, v2_neg)
        var_auc1 = cov_matrix[0, 0]
        var_auc2 = cov_matrix[1, 1]
        cov_auc1_auc2 = cov_matrix[0, 1]

        # Compute variance of the difference
        var_diff = var_auc1 + var_auc2 - 2 * cov_auc1_auc2

        # Compute Z-statistic
        z = (auc1 - auc2) / np.sqrt(var_diff)

        # Compute p-value
        log10_p_value = np.log10(2) + \
            norm.logsf(np.abs(z), loc=0, scale=1) / np.log(10)
        p_value = 10**log10_p_value.item()
        # p_value = 2 * (1 - norm.cdf(np.abs(z)))

        return {
            f'{model_names[0]} AUROC': auc1,
            f'{model_names[1]} AUROC': auc2,
            'log10(P-Value)': log10_p_value.item(),
            'P-Value': p_value
        }
