import numpy as np
from tqdm import tqdm

from .base_analyzer import BaseAnalyzer
from openood.evaluators.metrics import auc_and_fpr_recall


class Bootstrapping(BaseAnalyzer):
    @staticmethod
    def compute_all_metrics(scores, true_labels):
        auroc, aupr_in, aupr_out, fpr = \
            auc_and_fpr_recall(scores, true_labels - 1, 0.95)
        return {
            'AUROC': auroc,
            'AUPR_IN': aupr_in,
            'AUPR_OUT': aupr_out,
            'FPR95': fpr
        }

    def analyze(self, true_labels, model1_scores, model2_scores):
        n_bootstraps = self.analyzer_config.n_bootstraps
        confidence_level = self.analyzer_config.confidence_level

        n_samples = len(true_labels)
        metrics1 = self.compute_all_metrics(model1_scores, true_labels)
        metrics2 = self.compute_all_metrics(model2_scores, true_labels)

        metrics1_bootstrapped = []
        metrics2_bootstrapped = []
        diffs_bootstrapped = []

        for _ in tqdm(range(n_bootstraps), desc='Bootstrapping'):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            true_labels_boot = true_labels[indices]
            model1_scores_boot = model1_scores[indices]
            model2_scores_boot = model2_scores[indices]

            metrics1_boot = \
                self.compute_all_metrics(model1_scores_boot, true_labels_boot)
            metrics2_boot = \
                self.compute_all_metrics(model2_scores_boot, true_labels_boot)

            metrics1_bootstrapped.append(metrics1_boot)
            metrics2_bootstrapped.append(metrics2_boot)
            diffs_bootstrapped.append({
                k: metrics1_boot[k] - metrics2_boot[k]
                for k in metrics1_boot.keys()
            })

        results = {}
        for metric_name in metrics1.keys():
            metric1_bootstrapped = np.array(
                [m[metric_name] for m in metrics1_bootstrapped])
            metric2_bootstrapped = np.array(
                [m[metric_name] for m in metrics2_bootstrapped])
            diff_bootstrapped = np.array(
                [d[metric_name] for d in diffs_bootstrapped])

            # Compute confidence intervals
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            ci_model1 = np.percentile(metric1_bootstrapped,
                                      [lower_percentile, upper_percentile])
            ci_model2 = np.percentile(metric2_bootstrapped,
                                      [lower_percentile, upper_percentile])
            ci_diff = np.percentile(diff_bootstrapped,
                                    [lower_percentile, upper_percentile])

            # Calculate p-value for the difference in 'metric'
            diff_metric_observed = \
                metrics1[metric_name] - metrics2[metric_name]
            diff_metric_bootstrapped = \
                metric1_bootstrapped - metric2_bootstrapped
            abs_diff_observed = np.abs(diff_metric_observed)
            # Center the bootstrap differences around the observed difference
            abs_diffs_centered = \
                np.abs(diff_metric_bootstrapped - diff_metric_observed)
            p_value = max(np.mean(abs_diffs_centered >= abs_diff_observed),
                          1 / n_bootstraps)

            results[metric_name] = {
                'Model1': metrics1[metric_name],
                'Model1 Confidence-Interval': ci_model1,
                'Model2': metrics2[metric_name],
                'Model2 Confidence-Interval': ci_model2,
                'Diff Observed': diff_metric_observed,
                'Diff Confidence-Interval': ci_diff,
                'P-Value': p_value
            }

        return results
