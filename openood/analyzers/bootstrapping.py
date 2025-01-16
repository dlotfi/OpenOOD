import os
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from tqdm import tqdm

from .base_analyzer import BaseAnalyzer
from openood.evaluators.metrics import auc_and_fpr_recall
from openood.utils.vis_comm import save_fig_and_close


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
        model_names = self.config.analyzer.model_names

        n_samples = len(true_labels)
        metrics1 = self.compute_all_metrics(model1_scores, true_labels)
        metrics2 = self.compute_all_metrics(model2_scores, true_labels)

        metrics1_bootstrapped = []
        metrics2_bootstrapped = []

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

        results = {}
        for metric_name in metrics1.keys():
            metric_bootstrapped1 = np.array(
                [m[metric_name] for m in metrics1_bootstrapped])
            metric_bootstrapped2 = np.array(
                [m[metric_name] for m in metrics2_bootstrapped])
            # diff_bootstrapped = np.array([
            #     m1[metric_name] - m2[metric_name] for m1, m2 in
            #     zip(metrics1_bootstrapped, metrics2_bootstrapped)
            # ])

            # Compute confidence intervals
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            ci_model1 = np.percentile(metric_bootstrapped1,
                                      [lower_percentile, upper_percentile])
            ci_model2 = np.percentile(metric_bootstrapped2,
                                      [lower_percentile, upper_percentile])
            # ci_diff = np.percentile(diff_bootstrapped,
            #                         [lower_percentile, upper_percentile])

            # Calculate p-value for the difference in 'metric'
            diff_metric_observed = \
                metrics1[metric_name] - metrics2[metric_name]
            diff_metric_bootstrapped = \
                metric_bootstrapped1 - metric_bootstrapped2
            abs_diff_observed = np.abs(diff_metric_observed)
            # Center the bootstrap differences around the observed difference
            abs_diffs_centered = \
                np.abs(diff_metric_bootstrapped - diff_metric_observed)
            p_value = max(np.mean(abs_diffs_centered >= abs_diff_observed),
                          1 / n_bootstraps)

            results[metric_name] = {
                f'{model_names[0]}': metrics1[metric_name],
                f'{model_names[0]} Confidence-Interval': ci_model1.tolist(),
                f'{model_names[1]}': metrics2[metric_name],
                f'{model_names[1]} Confidence-Interval': ci_model2.tolist(),
                # 'Diff Observed': diff_metric_observed,
                # 'Diff Confidence-Interval': ci_diff.tolist(),
                'P-Value': p_value,
                'Bootstrapped_Model1': metric_bootstrapped1,
                'Bootstrapped_Model2': metric_bootstrapped2,
            }

        return results

    def generate_output(self, results):
        super().generate_output(results)
        if self.analyzer_config.box_plot:
            self._draw_box_plot(results, self.config.output_dir)

    @staticmethod
    def _get_labels_n_weights(results):
        split_labels = {'nearood': 'Near OOD', 'farood': 'Far OOD'}
        labels = []
        label_weights = []
        for split_name, dataset_name, _ in results:
            if dataset_name is not None:
                label, weight = dataset_name, 'regular'
            elif split_name is None and dataset_name is None:
                label, weight = 'Overall', 'bold'
            else:
                label = split_labels[split_name] \
                    if split_name in split_labels else \
                    ' '.join([w.capitalize() for w in split_name.split('_')])
                weight = 'bold'
            labels.append(label)
            label_weights.append(weight)
        return labels, label_weights

    def _draw_box_plot(self, results, output_dir):
        metrics = ['AUROC', 'AUPR_IN', 'AUPR_OUT', 'FPR95']
        model_names = self.config.analyzer.model_names
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        labels, label_weights = self._get_labels_n_weights(results)
        for metric in metrics:
            output_path = os.path.join(output_dir, f'box_plot_{metric}.svg')
            model1_data = []
            model2_data = []
            for split, dataset, result in results:
                model1_data.append(result[metric]['Bootstrapped_Model1'])
                model2_data.append(result[metric]['Bootstrapped_Model2'])

            fig, ax = plt.subplots(figsize=(ceil(len(labels) / 2), 5), dpi=300)
            ax.set_title(f'Comparison of {metric} between {model_names[0]} '
                         f'and {model_names[1]}')
            ax.set_ylabel(metric)
            ax.grid(axis='x')
            ax.boxplot(model1_data,
                       positions=np.arange(len(labels)) * 2.0 - 0.2,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[0], alpha=0.6),
                       flierprops=dict(markerfacecolor='none',
                                       markeredgecolor=colors[0],
                                       marker='o',
                                       alpha=0.6),
                       whiskerprops=dict(color=colors[0]),
                       medianprops=dict(color='black'))
            ax.boxplot(model2_data,
                       positions=np.arange(len(labels)) * 2.0 + 0.2,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[1], alpha=0.6),
                       flierprops=dict(markerfacecolor='none',
                                       markeredgecolor=colors[1],
                                       marker='o',
                                       alpha=0.6),
                       whiskerprops=dict(color=colors[1]),
                       medianprops=dict(color='black'))
            ax.set_xticks(np.arange(len(labels)) * 2.0)
            ax.set_xticklabels(labels, rotation=90)
            for label, weight in zip(ax.get_xticklabels(), label_weights):
                label.set_fontproperties(FontProperties(weight=weight))
            lgd_handles = [
                Patch(facecolor=colors[0],
                      edgecolor='black',
                      label=model_names[0],
                      alpha=0.6),
                Patch(facecolor=colors[1],
                      edgecolor='black',
                      label=model_names[1],
                      alpha=0.6)
            ]
            ax.legend(handles=lgd_handles, loc='lower left', fontsize='small')
            save_fig_and_close(output_path)
