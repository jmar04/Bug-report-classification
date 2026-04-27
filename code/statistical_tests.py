import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# CONFIG
datasets = ["tensorflow", "pytorch", "keras", "incubator-mxnet", "caffe"]

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]

baseline_suffix = "_NB.csv"
improved_suffix = "_LR.csv"

def parse_metric_list(metric_string):
    metric_string = metric_string.strip("[]")
    return [float(x.strip()) for x in metric_string.split(",")]

# WILCOXON TESTS
all_p_values = []
test_labels = []
all_statistics = []

aggregate_baseline = {m: [] for m in metrics}
aggregate_improved = {m: [] for m in metrics}

print("per dataset wilcoxon signed-rank tests")


for dataset in datasets:

    baseline_file =  dataset + baseline_suffix
    improved_file =  dataset + improved_suffix

    baseline_df = pd.read_csv(baseline_file)
    improved_df = pd.read_csv(improved_file)

    for metric in metrics:

        baseline_col = f"CV_list({metric})"
        improved_col = f"CV_list({metric})"

        baseline_scores = parse_metric_list(
            baseline_df.iloc[-1][baseline_col]
        )

        improved_scores = parse_metric_list(
            improved_df.iloc[-1][improved_col]
        )

        # Store for aggregate testing
        aggregate_baseline[metric].extend(baseline_scores)
        aggregate_improved[metric].extend(improved_scores)

        # Wilcoxon test
        stat, p = wilcoxon(baseline_scores, improved_scores)

        print(
            f"{metric}: "
            f"baseline={np.mean(baseline_scores):.4f}, "
            f"improved={np.mean(improved_scores):.4f}, "
            f"diff={np.mean(improved_scores)-np.mean(baseline_scores):+.4f}, "
            f"statistic={stat:.4f}, "
            f"p={p:.6f}"
        )

        all_p_values.append(p)
        test_labels.append(f"{dataset}_{metric}")
        all_statistics.append(stat)

# HOLM-BONFERRONI CORRECTION (PER-DATASET)
reject, corrected_p, _, _ = multipletests(
    all_p_values,
    alpha=0.05,
    method="holm"
)

for label, raw_p, adj_p, sig in zip(
    test_labels,
    all_p_values,
    corrected_p,
    reject
):
    print(
        f"{label}: "
        f"raw p={raw_p:.6f}, "
        f"adjusted p={adj_p:.6f}, "
        f"significant={sig}"
    )

# AGGREGATE WILCOXON TESTS
aggregate_p_values = []
aggregate_labels = []
aggregate_statistics = []

for metric in metrics:

    stat, p = wilcoxon(
        aggregate_baseline[metric],
        aggregate_improved[metric]
    )

    print(
        f"{metric}: "
        f"baseline={np.mean(aggregate_baseline[metric]):.4f}, "
        f"improved={np.mean(aggregate_improved[metric]):.4f}, "
        f"diff={np.mean(aggregate_improved[metric])-np.mean(aggregate_baseline[metric]):+.4f}, "
        f"statistic={stat:.4f}, "
        f"p={p:.6f}"
    )

    aggregate_p_values.append(p)
    aggregate_labels.append(metric)
    aggregate_statistics.append(stat)

# HOLM-BONFERRONI CORRECTION (AGGREGATE)
reject_agg, corrected_p_agg, _, _ = multipletests(
    aggregate_p_values,
    alpha=0.05,
    method="holm"
)

for label, raw_p, adj_p, sig in zip(
    aggregate_labels,
    aggregate_p_values,
    corrected_p_agg,
    reject_agg
):
    print(
        f"{label}: "
        f"raw p={raw_p:.6f}, "
        f"adjusted p={adj_p:.6f}, "
        f"significant={sig}"
    )

# save results to csv

# per-dataset results
means_data = []

for dataset in datasets:
    baseline_df = pd.read_csv(dataset + baseline_suffix)
    improved_df = pd.read_csv(dataset + improved_suffix)

    for metric in metrics:
        b = parse_metric_list(baseline_df.iloc[-1][f"CV_list({metric})"])
        i = parse_metric_list(improved_df.iloc[-1][f"CV_list({metric})"])

        means_data.append({
            "Dataset": dataset,
            "Metric": metric,
            "Baseline_Mean": np.mean(b),
            "Improved_Mean": np.mean(i),
            "Difference": np.mean(i) - np.mean(b)
        })

means_df = pd.DataFrame(means_data)

per_dataset_results = []
for label, raw_p, adj_p, sig, stat in zip(test_labels, all_p_values, corrected_p, reject, all_statistics):
    parts = label.split("_")
    metric = parts[-1]
    dataset = "_".join(parts[:-1])
    per_dataset_results.append({
        "Dataset": dataset,
        "Metric": metric,
        "Statistic": stat,
        "Raw_p": raw_p,
        "Adjusted_p": adj_p,
        "Significant": sig,
    })

per_dataset_df = pd.DataFrame(per_dataset_results)
final_df = pd.merge(means_df, per_dataset_df, on=["Dataset", "Metric"])
final_df.to_csv("statistical_results_per_dataset.csv", index=False)

# aggregate results
aggregate_rows = []

for metric, raw_p, adj_p, sig, stat in zip(aggregate_labels, aggregate_p_values, corrected_p_agg, reject_agg, aggregate_statistics):
    aggregate_rows.append({
        "Metric": metric,
        "Baseline_Mean": np.mean(aggregate_baseline[metric]),
        "Improved_Mean": np.mean(aggregate_improved[metric]),
        "Difference": np.mean(aggregate_improved[metric]) - np.mean(aggregate_baseline[metric]),
        "Statistic": stat,
        "Raw_p": raw_p,
        "Adjusted_p": adj_p,
        "Significant": sig
    })

aggregate_df = pd.DataFrame(aggregate_rows)
aggregate_df.to_csv("statistical_results_aggregate.csv", index=False)

print("\nResults saved to:")
print("  statistical_results_per_dataset.csv")
print("  statistical_results_aggregate.csv")