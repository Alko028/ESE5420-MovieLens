"""Make paper figures/tables from saved evaluation only; no training or raw data.

Checks saved predictions against sklearn, all saved bootstrap intervals and
paired differences, and 20 independent user-resamples before writing outputs.
The reproduction note deliberately retains the prior test observation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
METRICS = ("precision", "recall", "AP")
LABELS = {"movie_rate": "Movie-rate floor", "lr": "LR", "knn": "kNN (511)",
          "gnn": "Original GNN", "lr_genre": "LR + genre", "knn_genre": "kNN + genre (127)",
          "wide64": "GNN wide64", "wide128": "GNN wide128", "deep2x32": "GNN deep2x32"}
ORDER = list(LABELS)
NOTE = ("Full fixed-protocol reproducibility run after an earlier test evaluation. "
        "The same test set was already observed; this is not an additional independent test. "
        "No tuning, feature changes, threshold reselection, or best-of-runs selection used test results.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbid_raw_data():
    root = (BASE / "outputs/preprocessed").resolve()
    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if root == path or root in path.parents:
                raise PermissionError("Report generation must read saved results, never preprocessed data")
    sys.addaudithook(guard)


def audit(saved, protocol_root):
    execution = read_json(saved / "execution.json")
    protocol = read_json(protocol_root / "protocol.json")
    digest = sha(protocol_root / "protocol.json")
    assert digest == (protocol_root / "protocol.sha256").read_text().strip()
    assert execution["status"] == "PASS" and execution["final_loader_calls"] == 1
    assert execution["thresholds_reselected"] is False and execution["refit"] is False
    assert execution["protocol_sha256"] == digest
    result = read_json(saved / "results.json")
    assert result["partition"] == "test"
    with np.load(saved / "predictions.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    assert list(data["model_names"]) == ORDER
    assert data["scores"].shape == (result["n_samples"], len(ORDER))
    assert len(np.unique(data["row_id"])) == result["n_samples"]
    assert len(np.unique(data["user_id"])) == result["n_users"]
    assert np.isfinite(data["scores"]).all() and np.isin(data["y"], [0, 1]).all()
    y = data["y"]
    users, inverse = np.unique(data["user_id"], return_inverse=True)
    spec = {model["id"]: model for model in protocol["models"]}
    samples = {}
    alpha = (1 - protocol["evaluation"]["confidence_level"]) / 2
    for i, name in enumerate(ORDER):
        row = result["results"][name]
        assert row["threshold"] == spec[name]["threshold"]
        scores = data["scores"][:, i]
        predictions = scores >= row["threshold"]
        point = [precision_score(y, predictions), recall_score(y, predictions), average_precision_score(y, scores)]
        np.testing.assert_allclose(point, [row["metrics"][m] for m in METRICS], rtol=0, atol=2e-12)
        with np.load(saved / f"{name}_bootstrap.npz", allow_pickle=False) as boot:
            np.testing.assert_array_equal(boot["user_ids"], users)
            np.testing.assert_array_equal(boot["metric_names"], METRICS)
            samples[name] = boot["samples"]
        assert samples[name].shape == (protocol["evaluation"]["bootstrap_replicates"], 3)
        for j, metric in enumerate(METRICS):
            v = samples[name][:, j]
            v = v[np.isfinite(v)]
            interval = row["intervals"][metric]
            np.testing.assert_allclose(np.quantile(v, [alpha, 1-alpha]), [interval["lower"], interval["upper"]], rtol=0, atol=2e-12)
        rng = np.random.default_rng(protocol["evaluation"]["bootstrap_seed"])
        for b in range(20):
            weights = np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))[inverse]
            point = [precision_score(y, predictions, sample_weight=weights),
                     recall_score(y, predictions, sample_weight=weights),
                     average_precision_score(y, scores, sample_weight=weights)]
            np.testing.assert_allclose(point, samples[name][b], rtol=0, atol=2e-12)
    pairs = read_json(saved / "paired_differences.json")
    for pair in protocol["comparisons"]:
        row = pairs[pair["id"]]
        assert row["left"] == pair["left"] and row["right"] == pair["right"]
        for j, metric in enumerate(METRICS):
            delta = samples[pair["left"]][:, j] - samples[pair["right"]][:, j]
            delta = delta[np.isfinite(delta)]
            stats = row["metrics"][metric]
            np.testing.assert_allclose(np.quantile(delta, [alpha, 1-alpha]), [stats["lower"], stats["upper"]], rtol=0, atol=2e-12)
            np.testing.assert_allclose(stats["estimate"], result["results"][pair["left"]]["metrics"][metric] - result["results"][pair["right"]]["metrics"][metric], rtol=0, atol=2e-12)
    prevalence = read_json(saved / "prevalence.json")
    assert prevalence["positive_count"] == int(y.sum()) and prevalence["eligible_count"] == len(y)
    np.testing.assert_allclose(prevalence["positive_share"], y.mean(), rtol=0, atol=1e-15)
    return result, pairs, prevalence, protocol


def save_figure(figure, root, name):
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(root / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def precision_forest(results, pairs, root):
    names = [name for name in ORDER if name != "lr"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
    for i, name in enumerate(names):
        d = pairs[f"{name}_minus_lr"]["metrics"]["precision"] if f"{name}_minus_lr" in pairs else None
        if d is None:
            # kNN+genre is compared with kNN, not LR, in the frozen matrix.
            continue
        x, lo, hi = (100*d[key] for key in ("estimate", "lower", "upper"))
        color = "#b33d34" if name == "gnn" else "#32689a"
        ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", capsize=3, color=color)
    available = [(i, name) for i, name in enumerate(names) if f"{name}_minus_lr" in pairs]
    ax.set_yticks([i for i, _ in available], [LABELS[name] for _, name in available])
    ax.invert_yaxis()
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Precision difference vs LR (percentage points)")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, root, "precision_difference_vs_lr")


def pr_figure(saved, results, prevalence, names, root, filename):
    fig, ax = plt.subplots(figsize=(5.5, 3.7), layout="constrained")
    colors = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        curve = np.genfromtxt(saved / f"{name}_pr_curve.csv", delimiter=",", names=True)
        color = colors(i)
        ax.plot(np.r_[0, curve["recall"]], np.r_[1, curve["precision"]], label=LABELS[name], color=color, linewidth=1.2)
        row = results[name]["metrics"]
        ax.plot(row["recall"], row["precision"], "o", color=color, markersize=4)
    ax.axhline(prevalence["positive_share"], color="0.45", linestyle=":", linewidth=1, label="Positive share")
    ax.axvline(.5, color="0.65", linestyle="--", linewidth=.8)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(.45, 1.01))
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=.15)
    save_figure(fig, root, filename)


def pair_figure(pairs, names, labels, root, filename):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), layout="constrained")
    for ax, metric in zip(axes, ("precision", "AP")):
        for i, pair in enumerate(names):
            d = pairs[pair]["metrics"][metric]
            scale = 100 if metric == "precision" else 1
            x, lo, hi = (scale*d[k] for k in ("estimate", "lower", "upper"))
            ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", capsize=3, color="#32689a")
        ax.set_yticks(range(len(labels)), labels if metric == "precision" else [])
        ax.invert_yaxis()
        ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
        ax.set_xlabel("Precision difference (pp)" if metric == "precision" else "AP difference")
        ax.grid(axis="x", alpha=.2)
    save_figure(fig, root, filename)


def metric_cell(row, metric):
    value = row["metrics"][metric]
    interval = row["intervals"][metric]
    scale, digits = (1, 4) if metric == "AP" else (100, 2)
    return f"{scale*value:.{digits}f} [{scale*interval['lower']:.{digits}f}, {scale*interval['upper']:.{digits}f}]"


def latex_table(results, names, path, caption, label):
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & Precision (\%) [95\% CI] & Recall (\%) [95\% CI] & AP [95\% CI] \\", r"\midrule"]
    for name in names:
        lines.append(LABELS[name] + " & " + " & ".join(metric_cell(results[name], m) for m in METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=BASE / "outputs/final_test_rerun_20261125")
    parser.add_argument("--protocol-dir", type=Path, default=BASE / "protocols/reproduction_20261125")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "report/final_results")
    args = parser.parse_args()
    forbid_raw_data()
    result, pairs, prevalence, protocol = audit(args.results_dir, args.protocol_dir)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    figures, tables = out / "figures", out / "tables"
    figures.mkdir()
    tables.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = result["results"]
    precision_forest(rows, pairs, figures)
    pr_figure(args.results_dir, rows, prevalence, ORDER[:6], figures, "pr_main_models")
    pr_figure(args.results_dir, rows, prevalence, ["lr", "gnn", "wide64", "wide128", "deep2x32"], figures, "pr_gnn_capacity")
    pair_figure(pairs, ["lr_genre_minus_lr", "knn_genre_minus_knn"], ["LR: genre - history", "kNN: genre - history"], figures, "genre_effects")
    pair_figure(pairs, [name + "_minus_gnn" for name in ("wide64", "wide128", "deep2x32")],
                ["wide64 - original", "wide128 - original", "deep2x32 - original"], figures, "gnn_capacity_effects")
    caption = ("Fixed-threshold MovieLens-100K test results (9,570 ratings, 943 users). "
               "Thresholds were selected on validation for recall at least 50\\% and never adjusted on test. "
               "Intervals are 95\\% paired-user percentile bootstrap intervals (2,000 resamples), conditional on fitted models and thresholds. "
               "This complete fixed-protocol rerun reproduces an already observed test set, rather than adding independent test evidence.")
    latex_table(rows, ORDER, tables / "test_results_all.tex", caption, "tab:test_all")
    latex_table(rows, ORDER[:6], tables / "test_results_main.tex", caption, "tab:test_main")
    latex_table(rows, ["lr", "gnn", "wide64", "wide128", "deep2x32"], tables / "test_results_capacity.tex", caption, "tab:test_capacity")
    with (tables / "test_results_all.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "metric", "estimate", "ci_lower", "ci_upper", "frozen_threshold", "partition"])
        for name in ORDER:
            for metric in METRICS:
                row = rows[name]
                interval = row["intervals"][metric]
                writer.writerow([name, metric, row["metrics"][metric], interval["lower"], interval["upper"], row["threshold"], "test_reproduction"])
    with (tables / "paired_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "role", "metric", "estimate", "ci_lower", "ci_upper"])
        for name, pair in pairs.items():
            if name == "partition":
                continue
            for metric in METRICS:
                d = pair["metrics"][metric]
                writer.writerow([name, pair["role"], metric, d["estimate"], d["lower"], d["upper"]])
    pair_lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Prespecified paired precision differences on the reproduced test. Units are percentage points. Intervals are unadjusted for multiplicity; only original GNN minus LR is primary.}",
                  r"\label{tab:paired}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrr}", r"\toprule", r"Comparison & Role & $\Delta$Precision & 95\% CI \\", r"\midrule"]
    for name, pair in pairs.items():
        if name == "partition":
            continue
        d = pair["metrics"]["precision"]
        pair_lines.append(f"{LABELS[pair['left']]} $-$ {LABELS[pair['right']]} & {pair['role']} & {100*d['estimate']:+.2f} & [{100*d['lower']:+.2f}, {100*d['upper']:+.2f}]" + r" \\")
    pair_lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""]
    (tables / "paired_differences.tex").write_text("\n".join(pair_lines), encoding="utf-8")
    aggregate = {"provenance": NOTE, "results": result, "prevalence": prevalence,
                 "source_sha256": {p.name: sha(p) for p in args.results_dir.glob("*.json")},
                 "protocol_sha256": sha(args.protocol_dir / "protocol.json")}
    (out / "paper_results.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    (out / "audit.json").write_text(json.dumps({"status": "PASS", "point_metrics_sklearn": True,
        "all_model_and_paired_interval_quantiles": True, "independent_bootstrap_replicates_per_model": 20,
        "raw_data_opened": False, "inference_or_training_run": False, "test_previously_observed": True}, indent=2) + "\n", encoding="utf-8")
    summary = ["# 论文统一结果 — 完整固定协议复跑", "", NOTE, "", "| Method | Precision % [95% CI] | Recall % [95% CI] | AP [95% CI] |", "|---|---:|---:|---:|"]
    summary += ["| " + LABELS[n] + " | " + " | ".join(metric_cell(rows[n], m) for m in METRICS) + " |" for n in ORDER]
    summary += ["", "Positive share: " + f"{prevalence['positive_share']:.6f}" + "; all-positive recall=1 and constant-score AP=positive share.", "",
                "主检验为原 GNN−LR 的 Precision 配对差值。其余为补充/genre/探索比较，区间未经多重比较校正。", "",
                "## 本次完成", "", "九模型保存预测的独立 sklearn 点指标、全部区间/配对差值分位数与各前20次用户bootstrap核验通过；生成五组PDF/PNG和统一CSV/LaTeX表。图表生成未读取预处理或重新推理。", "",
                "## 整体剩余", "", "把新结果、方法、Related Work、Limitations、复现链接和AI致谢整合到NeurIPS论文，编译检查并提交新版draft.pdf；后续视频。", "",
                "## 下一任务及标准", "", "授权后更新论文；数字与paper_results.json一致，引用完整、图注/单位/阈值和复跑身份清楚，编译PDF可读且满足课程格式。", ""]
    (out / "README.md").write_text("\n".join(summary), encoding="utf-8")
    captions = {"precision_difference_vs_lr": "Test precision differences versus LR at frozen validation thresholds. Error bars: paired-user 95% percentile bootstrap intervals; original GNN is the primary comparison (red).",
                "pr_main_models": "Test precision-recall curves for the six main/reference/genre methods. Dots show validation-chosen frozen operating thresholds; the vertical line is recall 0.5, not a test tuning rule. The horizontal line is positive prevalence.",
                "pr_gnn_capacity": "Test precision-recall curves for LR and the original/expanded GNNs. Dots use frozen validation thresholds. Capacity variants are exploratory; no multiplicity correction.",
                "genre_effects": "Within-model genre-minus-history effects on precision (percentage points) and AP. Both kNN representations use the same extended CV candidate grid; their chosen k values may differ.",
                "gnn_capacity_effects": "Exploratory GNN capacity-minus-original effects on precision (percentage points) and AP. Error bars are paired-user 95% intervals without multiplicity correction. The deep model changes depth, parameter count and tap allocation simultaneously."}
    (out / "captions.md").write_text("# Figure captions\n\n" + "\n\n".join(f"**{k}**: {v} {NOTE}" for k, v in captions.items()) + "\n", encoding="utf-8")
    usage = ["% Copy final_results into your Overleaf project; requires booktabs and graphicx.",
             r"\input{final_results/tables/test_results_main.tex}", r"\input{final_results/tables/paired_differences.tex}",
             r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{final_results/figures/precision_difference_vs_lr.pdf}",
             r"\caption{" + captions["precision_difference_vs_lr"].replace("%", r"\%") + "}", r"\label{fig:primary}", r"\end{figure}", ""]
    (out / "latex_usage.tex").write_text("\n".join(usage), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(out), "figure_pairs": 5, "models": len(ORDER), "pairs": len(protocol["comparisons"])}))


if __name__ == "__main__":
    main()
