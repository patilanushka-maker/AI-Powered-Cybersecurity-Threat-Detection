"""
Evaluation Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Computing security-critical evaluation metrics on held-out test data:
  Accuracy, Precision, Recall (Detection Rate), F1-Score, False Positive Rate (FPR),
  ROC-AUC, and Confusion Matrix
- Strict positive-class semantics and probability index resolution
- Structured JSON-serializable report generation and optional plot export
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def compute_binary_metrics(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_prob: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics with security-specific metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = Benign, 1 = Attack).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class (Attack).
    pos_label : int, default 1
        Positive class identifier.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, precision, recall, f1_score, false_positive_rate,
        confusion_matrix breakdown, and optional roc_auc / pr_auc.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items."
        )

    # Validate binary label domain
    unique_true = set(np.unique(y_t))
    if not unique_true.issubset({0, 1}):
        raise ValueError(
            f"y_true contains non-binary labels: {unique_true - {0, 1}}. Expected subset of {{0, 1}}."
        )

    # Basic metrics with explicit zero_division=0.0
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0.0))
    rec = float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0.0))
    f1 = float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0.0))

    # Confusion Matrix: label order [0, 1] for binary
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "pos_label": int(pos_label),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "sample_count": int(len(y_t)),
    }

    # Area Under ROC / PR curves if probabilities are supplied
    if y_prob is not None:
        y_pr = np.asarray(y_prob)
        # ROC-AUC requires both classes present in y_true
        if len(np.unique(y_t)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_t, y_pr))
                metrics["pr_auc"] = float(average_precision_score(y_t, y_pr, pos_label=pos_label))
            except Exception:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics["roc_auc_applicable"] = bool(y_prob is not None and len(np.unique(y_t)) == 2)
    return metrics


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pos_label: int = 1,
    save_confusion_matrix_plot: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model or pipeline against held-out test data.

    Parameters
    ----------
    model : Any
        Fitted Scikit-learn estimator or Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target labels.
    pos_label : int, default 1
        Identifier for the positive attack class.
    save_confusion_matrix_plot : Optional[Union[str, Path]], default None
        Optional path to save confusion matrix heatmap image.

    Returns
    -------
    Dict[str, Any]
        Structured evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Determine probability column for pos_label without assuming column index 1
    y_prob = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = getattr(model, "classes_", None)
        if classes is not None:
            if pos_label in classes:
                pos_idx = list(classes).index(pos_label)
                y_prob = probabilities[:, pos_idx]
            else:
                raise ValueError(
                    f"Positive class '{pos_label}' not found in estimator classes: {list(classes)}."
                )
        elif probabilities.shape[1] > 1:
            y_prob = probabilities[:, 1]

    metrics = compute_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        pos_label=pos_label,
    )

    # Optionally generate confusion matrix image
    if save_confusion_matrix_plot is not None:
        plot_path = Path(save_confusion_matrix_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        cm = np.array(metrics["confusion_matrix"]["matrix"])

        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="large")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign (0)", "Attack (1)"])
        ax.set_yticklabels(["Benign (0)", "Attack (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        metrics["confusion_matrix_plot_path"] = str(plot_path)

    return metrics


def save_evaluation_report(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Serialize evaluation metrics to a formatted JSON report.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of evaluated metrics.
    output_path : Union[str, Path]
        Target file path (.json).
    overwrite : bool, default False
        Whether to overwrite existing report file.

    Returns
    -------
    Path
        Path to saved report.

    Raises
    ------
    FileExistsError
        If destination exists and overwrite=False.
    """
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation report already exists at '{path}'. Pass overwrite=True to overwrite."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return path
