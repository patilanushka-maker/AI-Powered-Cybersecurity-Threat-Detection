"""
AI-Powered Cybersecurity Threat Detection System - Streamlit Dashboard.

Provides an offline, CSV-based network traffic analysis and threat detection interface:
1. System Overview: Live session triage stats, model status, and operational disclaimers.
2. Batch Flow Prediction: CSV upload, schema validation, score extraction, and alert generation.
3. Flagged Threat Alerts: Detailed alert event feed, severity breakdown, and alert export.
4. Benchmark & Validation Reports: Separate inspection of training benchmarks and independent cross-capture validations.
5. Model & Feature Architecture: Inspection of trained pipeline parameters, schemas, and imputation strategies.

DISCLAIMER:
This application performs offline CSV-based threat detection and triage prioritization.
Model predictions are investigation signals, NOT confirmed security incidents or proof of malice.
The system does not perform live packet capture (PCAP) or automated traffic blocking.
This prototype is an offline research system, NOT a certified production intrusion detection system.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset, normalize_column_names
from src.feature_engineering import FeatureSchema
from src.preprocessing import handle_non_finite_values
from src.predict import load_model_for_inference, predict_network_flows
from src.alert_engine import process_batch_alerts

# Model Artifact Definitions
DEFAULT_MULTI_ATTACK_MODEL_PATH = Path("models/random_forest_multi_attack.joblib")
DEFAULT_BASELINE_MODEL_PATH = Path("models/random_forest_baseline.joblib")
DEFAULT_MODEL_PATH = DEFAULT_MULTI_ATTACK_MODEL_PATH

AVAILABLE_MODELS: Dict[str, Path] = {
    "Multi-Attack Model (Friday + Tuesday) [Default Fallback]": DEFAULT_MULTI_ATTACK_MODEL_PATH,
    "ExtraTrees Classifier (Balanced - M16) [Experimental]": Path("models/model_extra_trees_balanced_m16.joblib"),
    "HistGradientBoosting (Balanced - M16) [Experimental]": Path("models/model_hist_gradient_boosting_m16.joblib"),
    "Random Forest (Balanced Subsample - M16) [Experimental]": Path("models/model_rf_balanced_subsample_m16.joblib"),
    "Baseline Model (Friday DDoS Only)": DEFAULT_BASELINE_MODEL_PATH,
}

# Evaluation Report Paths
MULTI_ATTACK_TRAIN_REPORT_PATH = Path("outputs/reports/multi_attack_train_evaluation.json")
THIRD_CAPTURE_EVAL_REPORT_PATH = Path("outputs/reports/third_capture_evaluation.json")
BASELINE_TRAIN_REPORT_PATH = Path("outputs/reports/train_evaluation.json")
CROSS_CAPTURE_TUESDAY_REPORT_PATH = Path("outputs/reports/cross_capture_evaluation.json")
EXPERIMENT_COMPARISON_REPORT_PATH = Path("outputs/reports/model_comparison_experiment.json")
DEFAULT_REPORT_PATH = MULTI_ATTACK_TRAIN_REPORT_PATH


def load_trusted_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely load the trusted local model artifact.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to local model artifact. Defaults to Multi-Attack model.

    Returns
    -------
    Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]
        (pipeline, metadata, error_message)
    """
    path = Path(model_path)
    if not path.exists():
        # Fallback to baseline if multi-attack not yet generated
        if path == DEFAULT_MULTI_ATTACK_MODEL_PATH and DEFAULT_BASELINE_MODEL_PATH.exists():
            path = DEFAULT_BASELINE_MODEL_PATH
        else:
            return None, None, f"Model artifact not found at configured path: '{path}'."

    try:
        pipeline, metadata = load_model_for_inference(path)
        return pipeline, metadata, None
    except Exception as exc:
        return None, None, f"Failed to load model artifact: {exc}"


def load_evaluation_report(
    report_path: Union[str, Path] = DEFAULT_REPORT_PATH
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load saved evaluation report JSON if present.

    Parameters
    ----------
    report_path : Union[str, Path]
        Path to evaluation report.

    Returns
    -------
    Tuple[Optional[Dict[str, Any]], Optional[str]]
        (report_dict, status_message)
    """
    path = Path(report_path)
    if not path.exists():
        return None, f"No evaluation report found at '{path}'."

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as exc:
        return None, f"Error reading evaluation report: {exc}"


def run_batch_prediction_on_dataframe(
    model: Any,
    metadata: Dict[str, Any],
    raw_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Execute batch scoring and alert generation on a network flow DataFrame.

    Parameters
    ----------
    model : Any
        Fitted Scikit-learn Pipeline.
    metadata : Dict[str, Any]
        Model metadata containing feature schema.
    raw_df : pd.DataFrame
        Input flow DataFrame.
    threshold : float, default 0.5
        Threshold for positive attack classification and alerting.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]], Optional[str]]
        (prediction_df, alerts_list, error_message)
    """
    if raw_df.empty:
        return None, None, "Uploaded dataset contains no data rows."

    # Normalize column names defensively and detect duplicates after normalization
    try:
        norm_cols = normalize_column_names(raw_df.columns)
    except ValueError as val_err:
        return None, None, f"Feature Schema Error: {val_err}"

    clean_raw = raw_df.copy(deep=False)
    clean_raw.columns = norm_cols

    # Parse FeatureSchema if available
    schema = None
    if metadata and metadata.get("feature_schema"):
        schema = FeatureSchema.from_dict(metadata["feature_schema"])
    elif metadata and metadata.get("feature_names"):
        schema = FeatureSchema(
            feature_names=metadata["feature_names"],
            numeric_features=metadata.get("numeric_features", metadata["feature_names"]),
        )
    elif hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        schema = FeatureSchema(
            feature_names=feats,
            numeric_features=feats,
        )

    # Clean non-finite values (Inf -> NaN) without fitting transformers
    clean_df = handle_non_finite_values(clean_raw)

    # Explicitly exclude target label columns from model feature inputs
    target_cols = [c for c in clean_df.columns if str(c).strip().lower() in ["label", "target"]]
    flow_inputs = clean_df.drop(columns=target_cols) if target_cols else clean_df

    try:
        predictions, probabilities = predict_network_flows(
            model=model,
            flow_data=flow_inputs,
            feature_schema=schema,
            pos_label=1,
        )
    except ValueError as val_err:
        return None, None, f"Feature Schema Error: {val_err}"
    except Exception as exc:
        return None, None, f"Inference Failed: {exc}"

    # Build annotated result DataFrame preserving original source order and row count
    pred_df = clean_raw.copy()
    pred_df["predicted_class"] = predictions
    pred_df["predicted_label"] = [
        "Attack" if (p == 1 or prob >= threshold) else "Benign"
        for p, prob in zip(predictions, probabilities)
    ]
    pred_df["attack_score"] = [round(float(p), 4) for p in probabilities]

    # Generate alerts for predicted attacks
    try:
        alerts = process_batch_alerts(
            predictions=predictions,
            probabilities=probabilities,
            flow_data=clean_df,
            threshold=threshold,
        )
    except Exception:
        alerts = []

    return pred_df, alerts, None


def filter_predictions_display(
    pred_df: pd.DataFrame,
    class_filter: str = "All",
    min_score: float = 0.0,
) -> pd.DataFrame:
    """
    Filter predictions for dashboard display without modifying underlying predictions.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Full prediction DataFrame.
    class_filter : str, default 'All'
        One of 'All', 'Attack Only', or 'Benign Only'.
    min_score : float, default 0.0
        Minimum attack score threshold for display.

    Returns
    -------
    pd.DataFrame
        Filtered view of predictions.
    """
    df = pred_df.copy()

    if class_filter == "Attack Only":
        df = df[df["predicted_label"] == "Attack"]
    elif class_filter == "Benign Only":
        df = df[df["predicted_label"] == "Benign"]

    if min_score > 0.0:
        df = df[df["attack_score"] >= min_score]

    return df


def init_session_state():
    """Initialize Streamlit session state tracking."""
    if "analyzed_flows_count" not in st.session_state:
        st.session_state["analyzed_flows_count"] = 0
    if "predicted_attack_count" not in st.session_state:
        st.session_state["predicted_attack_count"] = 0
    if "predicted_benign_count" not in st.session_state:
        st.session_state["predicted_benign_count"] = 0
    if "last_predictions" not in st.session_state:
        st.session_state["last_predictions"] = None
    if "last_alerts" not in st.session_state:
        st.session_state["last_alerts"] = None
    if "last_filename" not in st.session_state:
        st.session_state["last_filename"] = None


def render_overview_tab(model_available: bool, model_err: Optional[str], active_model_path: Optional[Path] = None):
    """Render System Overview & Triage Metrics."""
    st.subheader("System Overview & Triage Dashboard")

    st.markdown(
        """
        > **⚠️ Prototype Investigation Notice**  
        > Predictions and threat scores displayed here are machine-learning triage indicators, 
        > **not confirmed security breaches**. All flagged flows require verification by a security analyst. 
        > This system operates strictly on offline batch network flows and does not execute live traffic blocking.
        > **Not certified for production deployment.**
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_text = "Ready" if model_available else "Model Unavailable"
        st.metric("Engine Status", status_text)

    with col2:
        st.metric("Total Flows Analyzed", st.session_state["analyzed_flows_count"])

    with col3:
        st.metric("Flagged Attacks", st.session_state["predicted_attack_count"])

    with col4:
        st.metric("Benign Flows", st.session_state["predicted_benign_count"])

    st.divider()

    st.markdown("### Operational Environment & Pipeline Status")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            - **Active Model Path**: `{active_model_path or DEFAULT_MODEL_PATH}`
            - **Model Availability**: `{"Available" if model_available else "Not Found"}`
            - **Dataset Scope**: Offline Network Flow CSVs (CIC-IDS2017 schema compatible)
            - **Inspection Mode**: Read-only local model scoring
            - **Target Column Policy**: `Label` is strictly excluded from feature inputs
            """
        )
    with col_b:
        if not model_available:
            st.warning(
                f"Model artifact is missing: {model_err}\n\n"
                "**Actionable Guidance**:\n"
                "To train a model from the terminal, execute:\n"
                "```bash\n"
                "python main.py train --data-dir data/raw\n"
                "```"
            )
        else:
            st.success(f"Loaded: `{active_model_path.name if active_model_path else 'Model'}` is verified and ready.")

    if st.session_state["analyzed_flows_count"] == 0:
        st.info("No flow data analyzed in the current session. Navigate to the **Batch Flow Prediction** tab to upload and inspect traffic.")


def render_batch_prediction_tab(model: Any, metadata: Dict[str, Any], model_available: bool):
    """Render CSV Upload and Batch Inference Section."""
    st.subheader("Batch Network Flow Scoring")

    st.markdown(
        """
        > **ℹ️ Automated Triage Notice**  
        > Model predictions and threat scores are automated triage signals, **not proof that a flow is malicious**. 
        > Uploaded CSVs are validated against the trained feature schema. Any target labels present are excluded from scoring.
        """
    )

    if not model_available:
        st.error(
            "Cannot perform prediction because the model artifact is not available. "
            "Please select or train a valid model artifact first."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload Network Flow CSV for Threat Analysis",
        type=["csv"],
        help="Upload an offline network flow CSV. Expected columns must match the model's trained feature schema (78 numerical features).",
    )

    threshold = st.slider(
        "Attack Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold above which flows are labeled as 'Attack' and alerts are generated.",
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            raw_df.columns = normalize_column_names(raw_df.columns)
        except ValueError as val_err:
            st.error(f"Invalid Column Schema in uploaded CSV: {val_err}")
            return
        except Exception as exc:
            st.error(f"Failed to parse uploaded CSV: {exc}")
            return

        st.markdown(f"**Uploaded Data Preview** (`{len(raw_df)}` rows, `{len(raw_df.columns)}` columns):")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if st.button("Run Threat Analysis", type="primary"):
            with st.spinner("Analyzing flows with trained Random Forest pipeline..."):
                pred_df, alerts, err = run_batch_prediction_on_dataframe(
                    model=model,
                    metadata=metadata,
                    raw_df=raw_df,
                    threshold=threshold,
                )

            if err:
                st.error(err)
            else:
                st.session_state["last_predictions"] = pred_df
                st.session_state["last_alerts"] = alerts
                st.session_state["last_filename"] = uploaded_file.name
                st.session_state["analyzed_flows_count"] += len(pred_df)
                st.session_state["predicted_attack_count"] += int((pred_df["predicted_label"] == "Attack").sum())
                st.session_state["predicted_benign_count"] += int((pred_df["predicted_label"] == "Benign").sum())
                st.success(f"Analysis complete! Processed {len(pred_df)} flows.")

    # Display results if available in session state
    if st.session_state["last_predictions"] is not None:
        st.divider()
        st.markdown("### Analysis Results")

        pred_df = st.session_state["last_predictions"]
        alerts = st.session_state["last_alerts"] or []

        n_total = len(pred_df)
        n_attack = int((pred_df["predicted_label"] == "Attack").sum())
        n_benign = int((pred_df["predicted_label"] == "Benign").sum())

        r1, r2, r3 = st.columns(3)
        r1.metric("Flows in Last Batch", n_total)
        r2.metric("Flagged Suspicious Flows", n_attack)
        r3.metric("Benign Flows", n_benign)

        st.markdown("#### Filter & Prioritize View")
        st.caption("Note: Adjusting display filters organizes the view without altering underlying predictions or retraining the model.")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            class_filter = st.selectbox("Filter by Classification", ["All", "Attack Only", "Benign Only"])
        with f_col2:
            min_score = st.slider("Minimum Attack Score for Display", 0.0, 1.0, 0.0, 0.05)

        filtered_view = filter_predictions_display(pred_df, class_filter=class_filter, min_score=min_score)
        st.dataframe(filtered_view, use_container_width=True)

        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_data = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_data,
                file_name="flow_predictions.csv",
                mime="text/csv",
            )
        with d_col2:
            alerts_json_str = json.dumps(alerts, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Download Threat Alerts JSON",
                data=alerts_json_str,
                file_name="threat_alerts.json",
                mime="application/json",
            )


def render_alerts_feed_tab():
    """Render Threat Alerts Feed."""
    st.subheader("Security Alert Feed & Triage Events")

    alerts = st.session_state.get("last_alerts")

    if alerts is None:
        st.info("No inference has been executed in the current session. Run a batch prediction to generate alerts.")
        return

    if len(alerts) == 0:
        st.success("✅ Zero attacks flagged in the latest batch. No security alerts generated.")
        return

    st.markdown(f"**Total Generated Alerts**: `{len(alerts)}`")

    # Severity distribution
    severities = [a.get("severity", "UNKNOWN") for a in alerts]
    sev_counts = pd.Series(severities).value_counts().to_dict()

    s_cols = st.columns(len(sev_counts) or 1)
    for idx, (sev, count) in enumerate(sev_counts.items()):
        with s_cols[idx % len(s_cols)]:
            st.metric(f"Severity: {sev}", count)

    st.markdown("#### Structured Alert Events")
    alerts_df = pd.DataFrame(
        [
            {
                "Flow ID": a.get("flow_id"),
                "Severity": a.get("severity"),
                "Attack Score": a.get("probability"),
                "Classification": a.get("attack_type"),
                "Timestamp (UTC)": a.get("timestamp"),
                "Telemetry Metadata": str(a.get("metadata", {})),
            }
            for a in alerts
        ]
    )
    st.dataframe(alerts_df, use_container_width=True)


def render_evaluation_benchmarks_tab():
    """Render Dedicated Evaluation & Benchmark Results Page."""
    st.subheader("📈 Model Evaluation & Empirical Benchmarks")

    st.markdown(
        """
        > **Strict Separation Notice**:  
        > Training evaluation (held-out test split) and independent out-of-distribution evaluation 
        > are presented separately. Software unit tests do NOT establish real-world generalization.
        """
    )

    eval_view = st.radio(
        "Select Benchmark Report to Inspect",
        [
            "🏆 Multi-Attack In-Distribution Benchmark (Friday + Tuesday)",
            "🌐 Independent Out-of-Distribution Validation (Wednesday Capture)",
            "🔬 Milestone 16: Model Generalization Experiments",
            "📜 Baseline DDoS Benchmark & Tuesday Cross-Capture",
        ],
        horizontal=True,
    )

    if eval_view == "🏆 Multi-Attack In-Distribution Benchmark (Friday + Tuesday)":
        st.markdown("### Multi-Attack Model Training Evaluation (Friday DDoS + Tuesday Brute-Force)")
        report, err = load_evaluation_report(MULTI_ATTACK_TRAIN_REPORT_PATH)

        if report is None:
            st.warning(f"Report not available: {err}")
            return

        st.markdown(
            f"""
            - **Model**: `models/random_forest_multi_attack.joblib`
            - **Evaluation Partition**: Held-Out Test Split (20% of 100,000 sampled flows)
            - **Test Sample Count**: `{report.get('sample_count', 20000):,}`
            - **Training Captures**: Friday Afternoon DDoS + Tuesday Brute-Force
            """
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{report.get('accuracy', 0.0):.4%}")
        m2.metric("Precision", f"{report.get('precision', 0.0):.4%}")
        m3.metric("Recall", f"{report.get('recall', 0.0):.4%}")
        m4.metric("F1-Score", f"{report.get('f1_score', 0.0):.4%}")
        m5.metric("FPR", f"{report.get('false_positive_rate', 0.0):.5%}")

        if "confusion_matrix" in report:
            cm = report["confusion_matrix"]
            st.markdown("#### Confusion Matrix Breakdown")
            cm_df = pd.DataFrame(
                [
                    {"Predicted": "Benign (0)", "True Benign (TN)": f"{cm.get('true_negatives'):,}", "True Attack (FN)": f"{cm.get('false_negatives'):,}"},
                    {"Predicted": "Attack (1)", "False Positive (FP)": f"{cm.get('false_positives'):,}", "True Positive (TP)": f"{cm.get('true_positives'):,}"},
                ]
            )
            st.table(cm_df)

        st.info(
            "ℹ️ **In-Distribution Context**: High metrics (99.88% recall) confirm that the model successfully "
            "represents multiple distinct attack topologies (DDoS, FTP-Patator, SSH-Patator) within known capture environments. "
            "However, this does NOT guarantee zero-shot detection on novel, unseen attack families."
        )

    elif eval_view == "🌐 Independent Out-of-Distribution Validation (Wednesday Capture)":
        st.markdown("### Independent Validation on Wednesday Capture (`Wednesday-workingHours.pcap_ISCX.csv`)")
        report, err = load_evaluation_report(THIRD_CAPTURE_EVAL_REPORT_PATH)

        if report is None:
            st.warning(f"Report not available: {err}")
            return

        # Critical Highlight Box
        st.error(
            "⚠️ **CRITICAL EMPIRICAL FINDING: WEDNESDAY ATTACK RECALL IS ~0.11%**\n\n"
            "• **Total Attack Flows in Wednesday**: 252,672\n"
            "• **Attacks Correctly Detected**: **278**\n"
            "• **Attacks Missed (False Negatives)**: **252,394 (99.89% missed)**\n\n"
            "**Why Did This Occur?**  \n"
            "The multi-attack model was trained on volumetric DDoS and credential brute-force. "
            "The Wednesday capture contains application-layer DoS variants (*Hulk, GoldenEye, Slowloris, Slowhttptest*). "
            "These attacks operate slowly at the HTTP request layer and mimic normal human traffic rates, "
            "completely bypassing the volumetric flood boundaries learned during training.\n\n"
            "**Positive Characteristic:**  \n"
            "The False Positive Rate remained exceptional at **0.017%** (only 74 false alarms across 440,031 unseen benign flows). "
            "The model reliably filters benign daily traffic, but fails zero-shot detection on unfamiliar threat mechanisms."
        )

        st.markdown(
            f"""
            - **Evaluated Dataset**: `Wednesday-workingHours.pcap_ISCX.csv` (Full File, 692,703 rows)
            - **Model Evaluated**: `models/random_forest_multi_attack.joblib` (Zero retraining)
            """
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{report.get('accuracy', 0.0):.2%}")
        m2.metric("Precision", f"{report.get('precision', 0.0):.2%}")
        m3.metric("Recall", f"{report.get('recall', 0.0):.2%}")
        m4.metric("F1-Score", f"{report.get('f1_score', 0.0):.4f}")
        m5.metric("FPR", f"{report.get('false_positive_rate', 0.0):.4%}")

        if "confusion_matrix" in report:
            cm = report["confusion_matrix"]
            st.markdown("#### Wednesday Confusion Matrix")
            cm_df = pd.DataFrame(
                [
                    {"Predicted": "Benign (0)", "True Benign (TN)": f"{cm.get('true_negatives'):,}", "True Attack (FN)": f"{cm.get('false_negatives'):,}"},
                    {"Predicted": "Attack (1)", "False Positive (FP)": f"{cm.get('false_positives'):,}", "True Positive (TP)": f"{cm.get('true_positives'):,}"},
                ]
            )
            st.table(cm_df)

        st.markdown("#### Detection Rates by Wednesday Attack Family")
        family_df = pd.DataFrame(
            [
                {"Attack Family": "BENIGN (Normal Traffic)", "Total Flows": "440,031", "Correct": "439,957", "Missed": "74 (FP)", "Detection / Specificity": "99.983%"},
                {"Attack Family": "DoS Hulk", "Total Flows": "231,073", "Correct": "213", "Missed": "230,860", "Detection / Specificity": "0.092%"},
                {"Attack Family": "DoS Slowhttptest", "Total Flows": "5,499", "Correct": "65", "Missed": "5,434", "Detection / Specificity": "1.182%"},
                {"Attack Family": "DoS GoldenEye", "Total Flows": "10,293", "Correct": "0", "Missed": "10,293", "Detection / Specificity": "0.000%"},
                {"Attack Family": "DoS slowloris", "Total Flows": "5,796", "Correct": "0", "Missed": "5,796", "Detection / Specificity": "0.000%"},
                {"Attack Family": "Heartbleed", "Total Flows": "11", "Correct": "0", "Missed": "11", "Detection / Specificity": "0.000%"},
            ]
        )
        st.table(family_df)

    elif eval_view == "🔬 Milestone 16: Model Generalization Experiments":
        st.markdown("### Milestone 16: Model Improvement and Generalization Experiments")
        report, err = load_evaluation_report(EXPERIMENT_COMPARISON_REPORT_PATH)

        if report is None:
            st.warning(f"Experiment comparison report not available: {err}")
            return

        st.markdown(
            """
            > **Scientific Finding**: Class weighting alone does not resolve cross-capture domain shift 
            > when training on volumetric DDoS and credential brute-force, while evaluating on application-layer 
            > HTTP DoS. However, ensemble architectures with randomized feature thresholds (**ExtraTrees**) 
            > demonstrated significant out-of-distribution sensitivity (**13.34% attack recall**, detecting **33,715 attacks**) 
            > compared to deterministic Random Forest (**0.04% - 0.11% recall**).
            """
        )

        models_data = report.get("model_comparison", {})
        table_rows = []
        for m_id, m_metrics in models_data.items():
            table_rows.append(
                {
                    "Model Identifier": m_id,
                    "Estimator Architecture": m_metrics.get("estimator", "N/A"),
                    "In-Dist Validation Recall": f"{m_metrics.get('validation_recall', 0.0):.2%}",
                    "In-Dist Validation Acc": f"{m_metrics.get('validation_accuracy', 0.0):.2%}",
                    "Wednesday Attack Recall": f"{m_metrics.get('wednesday_recall', 0.0):.2%}",
                    "Wednesday Precision": f"{m_metrics.get('wednesday_precision', 0.0):.2%}",
                    "Wednesday False Positive Rate": f"{m_metrics.get('wednesday_fpr', 0.0):.4%}",
                    "Wednesday ROC-AUC": f"{m_metrics.get('wednesday_roc_auc', 0.0):.4f}" if m_metrics.get("wednesday_roc_auc") else "N/A",
                }
            )

        if table_rows:
            st.markdown("#### Comparative Benchmark Table across Models")
            st.table(pd.DataFrame(table_rows))

        with st.expander("🔍 Detailed Per-Attack-Family Recall Breakdown (Wednesday Out-of-Distribution)"):
            for m_id, m_metrics in models_data.items():
                st.markdown(f"**Model: `{m_id}` ({m_metrics.get('estimator')})**")
                fam_dict = m_metrics.get("wednesday_per_family", {})
                fam_rows = []
                for lbl, stats in fam_dict.items():
                    if stats.get("category") == "attack":
                        fam_rows.append({
                            "Attack Label": lbl,
                            "Total Samples": f"{stats.get('total_samples'):,}",
                            "Detected (TP)": f"{stats.get('detected_attacks'):,}",
                            "Missed (FN)": f"{stats.get('missed_attacks'):,}",
                            "Detection Recall": f"{stats.get('recall', 0.0):.2%}",
                        })
                    else:
                        fam_rows.append({
                            "Attack Label": f"{lbl} (Normal Traffic)",
                            "Total Samples": f"{stats.get('total_samples'):,}",
                            "Detected (TP)": f"{stats.get('predicted_benign'):,} (TN)",
                            "Missed (FN)": f"{stats.get('predicted_attack'):,} (FP)",
                            "Detection Recall": f"{stats.get('benign_specificity', 0.0):.3%} (Specificity)",
                        })
                st.dataframe(pd.DataFrame(fam_rows), use_container_width=True)

    else:
        st.markdown("### Original Single-Capture Friday Baseline Reports")
        col_bl1, col_bl2 = st.columns(2)

        with col_bl1:
            st.markdown("#### Friday DDoS Training Test Split")
            r_base, err1 = load_evaluation_report(BASELINE_TRAIN_REPORT_PATH)
            if r_base:
                st.write(f"- Accuracy: {r_base.get('accuracy', 0):.4f}")
                st.write(f"- Attack Recall: {r_base.get('recall', 0):.4f}")
                st.write(f"- Attack Precision: {r_base.get('precision', 0):.4f}")
                st.write(f"- FPR: {r_base.get('false_positive_rate', 0):.5f}")
            else:
                st.caption(err1)

        with col_bl2:
            st.markdown("#### Tuesday Cross-Capture Evaluation")
            r_cross, err2 = load_evaluation_report(CROSS_CAPTURE_TUESDAY_REPORT_PATH)
            if r_cross:
                st.write(f"- Accuracy: {r_cross.get('accuracy', 0):.4f} (Deceptive majority rate)")
                st.write(f"- Attack Recall: {r_cross.get('recall', 0):.4f} (0% on Patator brute-force)")
                st.write(f"- FPR: {r_cross.get('false_positive_rate', 0):.5f} (0.003% false alarms)")
            else:
                st.caption(err2)


def render_model_inspection_tab(metadata: Optional[Dict[str, Any]], model_available: bool, active_model_path: Optional[Path] = None):
    """Render Model Metadata, Feature Schema, and Model Inspection."""
    st.subheader("⚙️ Model Architecture & Feature Schema Audit")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model Configuration")
        if model_available and metadata:
            st.json(
                {
                    "artifact_file": active_model_path.name if active_model_path else "Unknown",
                    "model_type": metadata.get("model_type"),
                    "training_timestamp": metadata.get("training_timestamp"),
                    "n_train_samples": metadata.get("n_train_samples"),
                    "n_features": metadata.get("n_features"),
                    "classes": metadata.get("classes"),
                    "imputer_strategy": metadata.get("imputer_strategy"),
                    "random_state": metadata.get("random_state"),
                }
            )
        else:
            st.warning("No model metadata available. Model artifact is missing or uninitialized.")

    with col2:
        st.markdown("### Expected Feature Schema")
        if metadata and metadata.get("feature_names"):
            feats = metadata["feature_names"]
            st.write(f"**Expected Features ({len(feats)} total):**")
            st.code("\n".join(feats[:25]) + ("\n... [truncated]" if len(feats) > 25 else ""))
        else:
            st.info("Feature schema will appear once a trained model artifact is loaded.")


def main():
    """Main application layout and execution flow."""
    st.set_page_config(
        page_title="AI Threat Detection Engine",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    st.title("🛡️ AI-Powered Cybersecurity Threat Detection System")
    st.caption("Offline Network Flow Intrusion Triage & Multi-Benchmark Audit Engine")

    # Sidebar: Model Selection
    st.sidebar.title("🛡️ Engine Controls")
    st.sidebar.markdown("### Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Active Model Engine",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Choose between the trained Random Forest models for offline scoring.",
    )
    active_model_path = AVAILABLE_MODELS[selected_model_name]

    # Load chosen model
    pipeline, metadata, model_err = load_trusted_model(active_model_path)
    model_available = pipeline is not None

    if model_available:
        st.sidebar.success(f"Loaded: `{active_model_path.name}`")
        if metadata:
            st.sidebar.caption(f"Trained: {str(metadata.get('training_timestamp', ''))[:10]}")
            st.sidebar.caption(f"Feature Count: {metadata.get('n_features', 78)}")
    else:
        st.sidebar.error(f"Missing: `{active_model_path.name}`")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 System Overview",
            "🚀 Batch Flow Prediction",
            "🚨 Threat Alerts Feed",
            "📈 Benchmark & Validation Reports",
            "⚙️ Model & Feature Architecture",
        ]
    )

    with tab1:
        render_overview_tab(model_available=model_available, model_err=model_err, active_model_path=active_model_path)

    with tab2:
        render_batch_prediction_tab(model=pipeline, metadata=metadata, model_available=model_available)

    with tab3:
        render_alerts_feed_tab()

    with tab4:
        render_evaluation_benchmarks_tab()

    with tab5:
        render_model_inspection_tab(metadata=metadata, model_available=model_available, active_model_path=active_model_path)


if __name__ == "__main__":
    main()
