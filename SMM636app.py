
# Heart Disease Decision Tree — Shiny App
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend required by Shiny
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# 1.  Prepare data

APP_DIR  = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "Heart_disease_cleveland_new.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}\n"
        "Fix: place Heart_disease_cleveland_new.csv next to SMM636app.py"
    )

df = pd.read_csv(CSV_PATH)

# Separate predictors (X) from the binary target (y: 0 = no disease, 1 = disease)
X = df.drop("target", axis=1)
y = df["target"]

# 80 / 20 stratified split — preserves class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# High-level dataset statistics displayed on the Introduction tab
N_SAMPLES      = len(df)
N_FEATURES     = X.shape[1]
DISEASE_PCT    = round(float((y == 1).mean() * 100), 1)
NO_DISEASE_PCT = round(100 - DISEASE_PCT, 1)


depths       = list(range(1, 11))
train_scores = []
test_scores  = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test,  y_test))

best_idx      = int(np.argmax(test_scores))
BEST_DEPTH    = depths[best_idx]
BEST_TEST_ACC = float(test_scores[best_idx])

# Fit the best model once so we can extract feature importances
best_clf = DecisionTreeClassifier(max_depth=BEST_DEPTH, random_state=42)
best_clf.fit(X_train, y_train)

# Feature importances sorted descending
FEATURE_IMP = (
    pd.Series(best_clf.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

# Plain-English labels for every feature (used in charts and insights)
FEATURE_LABELS = {
    "thal":     "Thalassaemia type",
    "ca":       "No. of major vessels",
    "cp":       "Chest pain type",
    "oldpeak":  "ST depression (exercise)",
    "thalach":  "Max heart rate achieved",
    "age":      "Age",
    "slope":    "ST slope",
    "exang":    "Exercise-induced angina",
    "chol":     "Serum cholesterol",
    "trestbps": "Resting blood pressure",
    "fbs":      "Fasting blood sugar",
    "restecg":  "Resting ECG result",
    "sex":      "Sex",
}

# 3.  design system — custom CSS
#     All visual tokens live here so the styling is consistent and easy to update

CUSTOM_CSS = """
/* ─── Colour tokens ─────────────────────────
   Primary   #1e3a5f  (deep navy)
   Accent    #2563eb  (blue)
   Purple    #7c3aed
   Teal      #0891b2
   Green     #059669
   Amber     #d97706
   Red       #dc2626
   BG        #f1f5f9
   Surface   #ffffff
   Border    #e2e8f0
   Text-1    #0f172a
   Text-2    #475569
   Text-3    #94a3b8
────────────────────────────────────────── */

/* ── Base ──────────────────────────────────── */
body {
    background : #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    font-size  : 14px;
    color      : #0f172a;
    line-height: 1.5;
}

/* ── Cards ─────────────────────────────────── */
.card {
    border       : none !important;
    border-radius: 14px !important;
    box-shadow   : 0 1px 3px rgba(15,23,42,.07),
                   0 6px 20px rgba(15,23,42,.06) !important;
    background   : #ffffff !important;
}
.card-header {
    background    : #ffffff !important;
    border-bottom : 1px solid #f1f5f9 !important;
    font-size     : 0.82rem !important;
    font-weight   : 600 !important;
    letter-spacing: 0.01em !important;
    color         : #64748b !important;
    border-radius : 14px 14px 0 0 !important;
    padding       : 14px 22px !important;
}

/* ── Navigation tabs ───────────────────────── */
.nav-tabs {
    border-bottom : 2px solid #e2e8f0 !important;
    margin-bottom : 22px !important;
    gap           : 2px;
}
.nav-tabs .nav-link {
    color         : #64748b !important;
    font-size     : 0.875rem !important;
    font-weight   : 500 !important;
    border        : none !important;
    border-bottom : 2px solid transparent !important;
    border-radius : 0 !important;
    padding       : 11px 24px !important;
    margin-bottom : -2px !important;
    transition    : color .15s, border-color .15s;
}
.nav-tabs .nav-link:hover {
    color              : #1d4ed8 !important;
    background         : transparent !important;
    border-bottom-color: #93c5fd !important;
}
.nav-tabs .nav-link.active {
    color         : #1d4ed8 !important;
    font-weight   : 600 !important;
    background    : transparent !important;
    border-bottom : 2px solid #1d4ed8 !important;
}

/* ── Sidebar ───────────────────────────────── */
.bslib-sidebar-layout > .sidebar {
    background  : #ffffff !important;
    border-right: 1px solid #f1f5f9 !important;
}

/* ── Section micro-labels ──────────────────── */
.sec-label {
    font-size     : 0.78rem;
    font-weight   : 600;
    letter-spacing: 0.01em;
    color         : #64748b;
    margin        : 0 0 8px 0;
    display       : block;
}

/* ── Form controls ─────────────────────────── */
.form-label {
    font-size    : 0.8rem !important;
    font-weight  : 500 !important;
    color        : #475569 !important;
    margin-bottom: 3px !important;
}
.form-control, .form-select {
    border-radius: 8px !important;
    border       : 1.5px solid #e2e8f0 !important;
    font-size    : 0.875rem !important;
    color        : #0f172a !important;
    transition   : border-color .15s, box-shadow .15s;
    padding      : 7px 11px !important;
}
.form-control:focus, .form-select:focus {
    border-color: #2563eb !important;
    box-shadow  : 0 0 0 3px rgba(37,99,235,.12) !important;
    outline     : none !important;
}

/* ── Range slider ──────────────────────────── */
.irs--shiny .irs-bar,
.irs--shiny .irs-bar-edge { background: #2563eb !important; border-color: #2563eb !important; }
.irs--shiny .irs-handle    { background: #fff !important; border: 2px solid #2563eb !important;
                              box-shadow: 0 1px 4px rgba(37,99,235,.25) !important; }
.irs--shiny .irs-from,
.irs--shiny .irs-to,
.irs--shiny .irs-single    { background: #1d4ed8 !important; border-radius: 6px !important; }

/* ── Primary button ────────────────────────── */
.btn-primary {
    background   : #1d4ed8 !important;
    border       : none !important;
    border-radius: 8px !important;
    font-size    : 0.85rem !important;
    font-weight  : 600 !important;
    padding      : 9px 20px !important;
    width        : 100% !important;
    letter-spacing: 0.01em !important;
    transition   : background .15s, box-shadow .15s !important;
}
.btn-primary:hover {
    background: #1e40af !important;
    box-shadow: 0 4px 14px rgba(29,78,216,.35) !important;
}

/* ── Tables ────────────────────────────────── */
table.dataframe {
    font-size      : 0.82rem !important;
    width          : 100% !important;
    border-collapse: collapse !important;
}
table.dataframe thead th {
    background    : #f8fafc !important;
    color         : #475569 !important;
    font-size     : 0.7rem !important;
    font-weight   : 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    border-bottom : 2px solid #e2e8f0 !important;
    padding       : 10px 14px !important;
}
table.dataframe tbody td {
    padding      : 9px 14px !important;
    border-bottom: 1px solid #f8fafc !important;
    color        : #334155 !important;
}
table.dataframe tbody tr:last-child td { border-bottom: none !important; }
table.dataframe tbody tr:hover td      { background: #f8fafc !important; }

/* ── Divider ───────────────────────────────── */
hr { border-color: #f1f5f9 !important; margin: 16px 0 !important; }

/* ── Insight cards ─────────────────────────── */
.ic {
    padding      : 14px 16px;
    border-radius: 10px;
    background   : #f8fafc;
    border-left  : 4px solid #2563eb;
    margin-bottom: 12px;
}
.ic.purple { border-left-color: #7c3aed; }
.ic.green  { border-left-color: #059669; }
.ic.amber  { border-left-color: #d97706; }
"""

# 4.  UI layout

app_ui = ui.page_fluid(

    ui.tags.style(CUSTOM_CSS),

    ui.div(

        # ── Global page header 
        ui.div(
            ui.div(
                ui.h2(
                    "Heart Disease · Decision Tree Explorer",
                    style="margin:0; font-weight:700; color:#0f172a; "
                          "font-size:1.4rem; letter-spacing:-0.015em;"
                ),
                style="flex:1;"
            ),
            ui.div(
                ui.span(
                    f"Best depth: {BEST_DEPTH}",
                    style="background:#dbeafe; color:#1d4ed8; padding:5px 14px; "
                          "border-radius:20px; font-size:0.78rem; font-weight:600; margin-right:8px;"
                ),
                ui.span(
                    f"Test accuracy: {BEST_TEST_ACC:.0%}",
                    style="background:#dcfce7; color:#15803d; padding:5px 14px; "
                          "border-radius:20px; font-size:0.78rem; font-weight:600;"
                ),
                style="display:flex; align-items:center; flex-shrink:0;"
            ),
            style=(
                "display:flex; align-items:center; justify-content:space-between; "
                "padding:18px 26px; background:#ffffff; border-radius:14px; "
                "box-shadow:0 1px 3px rgba(15,23,42,.07), 0 6px 20px rgba(15,23,42,.06); "
                "margin-bottom:22px;"
            )
        ),

        ui.navset_tab(

            # ╔═══════════════════════════════════════════╗
            # ║  TAB 1 · DECISION TREE (Core Model)       ║
            # ╚═══════════════════════════════════════════╝
            ui.nav_panel(
                "Decision Tree",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            ui.tags.span("Tree depth", class_="sec-label"),
                            ui.input_slider("depth", None, min=1, max=10, value=3),
                        ),
                        width=260
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Decision tree structure"),
                            ui.div(
                                ui.output_plot("tree_plot", height="560px"),
                                style="overflow:auto; padding:8px;"
                            )
                        ),
                        ui.card(
                            ui.card_header("Performance metrics"),
                            ui.output_ui("kpi_box"),
                            ui.div(
                                ui.tags.span("Confusion matrix", class_="sec-label",
                                             style="padding:0 22px; display:block; margin-top:4px;"),
                                ui.div(
                                    ui.output_table("cm_table"),
                                    style="padding:0 22px 22px 22px;"
                                )
                            )
                        ),
                        col_widths=(7, 5)
                    )
                )
            ),

            # ╔═══════════════════════════════════════════╗
            # ║  TAB 2 · MODEL COMPLEXITY                 ║
            # ╚═══════════════════════════════════════════╝
            ui.nav_panel(
                "Model Complexity",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Train vs test accuracy by max depth"),
                        ui.div(
                            ui.output_plot("acc_plot", height="480px"),
                            style="padding:8px 12px;"
                        )
                    ),
                    col_widths=(12,)
                )
            ),

            # ╔═══════════════════════════════════════════╗
            # ║  TAB 3 · RANDOM FOREST (Extension Model)  ║
            # ╚═══════════════════════════════════════════╝
            ui.nav_panel(
                "Random Forest",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(
                            ui.tags.span("No. of trees", class_="sec-label"),
                            ui.input_slider("rf_n", None, min=10, max=300, value=100, step=10),
                        ),
                        width=260
                    ),
                    ui.layout_columns(
                        # Middle: RF feature importance
                        ui.card(
                            ui.card_header("Feature importance  ·  Random Forest"),
                            ui.div(
                                ui.output_plot("rf_feat_imp_plot", height="480px"),
                                style="padding:18px 22px;"
                            )
                        ),
                        # Right: RF metrics + confusion matrix
                        ui.card(
                            ui.card_header("Performance metrics"),
                            ui.output_ui("rf_kpi_box"),
                            ui.div(
                                ui.tags.span("Confusion matrix", class_="sec-label",
                                             style="padding:0 22px; display:block; margin-top:4px;"),
                                ui.div(
                                    ui.output_table("rf_cm_table"),
                                    style="padding:0 22px 22px 22px;"
                                )
                            )
                        ),
                        col_widths=(7, 5)
                    )
                )
            ),

            # ╔═══════════════════════════════════════════╗
            # ║  TAB 3 · PATIENT SIMULATOR (Application)  ║
            # ╚═══════════════════════════════════════════╝
            ui.nav_panel(
                "Patient Simulator",
                ui.layout_columns(
                    ui.div(),
                    ui.card(
                        ui.card_header("Patient simulator"),
                        ui.div(
                            ui.p(
                                "Enter a patient's clinical values and click Predict. "
                                "Features not listed here are filled with dataset medians.",
                                style="font-size:0.82rem; color:#94a3b8; "
                                      "margin-bottom:20px; line-height:1.6;"
                            ),
                            ui.layout_columns(
                                ui.div(
                                    ui.input_numeric("age_in", "Age (years)",
                                                     value=55, min=20, max=90),
                                    ui.input_select("thal_in", "Thalassaemia type",
                                                    choices={"0": "0 – Normal",
                                                             "1": "1 – Fixed defect",
                                                             "2": "2 – Reversible defect",
                                                             "3": "3 – Other"},
                                                    selected="2"),
                                    ui.input_select("cp_in", "Chest pain type",
                                                    choices={"0": "0 – Typical angina",
                                                             "1": "1 – Atypical angina",
                                                             "2": "2 – Non-anginal",
                                                             "3": "3 – Asymptomatic"},
                                                    selected="2"),
                                ),
                                ui.div(
                                    ui.input_select("ca_in", "No. of major vessels",
                                                    choices=["0", "1", "2", "3", "4"],
                                                    selected="0"),
                                    ui.input_numeric("trestbps_in", "Resting BP (mmHg)",
                                                     value=130, min=80, max=220),
                                    ui.input_numeric("oldpeak_in", "ST depression (Oldpeak)",
                                                     value=1.0, min=0.0, max=6.0, step=0.1),
                                ),
                                col_widths=(6, 6)
                            ),
                            ui.div(style="height:16px;"),
                            ui.input_action_button("predict_btn", "Run prediction",
                                                   class_="btn btn-primary",
                                                   style="width:auto; padding:10px 32px;"),
                            ui.div(style="height:16px;"),
                            ui.output_ui("pred_result"),
                            style="padding:22px 26px;"
                        )
                    ),
                    ui.div(),
                    col_widths=(1, 10, 1)
                )
            ),
        ),

        style="max-width:1440px; margin:auto; padding:24px;"
    )
)

# 5.  sever logic

def server(input, output, session):

    # ── Reactive: retrain tree whenever depth slider changes ─────────────
    @reactive.calc
    def fitted_model():
        d = int(input.depth())
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        return clf

    # ── Reactive: evaluate the current model on the held-out test set ────
    @reactive.calc
    def preds_and_metrics():
        clf    = fitted_model()
        y_pred = clf.predict(X_test)
        return (
            accuracy_score( y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(   y_test, y_pred, zero_division=0),
            f1_score(       y_test, y_pred, zero_division=0),
            confusion_matrix(y_test, y_pred),
        )

    # ── Helper: render one KPI row ────────────────────────────────────────
    def _kpi_row(value, label, subtitle, color):
        return ui.div(
            ui.div(
                style=f"width:4px; background:{color}; border-radius:4px; "
                      f"margin-right:16px; flex-shrink:0; align-self:stretch;"
            ),
            ui.div(
                ui.div(
                    ui.span(value,
                            style=f"font-size:1.35rem; font-weight:700; color:{color};"),
                    ui.span(" / 1.00",
                            style="font-size:0.78rem; color:#cbd5e1;"),
                ),
                ui.p(label,
                     style="margin:2px 0 0 0; font-size:0.8rem; "
                           "font-weight:600; color:#334155;"),
                ui.p(subtitle,
                     style="margin:1px 0 0 0; font-size:0.72rem; color:#94a3b8;"),
            ),
            style="display:flex; align-items:center; padding:14px 22px; "
                  "border-bottom:1px solid #f8fafc;"
        )

    # ── Output: KPI cards 
    @output
    @render.ui
    def kpi_box():
        acc, prec, rec, f1, _ = preds_and_metrics()
        return ui.div(
            _kpi_row(f"{acc:.3f}",  "Test Accuracy",
                     "Proportion of correct predictions on unseen data",     "#1d4ed8"),
            _kpi_row(f"{prec:.3f}", "Precision · Disease",
                     "Of those flagged as Disease, fraction truly positive", "#7c3aed"),
            _kpi_row(f"{rec:.3f}",  "Recall · Disease",
                     "Of all true Disease cases, fraction correctly found",  "#0891b2"),
            _kpi_row(f"{f1:.3f}",   "F1-Score · Disease",
                     "Harmonic mean of Precision and Recall",                "#059669"),
        )

    # ── Output: confusion matrix 
    @output
    @render.table
    def cm_table():
        _, _, _, _, cm = preds_and_metrics()
        return pd.DataFrame(
            cm,
            index=["Actual: No Disease", "Actual: Disease"],
            columns=["Pred: No Disease", "Pred: Disease"]
        )

    # ── Output: decision tree visualisation 
    @output
    @render.plot
    def tree_plot():
        clf = fitted_model()
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("#ffffff")
        plot_tree(
            clf,
            feature_names=X.columns,
            class_names=["No Disease", "Disease"],
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=9,
        )
        ax.set_title(
            f"Decision Tree  ·  max_depth = {int(input.depth())}",
            fontsize=13, fontweight="bold", color="#1e293b", pad=16
        )
        plt.tight_layout()
        return fig

    # ── Output: train vs test accuracy curve ──────────────────────────────
    @output
    @render.plot
    def acc_plot():
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(depths, train_scores, marker="o", label="Train accuracy",
                color="#1d4ed8", linewidth=2.2, markersize=6.5, zorder=3)
        ax.plot(depths, test_scores, marker="o", label="Test accuracy",
                color="#f97316", linewidth=2.2, markersize=6.5, zorder=3)
        ax.axvline(x=BEST_DEPTH, color="#059669", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Best depth ({BEST_DEPTH})", zorder=2)
        ax.scatter([BEST_DEPTH], [BEST_TEST_ACC], s=120, color="#059669", zorder=5)
        for i, (tr, te) in enumerate(zip(train_scores, test_scores)):
            if tr - te > 0.05:
                ax.axvspan(depths[i] - 0.4, depths[i] + 0.4,
                           alpha=0.06, color="#dc2626", zorder=1)
        ax.set_xlabel("Max Depth", fontsize=11, color="#475569", labelpad=8)
        ax.set_ylabel("Accuracy", fontsize=11, color="#475569", labelpad=8)
        ax.set_title("Train vs Test Accuracy by Max Depth",
                     fontsize=13, fontweight="bold", color="#1e293b", pad=14)
        ax.legend(framealpha=0.95, fontsize=10, frameon=True)
        ax.tick_params(colors="#64748b", labelsize=10)
        ax.set_xticks(depths)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#e2e8f0")
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(axis="y", color="#f1f5f9", linewidth=1.2, zorder=0)
        plt.tight_layout()
        return fig

    # ── Output: patient prediction result
    @output
    @render.ui
    @reactive.event(input.predict_btn)
    def pred_result():
        clf = fitted_model()

        # Build a one-row dataframe; populate only the 6 user-supplied features
        row = pd.DataFrame([np.nan] * X.shape[1], index=X.columns).T
        row.loc[0, "age"]      = float(input.age_in())
        row.loc[0, "thal"]     = float(input.thal_in())
        row.loc[0, "cp"]       = float(input.cp_in())
        row.loc[0, "ca"]       = float(input.ca_in())
        row.loc[0, "trestbps"] = float(input.trestbps_in())
        row.loc[0, "oldpeak"]  = float(input.oldpeak_in())

        # Fill remaining features with training-set medians
        med = X.median(numeric_only=True)
        for col in X.columns:
            if pd.isna(row.loc[0, col]) and col in med.index:
                row.loc[0, col] = float(med[col])

        pred_class   = int(clf.predict(row)[0])
        prob_disease = (float(clf.predict_proba(row)[0][1])
                        if hasattr(clf, "predict_proba") else None)

        label = "Heart Disease Detected" if pred_class == 1 else "No Heart Disease"
        color = "#dc2626" if pred_class == 1 else "#059669"
        bg    = "#fef2f2" if pred_class == 1 else "#f0fdf4"

        content = [
            ui.p(label,
                 style=f"margin:0 0 3px 0; font-weight:700; "
                       f"color:{color}; font-size:0.95rem;")
        ]
        if prob_disease is not None:
            content.append(
                ui.p(f"P(Disease) = {prob_disease:.2f}",
                     style="margin:0 0 6px 0; font-size:0.85rem; color:#475569;")
            )
        content.append(
            ui.p("Model prediction only — always consult a medical professional.",
                 style="margin:0; font-size:0.72rem; color:#94a3b8; line-height:1.4;")
        )

        return ui.div(
            *content,
            style=f"padding:14px 16px; background:{bg}; border-radius:10px; "
                  f"border-left:4px solid {color}; margin-top:4px;"
        )

    # ── Random Forest: retrain when n_estimators slider changes 
    @reactive.calc
    def rf_model():
        n = int(input.rf_n())
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    # ── Random Forest: evaluate on test set
    @reactive.calc
    def rf_metrics():
        rf     = rf_model()
        y_pred = rf.predict(X_test)
        return (
            accuracy_score( y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(   y_test, y_pred, zero_division=0),
            f1_score(       y_test, y_pred, zero_division=0),
            confusion_matrix(y_test, y_pred),
        )

    # ── Output: RF KPI cards ──────────────────────────────────────────────
    @output
    @render.ui
    def rf_kpi_box():
        acc, prec, rec, f1, _ = rf_metrics()
        return ui.div(
            _kpi_row(f"{acc:.3f}",  "Test Accuracy",
                     "Proportion of correct predictions on unseen data",     "#1d4ed8"),
            _kpi_row(f"{prec:.3f}", "Precision · Disease",
                     "Of those flagged as Disease, fraction truly positive", "#7c3aed"),
            _kpi_row(f"{rec:.3f}",  "Recall · Disease",
                     "Of all true Disease cases, fraction correctly found",  "#0891b2"),
            _kpi_row(f"{f1:.3f}",   "F1-Score · Disease",
                     "Harmonic mean of Precision and Recall",                "#059669"),
        )

    # ── Output: RF confusion matrix
    @output
    @render.table
    def rf_cm_table():
        _, _, _, _, cm = rf_metrics()
        return pd.DataFrame(
            cm,
            index=["Actual: No Disease", "Actual: Disease"],
            columns=["Pred: No Disease", "Pred: Disease"]
        )

    # ── Output: RF feature importance chart
    @output
    @render.plot
    def rf_feat_imp_plot():
        rf  = rf_model()
        imp = (
            pd.Series(rf.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
        )
        labels = [FEATURE_LABELS.get(f, f) for f in imp.index]
        values = imp.values

        p66 = np.percentile(values, 66)
        p33 = np.percentile(values, 33)
        colors = [
            "#1d4ed8" if v == values.max() else
            "#2563eb" if v >= p66 else
            "#93c5fd" if v >= p33 else
            "#dbeafe"
            for v in values
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       height=0.65, zorder=3)

        for bar, val in zip(bars, values[::-1]):
            if val > 0.008:
                ax.text(
                    val + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    fontsize=9, color="#475569"
                )

        ax.set_xlabel("Importance Score", fontsize=10, color="#475569", labelpad=8)
        ax.set_title(
            f"Feature Importance  (n_estimators = {int(input.rf_n())})",
            fontsize=12, fontweight="bold", color="#1e293b", pad=12
        )
        ax.tick_params(colors="#64748b", labelsize=9.5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(axis="x", color="#f1f5f9", linewidth=1.2, zorder=0)
        ax.set_xlim(0, values.max() * 1.22)
        plt.tight_layout()
        return fig

# 6.  Lunch the app
app = App(app_ui, server)
