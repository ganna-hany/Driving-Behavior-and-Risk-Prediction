"""
Desktop GUI for the Driving Behavior Classification project.

This file only loads artifacts exported by the Jupyter notebook.
It does not train models.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("TkAgg")

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "dataset.csv",
    "model.pkl",
    "preprocessor.pkl",
    "all_models.pkl",
    "evaluation_results.pkl",
    "metadata.json",
]

NAV_ITEMS = [
    ("Home", "home"),
    ("Dataset Details", "dataset"),
    ("Data Analysis", "analysis"),
    ("Preprocessing", "preprocessing"),
    ("Feature Engineering", "features"),
    ("Models", "models"),
    ("Model Evaluation", "evaluation"),
    ("Prediction", "predict"),
    ("Prediction History", "history"),
]


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        missing = [name for name in REQUIRED_FILES if not (base_dir / name).exists()]
        if missing:
            joined = "\n".join(f"  - {name}" for name in missing)
            raise FileNotFoundError(
                "Required ML artifacts were not found:\n"
                f"{joined}\n\n"
                "Run all cells in the Jupyter notebook, including the last export cell."
            )

        self.df = pd.read_csv(base_dir / "dataset.csv")
        self.scaler = joblib.load(base_dir / "preprocessor.pkl")
        self.models = joblib.load(base_dir / "all_models.pkl")
        self.best_model = joblib.load(base_dir / "model.pkl")
        self.eval_data = joblib.load(base_dir / "evaluation_results.pkl")
        with open(base_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.eval_data["feature_names"]
        self.target_names = self.eval_data["target_names"]
        self.class_names = self.eval_data["class_names"]
        self.label_offset = int(self.eval_data.get("label_offset", 1))
        self.feature_stats = self.eval_data.get("feature_stats", {})
        self.best_model_name = self.eval_data["best_model_name"]

        model_cols = [c for c in self.feature_names if c in self.df.columns]
        self.X = self.df[model_cols].copy()

    def decode_label(self, encoded: int) -> str:
        original = int(encoded) + self.label_offset
        return self.class_names.get(original, self.class_names.get(str(original), str(original)))

    def predict(self, model_name: str, values: dict[str, float]) -> tuple[str, float | None, np.ndarray | None]:
        row = pd.DataFrame([[values[name] for name in self.feature_names]], columns=self.feature_names)
        scaled = self.scaler.transform(row)
        model = self.models[model_name]
        encoded = int(model.predict(scaled)[0])
        label = self.decode_label(encoded)
        proba = None
        vector = None
        if hasattr(model, "predict_proba"):
            vector = model.predict_proba(scaled)[0]
            proba = float(np.max(vector))
        return label, proba, vector


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Driving Behavior Classification")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        try:
            self.store = ArtifactStore(BASE_DIR)
        except Exception as exc:
            self.store = None
            self._show_startup_error(str(exc))
            return

        self.history: list[dict] = []
        self.current_page = None
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.input_vars: dict[str, ctk.StringVar] = {}
        self.proba_labels: dict[str, ctk.CTkLabel] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=("#eef2f6", "#12151c"))
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.show_page("home")

    def _show_startup_error(self, message: str) -> None:
        box = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=15))
        box.pack(fill="both", expand=True, padx=24, pady=24)
        box.insert("1.0", "Could not start the GUI.\n\n" + message)
        box.configure(state="disabled")

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color=("#1f3b57", "#0d1117"))
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="Driving Behavior\nML Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="left",
        ).pack(anchor="w", padx=18, pady=(24, 8))
        ctk.CTkLabel(
            sidebar,
            text="Notebook trains  ·  GUI predicts",
            font=ctk.CTkFont(size=12),
            text_color=("#c5d4e4", "#9aa8b8"),
        ).pack(anchor="w", padx=18, pady=(0, 20))

        for label, key in NAV_ITEMS:
            btn = ctk.CTkButton(
                sidebar,
                text=label,
                anchor="w",
                height=36,
                fg_color="transparent",
                hover_color=("#2d5a82", "#1c2330"),
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = btn

        ctk.CTkLabel(
            sidebar,
            text="Artifacts loaded from disk",
            font=ctk.CTkFont(size=11),
            text_color=("#c5d4e4", "#7d8a98"),
        ).pack(side="bottom", padx=18, pady=18)

    def _highlight_nav(self, key: str) -> None:
        for name, btn in self.nav_buttons.items():
            if name == key:
                btn.configure(fg_color=("#2d5a82", "#243044"))
            else:
                btn.configure(fg_color="transparent")

    def clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()

    def show_page(self, key: str) -> None:
        self.current_page = key
        self._highlight_nav(key)
        self.clear_content()
        pages = {
            "home": self.page_home,
            "dataset": self.page_dataset,
            "analysis": self.page_analysis,
            "preprocessing": self.page_preprocessing,
            "features": self.page_features,
            "models": self.page_models,
            "evaluation": self.page_evaluation,
            "predict": self.page_predict,
            "history": self.page_history,
        }
        pages[key]()

    def _scrollable(self) -> ctk.CTkScrollableFrame:
        frame = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _card(self, parent, title: str | None = None) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid_columnconfigure(0, weight=1)
        if title:
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=16, weight="bold")).pack(
                anchor="w", padx=16, pady=(14, 8)
            )
        return card

    def _kv(self, parent, label: str, value: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(row, text=label, width=220, anchor="w", text_color=("#4b5a6b", "#9aa8b8")).pack(side="left")
        ctk.CTkLabel(row, text=str(value), anchor="w", wraplength=720, justify="left").pack(side="left", fill="x", expand=True)

    def _embed_figure(self, parent, fig: Figure, height: int = 420) -> None:
        holder = ctk.CTkFrame(parent, fg_color="transparent", height=height)
        holder.pack(fill="both", expand=True, padx=8, pady=8)
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

    def page_home(self) -> None:
        page = self._scrollable()
        meta = self.store.metadata
        hero = self._card(page)
        hero.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(hero, text=meta["project_title"], font=ctk.CTkFont(size=28, weight="bold")).pack(
            anchor="w", padx=18, pady=(18, 6)
        )
        ctk.CTkLabel(
            hero,
            text=meta["project_description"],
            wraplength=900,
            justify="left",
            font=ctk.CTkFont(size=14),
        ).pack(anchor="w", padx=18, pady=(0, 16))

        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.pack(fill="x", pady=8)
        for i in range(4):
            stats.grid_columnconfigure(i, weight=1)
        cards = [
            ("Problem type", meta["problem_type"]),
            ("Samples", f"{meta['n_samples']}"),
            ("Model features", f"{meta['n_features']}"),
            ("Target variable", meta["target_variable"]),
        ]
        for i, (title, value) in enumerate(cards):
            card = ctk.CTkFrame(stats, corner_radius=12)
            card.grid(row=0, column=i, padx=6, sticky="nsew")
            ctk.CTkLabel(card, text=title, text_color=("#4b5a6b", "#9aa8b8")).pack(padx=14, pady=(14, 2), anchor="w")
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=18, weight="bold")).pack(padx=14, pady=(0, 14), anchor="w")

        info = self._card(page, "Project summary")
        info.pack(fill="x", pady=12)
        self._kv(info, "Source dataset", meta.get("source_file", "sero_features_4.csv"))
        self._kv(info, "Best model", f"{meta['best_model_name']} ({meta['best_model_type']})")
        self._kv(info, "Best accuracy", fmt_pct(self.store.eval_data["best_accuracy"]))
        self._kv(info, "Classes", ", ".join(meta["target_names"]))
        self._kv(info, "Engineered feature", ", ".join(meta["engineered_features"]))

    def page_dataset(self) -> None:
        page = self._scrollable()
        df = self.store.df
        overview = self._card(page, "Dataset overview")
        overview.pack(fill="x", pady=(0, 12))
        self._kv(overview, "Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
        self._kv(overview, "Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        self._kv(overview, "Duplicate rows remaining", str(int(df.duplicated().sum())))
        self._kv(overview, "Total missing values", str(int(df.isna().sum().sum())))

        cols = self._card(page, "Columns, types, missing values, unique counts")
        cols.pack(fill="x", pady=12)
        header = ctk.CTkFrame(cols, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(0, 4))
        for text, width in [("Column", 180), ("Dtype", 90), ("Missing", 80), ("Unique", 80), ("Example", 220)]:
            ctk.CTkLabel(header, text=text, width=width, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left")

        dtypes = df.dtypes.astype(str)
        missing = df.isna().sum()
        nuniques = df.nunique(dropna=False)
        for col in df.columns:
            row = ctk.CTkFrame(cols, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=1)
            example = str(df[col].iloc[0])
            values = [
                (col, 180),
                (dtypes[col], 90),
                (str(int(missing[col])), 80),
                (str(int(nuniques[col])), 80),
                (example[:42], 220),
            ]
            for text, width in values:
                ctk.CTkLabel(row, text=text, width=width, anchor="w").pack(side="left")

        dist = self._card(page, "Target / class distribution")
        dist.pack(fill="x", pady=12)
        if "Behavior_Name" in df.columns:
            counts = df["Behavior_Name"].value_counts()
        else:
            counts = df["Target"].value_counts().sort_index()
        for label, count in counts.items():
            self._kv(dist, str(label), f"{int(count)} samples ({count / len(df) * 100:.1f}%)")

        stats_card = self._card(page, "Numeric statistics")
        stats_card.pack(fill="x", pady=12)
        desc = df.describe().T
        header = ctk.CTkFrame(stats_card, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(0, 4))
        for text, width in [("Feature", 150), ("Mean", 90), ("Std", 90), ("Min", 90), ("Max", 90)]:
            ctk.CTkLabel(header, text=text, width=width, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left")
        for col, row in desc.iterrows():
            line = ctk.CTkFrame(stats_card, fg_color="transparent")
            line.pack(fill="x", padx=12, pady=1)
            vals = [
                (str(col), 150),
                (f"{row['mean']:.4f}", 90),
                (f"{row['std']:.4f}", 90),
                (f"{row['min']:.4f}", 90),
                (f"{row['max']:.4f}", 90),
            ]
            for text, width in vals:
                ctk.CTkLabel(line, text=text, width=width, anchor="w").pack(side="left")

    def page_analysis(self) -> None:
        page = self._scrollable()
        df = self.store.df
        hue_col = "Behavior_Name" if "Behavior_Name" in df.columns else "Target"

        card = self._card(page, "Target distribution")
        card.pack(fill="x", pady=(0, 12))
        fig = Figure(figsize=(8.5, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        counts = df[hue_col].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#3b82f6")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        self._embed_figure(card, fig, 320)

        card = self._card(page, "Feature distributions (selected numeric features)")
        card.pack(fill="x", pady=12)
        plot_cols = [c for c in ["AccTotalMagnitude", "AccMeanX", "AccMeanY", "GyroMeanZ", "GyroStdX"] if c in df.columns]
        fig = Figure(figsize=(9.2, 4.2), dpi=100)
        axes = fig.subplots(1, len(plot_cols))
        if len(plot_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_cols):
            ax.hist(df[col].dropna(), bins=30, color="#22c55e", alpha=0.85)
            ax.set_title(col, fontsize=9)
        fig.tight_layout()
        self._embed_figure(card, fig, 360)

        card = self._card(page, "Correlation heatmap (model features)")
        card.pack(fill="x", pady=12)
        corr = self.store.X.corr(numeric_only=True)
        fig = Figure(figsize=(8.8, 7.2), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
        ax.set_yticklabels(corr.columns, fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._embed_figure(card, fig, 620)

        card = self._card(page, "Boxplots by driving behavior")
        card.pack(fill="x", pady=12)
        box_cols = [c for c in ["AccTotalMagnitude", "GyroStdZ", "AccStdX"] if c in df.columns]
        fig = Figure(figsize=(9.2, 4.0), dpi=100)
        axes = fig.subplots(1, len(box_cols))
        if len(box_cols) == 1:
            axes = [axes]
        groups = list(df.groupby(hue_col))
        labels = [str(name) for name, _ in groups]
        for ax, col in zip(axes, box_cols):
            data = [group[col].dropna().values for _, group in groups]
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        self._embed_figure(card, fig, 360)

        if {"AccTotalMagnitude", "AccMeanY"}.issubset(df.columns):
            card = self._card(page, "Total acceleration vs AccMeanY")
            card.pack(fill="x", pady=12)
            fig = Figure(figsize=(8.5, 4.2), dpi=100)
            ax = fig.add_subplot(111)
            for name, group in df.groupby(hue_col):
                ax.scatter(group["AccTotalMagnitude"], group["AccMeanY"], s=12, alpha=0.7, label=str(name))
            ax.set_xlabel("AccTotalMagnitude")
            ax.set_ylabel("AccMeanY")
            ax.legend(fontsize=8)
            fig.tight_layout()
            self._embed_figure(card, fig, 380)

    def page_preprocessing(self) -> None:
        page = self._scrollable()
        card = self._card(page, "Preprocessing steps from the notebook")
        card.pack(fill="x")
        steps = self.store.eval_data["preprocessing_steps"]
        labels = {
            "missing_values": "Missing values",
            "duplicates": "Duplicates",
            "encoding": "Encoding",
            "scaling": "Scaling",
            "outlier_handling": "Outlier handling",
            "train_test_split": "Train / test split",
            "smote": "SMOTE",
            "label_mapping": "Target encoding",
        }
        for key, title in labels.items():
            if key in steps:
                self._kv(card, title, steps[key])
        self._kv(card, "Training samples", str(self.store.eval_data.get("n_train", "—")))
        self._kv(card, "Test samples", str(self.store.eval_data.get("n_test", "—")))
        self._kv(card, "Scaler type", type(self.store.scaler).__name__)

    def page_features(self) -> None:
        page = self._scrollable()
        data = self.store.eval_data
        original = data.get("original_features", [])
        engineered = data.get("engineered_features", [])
        selected = data.get("selected_features", data["feature_names"])

        card = self._card(page, "Feature engineering from the notebook")
        card.pack(fill="x", pady=(0, 12))
        self._kv(card, "Original sensor features", f"{len(original)} columns from sero_features_4.csv")
        self._kv(card, "Generated features", ", ".join(engineered) if engineered else "None")
        self._kv(card, "Generated formula", "AccTotalMagnitude = sqrt(AccMeanX² + AccMeanY² + AccMeanZ²)")
        self._kv(card, "Feature selection", "No feature selection was applied. All remaining numeric columns were used.")
        self._kv(card, "Final model features", str(len(selected)))
        self._kv(card, "Dropped before training", "Target, Behavior_Name")

        def list_card(title: str, items: list[str]) -> None:
            box = self._card(page, title)
            box.pack(fill="x", pady=8)
            text = ctk.CTkTextbox(box, height=min(220, 24 + 18 * max(len(items), 3)))
            text.pack(fill="x", padx=12, pady=(0, 12))
            text.insert("1.0", "\n".join(items))
            text.configure(state="disabled")

        list_card("Original features", original)
        list_card("Generated features", engineered)
        list_card("Selected / final features used by the model", selected)

    def page_models(self) -> None:
        page = self._scrollable()
        intro = self._card(page, "Trained models exported from the notebook")
        intro.pack(fill="x", pady=(0, 12))
        self._kv(intro, "Best model", self.store.best_model_name)
        self._kv(intro, "Note", "These objects were loaded from all_models.pkl. The GUI does not retrain them.")

        results = self.store.eval_data["results_by_model"]
        for name, info in results.items():
            title = name + ("  •  BEST" if name == self.store.best_model_name else "")
            card = self._card(page, title)
            card.pack(fill="x", pady=8)
            self._kv(card, "Model type", info.get("model_type", type(self.store.models[name]).__name__))
            params = info.get("parameters") or {}
            if params:
                pretty = ", ".join(f"{k}={v}" for k, v in params.items())
                self._kv(card, "Main parameters", pretty)
            else:
                self._kv(card, "Main parameters", "See the notebook training cell.")
            self._kv(card, "Test accuracy", fmt_pct(info.get("accuracy")))

    def page_evaluation(self) -> None:
        page = self._scrollable()
        eval_data = self.store.eval_data
        results = eval_data["results_by_model"]

        summary = self._card(page, "Test-set metrics (exported from the notebook)")
        summary.pack(fill="x", pady=(0, 12))
        header = ctk.CTkFrame(summary, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(0, 4))
        widths = [200, 90, 90, 90, 90, 90]
        titles = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        for text, width in zip(titles, widths):
            ctk.CTkLabel(header, text=text, width=width, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left")
        for name, info in results.items():
            row = ctk.CTkFrame(summary, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            vals = [
                name,
                fmt_pct(info.get("accuracy")),
                fmt_metric(info.get("precision")),
                fmt_metric(info.get("recall")),
                fmt_metric(info.get("f1")),
                fmt_metric(info.get("roc_auc")) if info.get("roc_auc") is not None else "N/A",
            ]
            for text, width in zip(vals, widths):
                ctk.CTkLabel(row, text=text, width=width, anchor="w").pack(side="left")

        best_name = eval_data["best_model_name"]
        report_card = self._card(page, f"Classification report — {best_name}")
        report_card.pack(fill="x", pady=12)
        report = eval_data.get("classification_report", {})
        text = ctk.CTkTextbox(report_card, height=220)
        text.pack(fill="x", padx=12, pady=(0, 12))
        lines = ["Class                  Precision    Recall       F1      Support"]
        for key, values in report.items():
            if not isinstance(values, dict):
                continue
            lines.append(
                f"{key:22} {values.get('precision', 0):10.4f} {values.get('recall', 0):10.4f} "
                f"{values.get('f1-score', 0):10.4f} {values.get('support', 0):8.0f}"
            )
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

        cm_card = self._card(page, f"Confusion matrix — {best_name}")
        cm_card.pack(fill="x", pady=12)
        cm = np.array(eval_data["confusion_matrix"])
        fig = Figure(figsize=(6.8, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=eval_data["target_names"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
        ax.set_title("Counts on the held-out test set")
        fig.tight_layout()
        self._embed_figure(cm_card, fig, 460)

        note = self._card(page)
        note.pack(fill="x", pady=8)
        ctk.CTkLabel(
            note,
            text="ROC-AUC is macro OVR on the test set when the loaded model supports predict_proba. "
            "SVC was trained without probability=True, so ROC-AUC is unavailable for that model.",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", padx=16, pady=12)

    def page_predict(self) -> None:
        page = self._scrollable()
        top = self._card(page, "Predict driving behavior")
        top.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            top,
            text="Inputs are scaled with the saved StandardScaler, then passed to a model loaded from disk.",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        controls = ctk.CTkFrame(top, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(controls, text="Model").pack(side="left", padx=(0, 8))
        self.model_choice = ctk.CTkComboBox(controls, values=list(self.store.models.keys()), width=260)
        self.model_choice.set(self.store.best_model_name)
        self.model_choice.pack(side="left", padx=6)
        ctk.CTkButton(controls, text="Load random dataset row", command=self._load_random_sample, width=200).pack(
            side="left", padx=8
        )
        ctk.CTkButton(controls, text="Fill medians", command=self._fill_medians, width=130).pack(side="left", padx=8)

        form_card = self._card(page, "Model features")
        form_card.pack(fill="x", pady=8)
        grid = ctk.CTkFrame(form_card, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 12))
        self.input_vars = {}
        columns = 3
        for i, name in enumerate(self.store.feature_names):
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=i // columns, column=i % columns, sticky="ew", padx=8, pady=4)
            ctk.CTkLabel(cell, text=name, anchor="w").pack(fill="x")
            stats = self.store.feature_stats.get(name, {})
            default = stats.get("median", 0.0)
            var = ctk.StringVar(value=f"{default:.6f}")
            entry = ctk.CTkEntry(cell, textvariable=var, width=200)
            entry.pack(fill="x")
            self.input_vars[name] = var
            for j in range(columns):
                grid.grid_columnconfigure(j, weight=1)

        action = ctk.CTkFrame(page, fg_color="transparent")
        action.pack(fill="x", pady=8)
        ctk.CTkButton(action, text="Predict", width=180, height=40, command=self._run_prediction).pack(side="left")

        self.result_card = self._card(page, "Result")
        self.result_card.pack(fill="x", pady=8)
        self.result_label = ctk.CTkLabel(self.result_card, text="No prediction yet", font=ctk.CTkFont(size=22, weight="bold"))
        self.result_label.pack(anchor="w", padx=16, pady=(0, 6))
        self.conf_label = ctk.CTkLabel(self.result_card, text="")
        self.conf_label.pack(anchor="w", padx=16, pady=(0, 10))
        self.proba_box = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.proba_box.pack(fill="x", padx=16, pady=(0, 12))

    def _collect_inputs(self) -> dict[str, float]:
        values = {}
        for name, var in self.input_vars.items():
            raw = var.get().strip()
            try:
                values[name] = float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be numeric. Got: {raw!r}") from exc
        if {"AccMeanX", "AccMeanY", "AccMeanZ", "AccTotalMagnitude"}.issubset(values):
            values["AccTotalMagnitude"] = float(
                np.sqrt(values["AccMeanX"] ** 2 + values["AccMeanY"] ** 2 + values["AccMeanZ"] ** 2)
            )
            self.input_vars["AccTotalMagnitude"].set(f"{values['AccTotalMagnitude']:.6f}")
        return values

    def _fill_medians(self) -> None:
        for name, var in self.input_vars.items():
            median = self.store.feature_stats.get(name, {}).get("median", 0.0)
            var.set(f"{median:.6f}")

    def _load_random_sample(self) -> None:
        row = self.store.df.sample(1, random_state=None).iloc[0]
        for name, var in self.input_vars.items():
            var.set(f"{float(row[name]):.6f}")

    def _run_prediction(self) -> None:
        try:
            values = self._collect_inputs()
        except ValueError as exc:
            self.result_label.configure(text="Invalid input")
            self.conf_label.configure(text=str(exc))
            return

        model_name = self.model_choice.get()
        label, proba, vector = self.store.predict(model_name, values)
        self.result_label.configure(text=label)
        if proba is None:
            self.conf_label.configure(text="This model has no predict_proba(), so confidence is unavailable.")
        else:
            self.conf_label.configure(text=f"Confidence (max class probability): {proba * 100:.2f}%")

        for child in self.proba_box.winfo_children():
            child.destroy()
        if vector is not None:
            for idx, p in enumerate(vector):
                class_name = self.store.decode_label(idx)
                line = ctk.CTkFrame(self.proba_box, fg_color="transparent")
                line.pack(fill="x", pady=2)
                ctk.CTkLabel(line, text=class_name, width=200, anchor="w").pack(side="left")
                bar = ctk.CTkProgressBar(line, width=360)
                bar.set(float(p))
                bar.pack(side="left", padx=8)
                ctk.CTkLabel(line, text=f"{p * 100:.2f}%").pack(side="left")

        self.history.append(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "prediction": label,
                "probability": None if proba is None else round(proba, 6),
                "inputs": {k: round(v, 6) for k, v in values.items()},
            }
        )

    def page_history(self) -> None:
        page = self._scrollable()
        header = self._card(page, "Predictions made in this GUI session")
        header.pack(fill="x", pady=(0, 12))
        ctk.CTkButton(header, text="Clear history", width=140, command=self._clear_history).pack(
            anchor="w", padx=16, pady=(0, 12)
        )

        if not self.history:
            empty = self._card(page)
            empty.pack(fill="x")
            ctk.CTkLabel(empty, text="No predictions yet. Use the Prediction page first.").pack(
                padx=16, pady=16, anchor="w"
            )
            return

        for i, item in enumerate(reversed(self.history), start=1):
            card = self._card(page, f"{item['time']}  ·  {item['model']}")
            card.pack(fill="x", pady=8)
            self._kv(card, "Prediction", item["prediction"])
            prob = item["probability"]
            self._kv(card, "Probability / confidence", "N/A" if prob is None else f"{prob * 100:.2f}%")
            preview = ", ".join(f"{k}={v}" for k, v in list(item["inputs"].items())[:8])
            self._kv(card, "Inputs (first 8)", preview + (" ..." if len(item["inputs"]) > 8 else ""))
            details = ctk.CTkTextbox(card, height=110)
            details.pack(fill="x", padx=12, pady=(0, 12))
            details.insert("1.0", json.dumps(item["inputs"], indent=2))
            details.configure(state="disabled")

    def _clear_history(self) -> None:
        self.history.clear()
        self.show_page("history")


if __name__ == "__main__":
    app = App()
    app.mainloop()
