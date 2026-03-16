"""Metrics-grounded chat helpers for the Insights page."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI


PROVIDER_CONFIG = {
    "Groq Cloud": {
        "env_var": "GROQ_API_KEY",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile",
    },
    "Cerebras Cloud": {
        "env_var": "CEREBRAS_API_KEY",
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "default_model": "llama-3.3-70b",
    },
}

# Keep Insights answers short and actionable.
MAX_RESPONSE_WORDS = 140
MAX_RESPONSE_CHARS = 900
MAX_RESPONSE_PARAGRAPHS = 3

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def resolve_metric_columns(df: pd.DataFrame):
    """Resolve canonical metric columns across legacy/current CSV formats."""
    model_col = "model_name" if "model_name" in df.columns else "model"
    date_col = "date" if "date" in df.columns else "timestamp"
    prec_col = "overall_precision" if "overall_precision" in df.columns else "precision"
    rec_col = "overall_recall" if "overall_recall" in df.columns else "recall"
    return model_col, date_col, prec_col, rec_col


def load_metrics_csv(metrics_source) -> pd.DataFrame:
    """Load metrics data from a file path or uploaded file-like object."""
    if metrics_source is None:
        return pd.DataFrame()

    try:
        if hasattr(metrics_source, "seek"):
            metrics_source.seek(0)
        df = pd.read_csv(metrics_source)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    model_col, date_col, prec_col, rec_col = resolve_metric_columns(df)
    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df[prec_col] = pd.to_numeric(work_df[prec_col], errors="coerce")
    work_df[rec_col] = pd.to_numeric(work_df[rec_col], errors="coerce")
    if "f1_score" in work_df.columns:
        work_df["f1_score"] = pd.to_numeric(work_df["f1_score"], errors="coerce")
    else:
        denom = work_df[prec_col].fillna(0.0) + work_df[rec_col].fillna(0.0)
        work_df["f1_score"] = ((2 * work_df[prec_col].fillna(0.0) * work_df[rec_col].fillna(0.0)) / denom.where(denom != 0, 1)).fillna(0.0)

    return work_df


def detect_metrics_signature(metrics_source) -> str:
    """Build a stable signature for the currently selected metrics source."""
    if metrics_source is None:
        return "missing"

    if hasattr(metrics_source, "getvalue"):
        payload = metrics_source.getvalue()
        name = getattr(metrics_source, "name", "uploaded_metrics.csv")
        return f"upload:{name}:{len(payload)}"

    path = Path(metrics_source)
    if not path.exists():
        return f"path:{path}:missing"
    stat = path.stat()
    return f"path:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def _tokenize(text: str):
    return [token for token in re.findall(r"[a-z0-9_]+", str(text).lower()) if token not in STOPWORDS]


def _format_pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def _display_class_name(class_key: str) -> str:
    return str(class_key).replace("_", " ").title()


def extract_class_keys(df: pd.DataFrame):
    """Return normalized class keys found in per-class metric columns."""
    class_keys = set()
    for column in df.columns:
        column_name = str(column)
        if column_name.startswith("precision_") and column_name not in {"precision", "overall_precision"}:
            class_keys.add(column_name.replace("precision_", "", 1))
        if column_name.startswith("recall_") and column_name not in {"recall", "overall_recall"}:
            class_keys.add(column_name.replace("recall_", "", 1))
    return sorted(class_keys)


def build_class_analysis(df: pd.DataFrame):
    """Summarize average, latest, and best run performance per class."""
    if df.empty:
        return []

    model_col, date_col, _, _ = resolve_metric_columns(df)
    class_rows = []
    work_df = df.sort_values(date_col, na_position="last")

    for class_key in extract_class_keys(work_df):
        precision_col = f"precision_{class_key}"
        recall_col = f"recall_{class_key}"
        if precision_col not in work_df.columns and recall_col not in work_df.columns:
            continue

        class_df = work_df[[model_col, date_col]].copy()
        class_df[precision_col] = pd.to_numeric(work_df.get(precision_col), errors="coerce").fillna(0.0)
        class_df[recall_col] = pd.to_numeric(work_df.get(recall_col), errors="coerce").fillna(0.0)
        denom = class_df[precision_col] + class_df[recall_col]
        class_df["f1"] = ((2 * class_df[precision_col] * class_df[recall_col]) / denom.where(denom != 0, 1)).fillna(0.0)
        class_df = class_df[(class_df[precision_col] > 0) | (class_df[recall_col] > 0) | (class_df["f1"] > 0)]
        if class_df.empty:
            continue

        latest_row = class_df.sort_values(date_col, na_position="last").tail(1).iloc[0]
        best_row = class_df.sort_values("f1", ascending=False).iloc[0]
        avg_precision = float(class_df[precision_col].mean())
        avg_recall = float(class_df[recall_col].mean())
        avg_f1 = float(class_df["f1"].mean())

        class_rows.append(
            {
                "class_key": class_key,
                "class_name": _display_class_name(class_key),
                "runs": int(len(class_df)),
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "latest_model": str(latest_row[model_col]),
                "latest_date": latest_row[date_col],
                "latest_precision": float(latest_row[precision_col]),
                "latest_recall": float(latest_row[recall_col]),
                "latest_f1": float(latest_row["f1"]),
                "best_model": str(best_row[model_col]),
                "best_date": best_row[date_col],
                "best_precision": float(best_row[precision_col]),
                "best_recall": float(best_row[recall_col]),
                "best_f1": float(best_row["f1"]),
                "recall_gap": float(avg_precision - avg_recall),
            }
        )

    return sorted(class_rows, key=lambda item: item["avg_f1"])


def build_workspace_snapshot(df: pd.DataFrame):
    """Return top-line summary values for the Insights dashboard."""
    if df.empty:
        return {
            "run_count": 0,
            "model_count": 0,
            "latest_run": None,
            "best_run": None,
            "weakest_classes": [],
        }

    model_col, date_col, prec_col, rec_col = resolve_metric_columns(df)
    work_df = df.copy().sort_values(date_col, na_position="last")
    latest_row = work_df.tail(1).iloc[0]
    best_row = work_df.sort_values("f1_score", ascending=False).iloc[0]
    class_rows = build_class_analysis(work_df)

    latest_run = {
        "model": str(latest_row.get(model_col, "-")),
        "date": latest_row.get(date_col),
        "precision": float(latest_row.get(prec_col, 0.0)),
        "recall": float(latest_row.get(rec_col, 0.0)),
        "f1": float(latest_row.get("f1_score", 0.0)),
    }
    best_run = {
        "model": str(best_row.get(model_col, "-")),
        "date": best_row.get(date_col),
        "precision": float(best_row.get(prec_col, 0.0)),
        "recall": float(best_row.get(rec_col, 0.0)),
        "f1": float(best_row.get("f1_score", 0.0)),
    }

    return {
        "run_count": int(len(work_df)),
        "model_count": int(work_df[model_col].dropna().nunique()),
        "latest_run": latest_run,
        "best_run": best_run,
        "weakest_classes": class_rows[:3],
    }


def _build_documents(df: pd.DataFrame):
    """Build small retrievable metric documents from the CSV."""
    if df.empty:
        return []

    model_col, date_col, prec_col, rec_col = resolve_metric_columns(df)
    work_df = df.copy().sort_values(date_col, na_position="last")
    docs = []

    latest_row = work_df.tail(1).iloc[0]
    best_row = work_df.sort_values("f1_score", ascending=False).iloc[0]
    class_rows = build_class_analysis(work_df)
    models = sorted(work_df[model_col].dropna().astype(str).unique().tolist())
    date_min = work_df[date_col].min()
    date_max = work_df[date_col].max()

    docs.append(
        {
            "title": "metrics_overview",
            "text": (
                f"The metrics file contains {len(work_df)} runs across {len(models)} models: {', '.join(models)}. "
                f"The recorded date range is {date_min} to {date_max}. "
                f"The CSV tracks precision, recall, F1, false positives, false negatives, and per-class metrics. "
                f"It does not contain direct classification accuracy."
            ),
        }
    )
    docs.append(
        {
            "title": "latest_run",
            "text": (
                f"Latest run: model {latest_row.get(model_col, '-')} on {latest_row.get(date_col)} with precision {_format_pct(latest_row.get(prec_col, 0.0))}, "
                f"recall {_format_pct(latest_row.get(rec_col, 0.0))}, and F1 {_format_pct(latest_row.get('f1_score', 0.0))}."
            ),
        }
    )
    docs.append(
        {
            "title": "best_run",
            "text": (
                f"Best F1 run: model {best_row.get(model_col, '-')} on {best_row.get(date_col)} with precision {_format_pct(best_row.get(prec_col, 0.0))}, "
                f"recall {_format_pct(best_row.get(rec_col, 0.0))}, and F1 {_format_pct(best_row.get('f1_score', 0.0))}."
            ),
        }
    )

    for class_row in class_rows:
        docs.append(
            {
                "title": f"class_{class_row['class_key']}",
                "text": (
                    f"Class {class_row['class_name']} appears in {class_row['runs']} measured runs. "
                    f"Average precision is {_format_pct(class_row['avg_precision'])}, average recall is {_format_pct(class_row['avg_recall'])}, "
                    f"and average F1 is {_format_pct(class_row['avg_f1'])}. Latest run for this class is model {class_row['latest_model']} on {class_row['latest_date']} "
                    f"with precision {_format_pct(class_row['latest_precision'])}, recall {_format_pct(class_row['latest_recall'])}, and F1 {_format_pct(class_row['latest_f1'])}. "
                    f"Best recorded F1 for this class is {_format_pct(class_row['best_f1'])} from model {class_row['best_model']} on {class_row['best_date']}. "
                    f"The average precision minus recall gap is {class_row['recall_gap']:+.1%}."
                ),
            }
        )

    latest_by_model = work_df.groupby(model_col, as_index=False).tail(1)
    for _, model_row in latest_by_model.iterrows():
        model_name = str(model_row[model_col])
        model_history = work_df[work_df[model_col] == model_name].sort_values(date_col, na_position="last")
        first_row = model_history.head(1).iloc[0]
        docs.append(
            {
                "title": f"model_{model_name}",
                "text": (
                    f"Model {model_name} has {len(model_history)} recorded runs. Its latest run on {model_row[date_col]} has precision {_format_pct(model_row[prec_col])}, "
                    f"recall {_format_pct(model_row[rec_col])}, and F1 {_format_pct(model_row['f1_score'])}. Compared with its first run on {first_row[date_col]}, "
                    f"precision changed by {float(model_row[prec_col]) - float(first_row[prec_col]):+.1%}, recall changed by {float(model_row[rec_col]) - float(first_row[rec_col]):+.1%}, "
                    f"and F1 changed by {float(model_row['f1_score']) - float(first_row['f1_score']):+.1%}."
                ),
            }
        )

    recent_cols = [column for column in [date_col, model_col, prec_col, rec_col, "f1_score", "false_positives", "false_negatives"] if column in work_df.columns]
    recent_csv = work_df[recent_cols].tail(6).to_csv(index=False)
    docs.append(
        {
            "title": "recent_runs_table",
            "text": "Recent runs table:\n" + recent_csv,
        }
    )

    for doc in docs:
        doc["tokens"] = set(_tokenize(doc["title"] + " " + doc["text"]))

    return docs


def retrieve_context(df: pd.DataFrame, query: str, max_docs: int = 6):
    """Retrieve the most relevant metric documents for a user question."""
    docs = _build_documents(df)
    if not docs:
        return [], ""

    query_tokens = set(_tokenize(query))
    class_tokens = {row["class_key"] for row in build_class_analysis(df)}

    ranked = []
    for index, doc in enumerate(docs):
        score = len(query_tokens & doc["tokens"])
        if any(class_token in query_tokens for class_token in class_tokens):
            for class_token in class_tokens:
                if class_token in query_tokens and class_token in doc["title"]:
                    score += 5
        if "f1" in query_tokens and ("f1" in doc["tokens"] or "best_run" == doc["title"]):
            score += 2
        if "precision" in query_tokens and "precision" in doc["tokens"]:
            score += 2
        if "recall" in query_tokens and "recall" in doc["tokens"]:
            score += 2
        if any(term in query_tokens for term in {"improve", "better", "worse", "weak", "improvement"}):
            if doc["title"].startswith("class_") or doc["title"] in {"latest_run", "best_run"}:
                score += 1
        ranked.append((score, index, doc))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [item[2] for item in ranked[:max_docs] if item[0] > 0]
    if not selected:
        selected = docs[:max_docs]

    context_parts = []
    for doc in selected:
        context_parts.append(f"[{doc['title']}]\n{doc['text']}")

    return selected, "\n\n".join(context_parts)


def build_system_prompt() -> str:
    """Instruction prompt for the metrics assistant."""
    return (
        "You are a model-evaluation analyst for a computer-vision workflow. "
        "Answer only from the provided metrics context and conversation history. "
        "Be concise, specific, and practical. Keep answers short and to the point. "
        "Prefer at most 3 short paragraphs or up to 6 brief bullets, and stay under about 140 words unless explicitly asked for detail. "
        "Use exact metric values when available. "
        "If the user asks how to improve performance, explain whether the issue is mainly precision, recall, or both, then give concrete corrective actions. "
        "If the metrics file does not support a claim, say that directly and name the extra evidence needed, such as confusion matrices, sample predictions, or more class-balanced data. "
        "Do not invent classes, runs, or improvements that are not grounded in the supplied context."
    )


def _limit_response_size(content: str) -> str:
    """Trim verbose model output to a compact, readable response."""
    text = str(content or "").strip()
    if not text:
        return ""

    # Hard character cap first for safety across providers.
    if len(text) > MAX_RESPONSE_CHARS:
        text = text[:MAX_RESPONSE_CHARS].rstrip()

    # Keep only first few paragraphs.
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    if len(paragraphs) > MAX_RESPONSE_PARAGRAPHS:
        paragraphs = paragraphs[:MAX_RESPONSE_PARAGRAPHS]
    text = "\n\n".join(paragraphs)

    # Final word cap to keep responses tight.
    words = text.split()
    if len(words) > MAX_RESPONSE_WORDS:
        text = " ".join(words[:MAX_RESPONSE_WORDS]).rstrip(" ,;:") + "..."

    return text


def build_messages(chat_history, metrics_context: str):
    """Compose the provider chat payload messages."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "system",
            "content": (
                "Retrieved metrics context for this conversation:\n"
                f"{metrics_context}\n\n"
                "Use this context as the knowledge base for your answer."
            ),
        },
    ]

    for message in chat_history[-10:]:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    return messages


def request_chat_completion(provider: str, api_key: str, model_name: str, messages, timeout: int = 90) -> str:
    """Call the selected provider with an OpenAI-compatible chat payload."""
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unsupported provider: {provider}")
    if not api_key:
        raise ValueError("Missing API key.")
    if not model_name:
        raise ValueError("Missing model name.")

    provider_config = PROVIDER_CONFIG[provider]
    
    # Get the base URL (remove /chat/completions endpoint)
    base_url = provider_config["endpoint"].replace("/chat/completions", "").replace("/v1", "")
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.25,
            messages=messages,
            max_tokens=220,
        )
        return _limit_response_size(str(response.choices[0].message.content).strip())
    except Exception as exc:
        raise RuntimeError(f"Provider request failed: {exc}") from exc
