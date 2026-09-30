"""Rule-based query classifier. The category re-weights the retrieval channels (see CATEGORY_MULTIPLIERS)."""
from __future__ import annotations

import re

from .preprocess import QueryInfo

CATEGORIES = ["FUNCTION_LOOKUP", "BUG_LOCALIZATION", "DATA_FLOW", "CONTROL_FLOW", "API_USAGE", "FILE_LOOKUP", "CLASS_LOOKUP",
              "ERROR_MESSAGE", "CONCEPTUAL_CODE_SEARCH", "DEPENDENCY_SEARCH", "CONFIGURATION_SEARCH", "GENERAL_SEMANTIC_SEARCH"]

# (dense, bm25, identifier) multipliers applied to the base weights, then renormalised to sum to 1.
CATEGORY_MULTIPLIERS: dict[str, tuple[float, float, float]] = {
    "FUNCTION_LOOKUP": (0.6, 1.4, 2.0), "CLASS_LOOKUP": (0.6, 1.3, 2.0), "FILE_LOOKUP": (0.5, 1.3, 1.8),
    "ERROR_MESSAGE": (0.9, 1.3, 0.8), "API_USAGE": (0.9, 1.2, 1.0), "CONCEPTUAL_CODE_SEARCH": (1.3, 0.8, 0.5),
    "DATA_FLOW": (1.2, 0.9, 0.8), "CONTROL_FLOW": (1.1, 1.0, 0.7), "BUG_LOCALIZATION": (1.0, 1.1, 0.9),
    "DEPENDENCY_SEARCH": (0.8, 1.3, 1.2), "CONFIGURATION_SEARCH": (0.8, 1.3, 1.1), "GENERAL_SEMANTIC_SEARCH": (1.0, 1.0, 1.0),
}

_RULES: list[tuple[str, str]] = [
    ("CONFIGURATION_SEARCH", r"\b(config|configuration|settings?|environment variable|env var|\.env|yaml|flags?|options?)\b"),
    ("DEPENDENCY_SEARCH", r"\b(depends? on|dependenc|imports?|requires?|package|installed|used by|who uses)\b"),
    ("BUG_LOCALIZATION", r"\b(bug|crash(es)?|fails?|failing|wrong|incorrect|broken|not working|why (does|is|do)|regression|off[- ]by)\b"),
    ("API_USAGE", r"\b(how (do i|to) use|usage of|endpoint|sdk|library|call(s|ing)? the \w+ api)\b"),
    ("DATA_FLOW", r"\b(passed (to|into)|flows?|before (being )?(passed|sent|reach|going|forward)|after|preprocess|transform|converts?|normaliz|sanitiz|incoming|into the)\b"),
    ("CONTROL_FLOW", r"\b(when|if|condition|loop|branch|retry|fallback|handles?|handled|order of|called (from|by)|happens)\b"),
    ("CONCEPTUAL_CODE_SEARCH", r"\b(how does|how is|implementation|algorithm|logic|mechanism|approach|responsible for)\b"),
]


def classify_query(q: QueryInfo) -> str:
    text = q.text.lower()
    n_words = len(text.split())
    if q.has_error_text and re.search(r"error|exception|traceback|raise|failed|cannot|could not|unable", text):
        return "ERROR_MESSAGE"
    if q.file_names:
        return "FILE_LOOKUP"
    if q.class_names and re.search(r"\bclass\b|\bdefin", text):
        return "CLASS_LOOKUP"
    where_is = re.search(r"\b(where is|definition of|defined|implementation of)\b", text)
    if (q.identifiers and (n_words <= 6 or where_is)) or (q.function_names and n_words <= 6):
        return "FUNCTION_LOOKUP"
    for cat, rx in _RULES:
        if re.search(rx, text):
            if cat == "API_USAGE" and not (q.libraries or "api" in text or "how" in text):
                continue
            return cat
    if q.libraries:
        return "API_USAGE"
    return "GENERAL_SEMANTIC_SEARCH"


def category_weights(base: tuple[float, float, float], category: str) -> tuple[float, float, float]:
    m = CATEGORY_MULTIPLIERS.get(category, (1.0, 1.0, 1.0))
    w = [b * mm for b, mm in zip(base, m)]
    s = sum(w) or 1.0
    return tuple(x / s for x in w)  # type: ignore[return-value]
