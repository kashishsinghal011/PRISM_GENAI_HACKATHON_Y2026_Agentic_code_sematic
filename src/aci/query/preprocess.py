"""Query preprocessing: entity/identifier extraction + light concept expansion.

Programming vocabulary is NOT removed (only prose stop-words are), and identifiers are kept whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..errors import EmptyQueryError
from ..tokenize import STOPWORDS, split_identifier, stem, tokenize

FILE_RE = re.compile(r"[\w./-]+\.(?:py|js|jsx|ts|tsx|java|go|c|h|cpp|cc|hpp|cs|rs|kt|php|scala|ya?ml|json|toml|ini|cfg|env|md)\b", re.I)
CALL_RE = re.compile(r"\b([A-Za-z_][\w.]*)\(\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")
PASCAL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")
DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
FN_BEFORE = re.compile(r"\b([A-Za-z_]\w*)\s+(?:function|method|routine|func)\b", re.I)
FN_AFTER = re.compile(r"\b(?:function|method|def|func)\s+([A-Za-z_]\w*)", re.I)
CLASS_AFTER = re.compile(r"\bclass\s+([A-Za-z_]\w*)", re.I)
CLASS_BEFORE = re.compile(r"\b([A-Za-z_]\w*)\s+class\b", re.I)
ERR_RE = re.compile(r"\b\w+(?:Error|Exception)\b|traceback|\braise\b|error:|exception:|failed with|cannot |could not |unable to ", re.I)
LANG_RE = re.compile(r"\b(python|javascript|typescript|java|golang|go|rust|c\+\+|cpp|c#|csharp|kotlin|php|scala)\b", re.I)
LIBRARIES = {"numpy", "pandas", "requests", "jwt", "sqlalchemy", "flask", "django", "fastapi", "torch", "pytorch", "tensorflow",
             "redis", "boto3", "pytest", "scipy", "sklearn", "celery", "kafka", "grpc", "pydantic", "react", "express", "axios",
             "lodash", "spring", "hibernate", "junit", "asyncio", "aiohttp", "urllib", "json", "yaml", "argparse", "logging"}
NOT_NAMES = {"which", "this", "that", "the", "a", "an", "each", "any", "every", "what", "another", "some", "same", "one", "single",
             "my", "our", "your", "its", "their", "helper", "utility", "public", "private", "static", "python", "java", "js"}

# Related-term expansion. Keys/values are plain words; they are stemmed by tokenize().
CONCEPTS: dict[str, list[str]] = {
    "normalize": ["preprocess", "sanitize", "clean", "strip", "transform", "canonical", "trim", "lower"],
    "preprocess": ["normalize", "clean", "prepare", "transform", "tokenize"],
    "sanitize": ["clean", "escape", "validate", "strip", "filter", "normalize"],
    "validate": ["check", "verify", "assert", "ensure", "sanitize", "valid"],
    "authenticate": ["auth", "login", "token", "credential", "verify", "password", "session"],
    "authentication": ["auth", "login", "token", "credential", "verify", "password", "session"],
    "initialize": ["init", "setup", "create", "connect", "open", "configure"],
    "initialized": ["init", "setup", "create", "connect", "open", "configure"],
    "connect": ["connection", "open", "client", "session", "init"],
    "connection": ["connect", "open", "client", "session", "pool", "init"],
    "retry": ["attempt", "backoff", "retries", "sleep", "exponential", "fail"],
    "convert": ["transform", "parse", "map", "cast", "serialize", "decode"],
    "converts": ["transform", "parse", "map", "cast", "serialize", "decode"],
    "parse": ["decode", "load", "read", "extract", "tokenize", "split"],
    "request": ["req", "http", "api", "fetch", "call"],
    "validated": ["check", "verify", "decode", "assert", "valid"],
    "token": ["jwt", "bearer", "auth", "credential", "decode"],
    "error": ["exception", "raise", "fail", "except", "catch"],
    "cache": ["memo", "store", "ttl", "lru"],
    "serialize": ["dump", "encode", "json", "pickle", "marshal"],
    "deserialize": ["load", "decode", "parse", "json", "unmarshal"],
    "log": ["logger", "logging", "print", "debug", "info"],
    "delete": ["remove", "drop", "erase", "unlink", "pop"],
    "create": ["make", "build", "new", "init", "construct", "add"],
    "sort": ["order", "sorted", "compare", "rank", "key"],
    "search": ["find", "lookup", "query", "match", "scan"],
    "config": ["configuration", "settings", "options", "env", "params"],
    "database": ["db", "sql", "query", "connection", "cursor", "table"],
    "send": ["post", "emit", "dispatch", "write", "publish", "transmit"],
    "read": ["load", "open", "get", "fetch", "parse"],
    "write": ["save", "dump", "store", "put", "emit"],
    "compute": ["calculate", "count", "sum", "evaluate", "solve"],
    "count": ["sum", "total", "compute", "number", "tally"],
    "check": ["validate", "verify", "test", "ensure", "assert"],
}


_CONCEPT_STEMS = {stem(k): k for k in CONCEPTS}


@dataclass
class QueryInfo:
    raw: str
    text: str
    tokens: list[str]
    expansion: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    language: str = ""
    has_error_text: bool = False
    category: str = "GENERAL_SEMANTIC_SEARCH"

    @property
    def entity_names(self) -> list[str]:
        """Lowercased names a chunk could literally define/call (for identifier + structural features)."""
        return list(dict.fromkeys([*self.function_names, *self.class_names, *self.identifiers]))

    @property
    def dense_text(self) -> str:
        return self.text + (("  related: " + " ".join(self.expansion)) if self.expansion else "")

    def bm25_terms(self, expansion_weight: float = 0.35) -> dict[str, float]:
        terms: dict[str, float] = {}
        for t in self.tokens:
            terms[t] = terms.get(t, 0.0) + 1.0
        for ident in self.identifiers:  # exact identifiers get an extra boost
            terms[ident] = terms.get(ident, 0.0) + 1.0
        for t in self.expansion:
            terms.setdefault(t, expansion_weight)
        return terms


def preprocess_query(query: str, expand: bool = True) -> QueryInfo:
    if query is None or not str(query).strip():
        raise EmptyQueryError("Query is empty. Provide a natural-language question or identifier.")
    q = " ".join(str(query).split())
    files = FILE_RE.findall(q)
    q_nofiles = FILE_RE.sub(" ", q)
    idents: list[str] = []
    for rx in (CALL_RE, BACKTICK_RE):
        for m in rx.finditer(q_nofiles):
            idents.append(m.group(1))
    for rx in (SNAKE_RE, CAMEL_RE, PASCAL_RE, DOTTED_RE):
        idents += rx.findall(q_nofiles)
    idents = [i.rstrip("()").lower() for i in idents if len(i) > 2]
    dotted_parts = [p for i in idents if "." in i for p in i.split(".")]
    idents = list(dict.fromkeys(idents + dotted_parts))

    fnames = [m.lower() for rx in (FN_BEFORE, FN_AFTER) for m in rx.findall(q_nofiles) if m.lower() not in NOT_NAMES and m.lower() not in STOPWORDS]
    fnames += [m.group(1).lower() for m in CALL_RE.finditer(q_nofiles)]
    cnames = [m.lower() for rx in (CLASS_AFTER, CLASS_BEFORE) for m in rx.findall(q_nofiles) if m.lower() not in NOT_NAMES and m.lower() not in STOPWORDS]
    cnames += [p.lower() for p in PASCAL_RE.findall(q_nofiles)]

    tokens = tokenize(q_nofiles)
    for f in files:  # file names contribute path words too
        tokens += tokenize(f)
    words = [w.lower() for w in re.findall(r"[A-Za-z_]\w*", q_nofiles)]
    libs = [w for w in words if w in LIBRARIES]
    actions = [w for w in words if stem(w) in _CONCEPT_STEMS]
    expansion: list[str] = []
    if expand:
        have = set(tokens)
        for a in actions:
            for rel in CONCEPTS[_CONCEPT_STEMS[stem(a)]]:
                for t in tokenize(rel):
                    if t not in have and t not in expansion:
                        expansion.append(t)
        expansion = expansion[:16]
    lang_m = LANG_RE.search(q)
    lang = {"golang": "go", "c++": "cpp", "c#": "csharp"}.get(lang_m.group(1).lower(), lang_m.group(1).lower()) if lang_m else ""
    keywords = [w for w in dict.fromkeys(words) if w not in STOPWORDS and len(w) > 2]
    return QueryInfo(raw=query, text=q, tokens=tokens, expansion=expansion, identifiers=idents,
                     function_names=list(dict.fromkeys(fnames)), class_names=list(dict.fromkeys(cnames)),
                     file_names=[f.lower() for f in files], libraries=libs, keywords=keywords, actions=actions,
                     language=lang, has_error_text=bool(ERR_RE.search(q)))
