"""Language-aware parsing into retrieval units (functions, methods, classes, blocks).

Python uses the real `ast`. Brace languages (JS/TS/Java/Go/C/C++/C#/Rust/PHP/Kotlin/Scala) use a
lightweight brace-matching parser: less precise than tree-sitter but dependency-free and CPU-cheap.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import PurePosixPath

from ..errors import UnsupportedLanguageError
from ..models import Chunk

log = logging.getLogger(__name__)

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rs": "rust", ".kt": "kotlin", ".php": "php", ".scala": "scala",
}
BRACE_LANGS = set(EXT_LANG.values()) - {"python"}
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "else", "sizeof", "synchronized", "do",
             "try", "new", "throw", "elif", "foreach", "using", "lock", "typeof", "await", "delete"}


def detect_language(path: str) -> str:
    lang = EXT_LANG.get(PurePosixPath(path).suffix.lower())
    if not lang:
        raise UnsupportedLanguageError(f"Unsupported file type: {path}")
    return lang


# ------------------------------------------------------------------ python
def _callee_names(node: ast.AST) -> list[str]:
    seen: dict[str, None] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name:
                seen.setdefault(name, None)
    return list(seen)


def _import_names(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.module or "") + "." + ",".join(a.name for a in n.names) if n.module else ",".join(a.name for a in n.names))
    return out[:40]


def _start(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decos])


def _signature(node: ast.AST, lines: list[str]) -> str:
    body0 = node.body[0].lineno if node.body else node.lineno + 1
    sig = " ".join(l.strip() for l in lines[node.lineno - 1: max(node.lineno, body0 - 1)])
    return sig if sig else lines[node.lineno - 1].strip()


def _mk(base: dict, code: str, start: int, end: int, **kw) -> Chunk:
    return Chunk(code=code, start_line=start, end_line=end, **base, **kw)


def _split_body(stmts: list[ast.stmt], max_lines: int) -> list[list[ast.stmt]]:
    groups: list[list[ast.stmt]] = []
    cur: list[ast.stmt] = []
    cur_lines = 0
    for s in stmts:
        n = (s.end_lineno or s.lineno) - _start(s) + 1
        if cur and cur_lines + n > max_lines:
            groups.append(cur)
            cur, cur_lines = [], 0
        cur.append(s)
        cur_lines += n
    if cur:
        groups.append(cur)
    return groups


def _function_chunks(node, lines, base, class_name, imports, max_lines) -> list[Chunk]:
    start, end = _start(node), node.end_lineno or node.lineno
    sig = _signature(node, lines)
    doc = ast.get_docstring(node) or ""
    kind = "method" if class_name else "function"
    common = dict(class_name=class_name, function=node.name, signature=sig, doc=doc, imports=imports)
    if end - start + 1 <= max_lines:
        return [_mk(base, "\n".join(lines[start - 1:end]), start, end, kind=kind, calls=_callee_names(node), **common)]
    out: list[Chunk] = []
    for i, grp in enumerate(_split_body(node.body, max_lines)):
        s, e = _start(grp[0]), grp[-1].end_lineno or grp[-1].lineno
        if i == 0:
            s = start   # first block keeps decorators/signature/docstring
        calls: dict[str, None] = {}
        for st in grp:
            for c in _callee_names(st):
                calls.setdefault(c, None)
        out.append(_mk(base, "\n".join(lines[s - 1:e]), s, e, kind="block", parent=node.name, part=i,
                       calls=list(calls), **common))
    return out


def parse_python(source: str, path: str, base: dict, max_lines: int = 60) -> list[Chunk]:
    tree = ast.parse(source)
    lines = source.splitlines()
    imports = _import_names(tree)
    chunks: list[Chunk] = []
    module_stmts: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks += _function_chunks(node, lines, base, "", imports, max_lines)
        elif isinstance(node, ast.ClassDef):
            start, end = _start(node), node.end_lineno or node.lineno
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            summary = [lines[node.lineno - 1].strip()] + [f"    {_signature(m, lines)}" for m in methods]
            chunks.append(_mk(base, "\n".join(summary), start, end, kind="class", class_name=node.name,
                              signature=lines[node.lineno - 1].strip(), doc=ast.get_docstring(node) or "",
                              imports=imports, calls=[]))
            for m in methods:
                chunks += _function_chunks(m, lines, base, node.name, imports, max_lines)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue  # module docstring
        else:
            module_stmts.append(node)
    if module_stmts:  # script-style code (e.g. competitive-programming solutions)
        for i, grp in enumerate(_split_body(module_stmts, max_lines)):
            s, e = _start(grp[0]), grp[-1].end_lineno or grp[-1].lineno
            calls: dict[str, None] = {}
            for st in grp:
                for c in _callee_names(st):
                    calls.setdefault(c, None)
            chunks.append(_mk(base, "\n".join(lines[s - 1:e]), s, e, kind="module", part=i, imports=imports, calls=list(calls)))
    return chunks


# ------------------------------------------------------------------ brace languages
_FUNC_LINE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|async|export|default|abstract|override|virtual|inline|extern|unsafe|pub|internal|suspend)\s+)*"
    r"(?:function\*?\s+|func\s+(?:\([^)]*\)\s*)?|fn\s+|fun\s+|def\s+)?"
    r"(?:[\w<>\[\]\*&:,.?]+\s+){0,3}(\w+)\s*\(([^;{}]*)\)\s*(?:->\s*[\w<>\[\]&\s]+|:\s*[\w<>\[\]\s|]+|throws\s+[\w,\s]+|const|override)?\s*\{?\s*$")
_ARROW_LINE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>\s*\{?\s*$")
_CLASS_LINE = re.compile(r"^\s*(?:(?:public|private|protected|static|final|abstract|export|default|pub|data|sealed)\s+)*(?:class|interface|struct|enum|impl|trait|object)\s+(\w+)")


def _match_brace(lines: list[str], start_idx: int) -> int:
    depth, seen = 0, False
    for i in range(start_idx, min(len(lines), start_idx + 2000)):
        line = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//.*', "", lines[i])
        for ch in line:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
                if seen and depth == 0:
                    return i
    return -1


def _leading_comment(lines: list[str], idx: int) -> str:
    out: list[str] = []
    j = idx - 1
    while j >= 0 and idx - j <= 8 and re.match(r"^\s*(//|/\*|\*|#|///)", lines[j]):
        out.append(re.sub(r"^\s*(///?|/\*+|\*+/?|#)\s?", "", lines[j]).strip())
        j -= 1
    return " ".join(reversed(out)).strip()


def parse_brace(source: str, path: str, language: str, base: dict, max_lines: int = 60) -> list[Chunk]:
    lines = source.splitlines()
    imports = [l.strip() for l in lines if re.match(r"^\s*(import|using|#include|require|use)\b", l)][:40]
    classes: list[tuple[int, int, str]] = []
    for i, l in enumerate(lines):
        m = _CLASS_LINE.match(l)
        if m:
            j = i if "{" in l else i + 1
            e = _match_brace(lines, j) if j < len(lines) else -1
            if e >= 0:
                classes.append((i, e, m.group(1)))
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        m = _FUNC_LINE.match(l) or _ARROW_LINE.match(l)
        brace_line = i
        if m and "{" not in l and i + 1 < len(lines) and lines[i + 1].strip().startswith("{"):
            brace_line = i + 1
        elif m and "{" not in l:
            m = None
        if m and m.group(1) not in _KEYWORDS and not _CLASS_LINE.match(l):
            end = _match_brace(lines, brace_line)
            if end >= i:
                name = m.group(1)
                cls = next((c[2] for c in sorted(classes, key=lambda c: c[1] - c[0]) if c[0] < i <= c[1]), "")
                code = "\n".join(lines[i:end + 1])
                calls = [c for c in dict.fromkeys(re.findall(r"\b(\w+)\s*\(", "\n".join(lines[i + 1:end + 1]))) if c not in _KEYWORDS and c != name]
                common = dict(class_name=cls, function=name, signature=l.strip().rstrip("{").strip(), doc=_leading_comment(lines, i), imports=imports)
                if end - i + 1 <= max_lines:
                    chunks.append(_mk(base, code, i + 1, end + 1, kind="method" if cls else "function", calls=calls, **common))
                else:
                    for p, s in enumerate(range(i, end + 1, max_lines)):
                        e = min(s + max_lines - 1, end)
                        chunks.append(_mk(base, "\n".join(lines[s:e + 1]), s + 1, e + 1, kind="block", parent=name, part=p, calls=calls, **common))
                i = end + 1
                continue
        i += 1
    for s, e, name in classes:
        chunks.append(_mk(base, lines[s], s + 1, e + 1, kind="class", class_name=name, signature=lines[s].strip(),
                          doc=_leading_comment(lines, s), imports=imports))
    return chunks


def parse_source(source: str, path: str, base: dict, max_lines: int = 60) -> list[Chunk]:
    """Dispatch by language. Raises UnsupportedLanguageError; falls back to file-level windows on parse errors."""
    language = detect_language(path)
    base = {**base, "file": path, "language": language}
    try:
        if language == "python":
            chunks = parse_python(source, path, base, max_lines)
        else:
            chunks = parse_brace(source, path, language, base, max_lines)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        log.warning("Parse failed for %s (%s); falling back to line windows", path, e)
        chunks = []
    if not chunks and source.strip():  # nothing extracted or corrupt file -> line windows so it stays searchable
        lines = source.splitlines()
        for p, s in enumerate(range(0, len(lines), max_lines)):
            e = min(s + max_lines, len(lines))
            chunks.append(_mk(base, "\n".join(lines[s:e]), s + 1, e, kind="file", part=p))
    return chunks
