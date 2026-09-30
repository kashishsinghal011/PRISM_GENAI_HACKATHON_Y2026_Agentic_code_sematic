import pytest
from aci.errors import UnsupportedLanguageError
from aci.ingestion.chunker import chunk_file, lexical_text, represent
from aci.ingestion.parser import detect_language

SRC = '''import os
from x import y

def normalize(s):
    """Normalize."""
    return forward(s.strip())

class A:
    """Thing."""
    def m(self, q):
        return normalize(q)
'''


def test_function_method_class_chunks_with_lines():
    ch = {c.qualname: c for c in chunk_file(SRC, "pre.py", "repo", commit="abc", version="v1")}
    assert {"normalize", "A", "A.m"} <= set(ch)
    n = ch["normalize"]
    assert (n.start_line, n.end_line) == (4, 6) and sorted(n.calls) == ["forward", "strip"]
    assert n.doc == "Normalize." and n.language == "python" and n.commit == "abc" and n.version == "v1"
    assert ch["A.m"].kind == "method" and ch["A.m"].class_name == "A"
    assert "os" in n.imports


def test_large_function_split_keeps_parent_metadata():
    body = "\n".join(f"    x{i} = {i}" for i in range(40))
    src = f"def big(a):\n{body}\n    return a\n"
    parts = chunk_file(src, "big.py", "repo", max_chunk_lines=15)
    assert len(parts) >= 3 and all(p.parent == "big" and p.function == "big" and p.kind == "block" for p in parts)
    assert [p.part for p in parts] == list(range(len(parts)))


def test_script_style_code_becomes_module_chunk():
    parts = chunk_file("n = int(input())\nprint(n * 2)\n", "sol.py", "repo")
    assert len(parts) == 1 and parts[0].kind == "module"


def test_syntax_error_falls_back_to_line_windows():
    parts = chunk_file("def broken(:\n  pass\n" * 3, "bad.py", "repo")
    assert parts and parts[0].kind == "file"


def test_javascript_functions_and_classes():
    js = "function foo(a, b) {\n  return bar(a) + b;\n}\nclass K {\n  run(x) {\n    return foo(x, 1);\n  }\n}\n"
    q = {c.qualname: c for c in chunk_file(js, "a.js", "repo")}
    assert "foo" in q and "K.run" in q and q["foo"].calls == ["bar"]


def test_large_file_skipped_and_unsupported_language():
    assert chunk_file("x = 1\n" * 100, "big.py", "repo", max_file_bytes=10) == []
    with pytest.raises(UnsupportedLanguageError):
        detect_language("README.md")


def test_representations_differ_and_unknown_mode_errors():
    c = chunk_file(SRC, "pre.py", "repo")[0]
    raw, meta, ctx = represent(c, "raw"), represent(c, "metadata"), represent(c, "contextual")
    assert raw == c.code and "File: pre.py" in ctx and "Function: normalize" in ctx and "normalize" in meta and ctx != raw
    assert "normalize" in lexical_text(c)
    with pytest.raises(ValueError):
        represent(c, "nope")


def test_duplicate_chunk_hash_is_stable():
    a = chunk_file(SRC, "pre.py", "repo")[0]
    b = chunk_file(SRC.replace("    ", "        ", 0), "pre.py", "repo")[0]
    assert a.chash == b.chash and a.cache_key == b.cache_key
