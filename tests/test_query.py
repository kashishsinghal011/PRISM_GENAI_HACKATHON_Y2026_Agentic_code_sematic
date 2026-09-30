import pytest
from aci.errors import EmptyQueryError
from aci.query.classifier import category_weights, classify_query
from aci.query.preprocess import preprocess_query


def test_empty_query_raises():
    for q in ("", "   ", None):
        with pytest.raises(EmptyQueryError):
            preprocess_query(q)


def test_extracts_identifiers_files_and_entities():
    q = preprocess_query("Where is authenticate_user() called in auth/views.py with UserService and jwt?")
    assert "authenticate_user" in q.identifiers and "authenticate_user" in q.function_names
    assert "auth/views.py" in q.file_names
    assert "userservice" in q.class_names
    assert "jwt" in q.libraries


def test_programming_terms_are_kept():
    q = preprocess_query("Where is the input normalized before being passed to the main function?")
    from aci.tokenize import stem
    assert {stem("input"), "main", stem("function")} <= set(q.tokens)   # stems are applied symmetrically to docs and queries
    assert "main" in q.function_names


def test_concept_expansion_and_toggle():
    on = preprocess_query("where is the input normalized", expand=True)
    off = preprocess_query("where is the input normalized", expand=False)
    assert "sanit" in on.expansion or "clean" in on.expansion
    assert off.expansion == []


@pytest.mark.parametrize("text,cat", [
    ("authenticate_user JWT", "FUNCTION_LOOKUP"),
    ("class UserService", "CLASS_LOOKUP"),
    ("what is in settings.py", "FILE_LOOKUP"),
    ("ValueError: invalid literal for int() raised in parser", "ERROR_MESSAGE"),
    ("Where is the input normalized before being passed to the main function?", "DATA_FLOW"),
    ("Which code handles retry logic for failed API calls?", "CONTROL_FLOW"),
])
def test_classification(text, cat):
    assert classify_query(preprocess_query(text)) == cat


def test_category_weights_shift_and_normalise():
    base = (0.55, 0.30, 0.15)
    f = category_weights(base, "FUNCTION_LOOKUP")
    c = category_weights(base, "CONCEPTUAL_CODE_SEARCH")
    assert abs(sum(f) - 1) < 1e-9 and abs(sum(c) - 1) < 1e-9
    assert f[0] < c[0]            # function lookups trust dense less than conceptual queries
    assert f[2] > c[2]            # ... and identifiers more
