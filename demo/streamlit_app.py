"""Streamlit demo:  streamlit run demo/streamlit_app.py"""
import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aci.config import load_config  # noqa: E402
from aci.engine import CodeSearchEngine  # noqa: E402
from aci.errors import ACIError  # noqa: E402
from aci.query.preprocess import preprocess_query  # noqa: E402

st.set_page_config(page_title="Agentic Code Intelligence", layout="wide")
st.title("Agentic Code Intelligence: semantic code retrieval")
cfg = load_config()
idx_dir = Path(cfg["index"]["dir"])
repos = sorted(p.name for p in idx_dir.iterdir() if (p / "refs.json").exists()) if idx_dir.exists() else []
if not repos:
    st.warning("No indexed repositories. Run `python scripts/build_index.py --repo <path>` first.")
    st.stop()

with st.sidebar:
    repo = st.selectbox("Repository", repos)
    engine = st.cache_resource(lambda r: CodeSearchEngine(r, cfg))(repo)
    labels = [v["label"] for v in engine.versions()]
    version = st.selectbox("Version", ["latest", *labels, "all (cross-version)"])
    top_k = st.slider("Top-K", 1, 30, 10)
    history = st.checkbox("Show every version (no collapsing)", value=False, disabled=not version.startswith("all"))

query = st.text_input("Natural-language query", "Where is the input normalized before reaching the main function?")
if st.button("Search", type="primary") and query:
    try:
        ver = "all" if version.startswith("all") else (None if version == "latest" else version)
        results = engine.search(query, version=ver, top_k=top_k, history=history)
    except (ACIError, ValueError) as e:
        st.error(str(e))
        st.stop()
    qi = preprocess_query(query)
    terms = sorted({t for t in re.findall(r"[A-Za-z_]\w+", " ".join([query, *qi.identifiers])) if len(t) > 3})
    st.caption(f"{len(results)} result(s) - {engine.last_timings.get('total_ms', 0):.1f} ms - entities: {', '.join(qi.entity_names) or 'none'}")
    for r in results:
        c = r.chunk
        with st.container(border=True):
            st.markdown(f"**{r.rank}. `{c.file}`** &nbsp; `{c.qualname}` &nbsp; lines {c.start_line}-{c.end_line} &nbsp; score **{r.score:.3f}**")
            st.caption(f"version: {c.version or c.commit[:8]} - retrieved via: {r.method} - {r.explanation}"
                       + (f" - also in: {', '.join(r.also_in)}" if r.also_in else ""))
            code = html.escape(c.code)
            for t in terms:
                code = re.sub(f"(?i)({re.escape(html.escape(t))})", r"<mark>\1</mark>", code)
            st.markdown(f"<pre style='white-space:pre-wrap;font-size:0.85em'>{code}</pre>", unsafe_allow_html=True)
            with st.expander("ranking features"):
                st.json({k: round(v, 3) for k, v in r.features.items()})
