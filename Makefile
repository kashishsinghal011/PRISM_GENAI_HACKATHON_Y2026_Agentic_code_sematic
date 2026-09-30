.PHONY: install test demo bench ablation eval-local eval-mteb ui
install: ; pip install -r requirements.txt && pip install -e .
test: ; python -m pytest -q
demo: ; python scripts/demo_queries.py
bench: ; python scripts/benchmark.py
ablation: ; python scripts/make_local_benchmark.py && python scripts/ablation.py
eval-local: ; python scripts/evaluate.py --mode local
eval-mteb: ; python scripts/evaluate.py --mode mteb --task AppsRetrieval
ui: ; streamlit run demo/streamlit_app.py
