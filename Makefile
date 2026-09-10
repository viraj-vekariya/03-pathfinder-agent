# Pathfinder Agent
#
#   make setup     install dependencies
#   make graph     fetch the real dependency graph from PyPI
#   make test      run the suite
#   make sweep     the depth sweep (the core measurement)
#   make explain   attribute the failures to a mechanism
#   make report    consolidate into outputs/results.json
#   make serve     the comparison UI on :8300

PY ?= python3

.PHONY: setup graph test sweep explain report serve docker clean all

setup:
	$(PY) -m pip install -r requirements.txt

graph:
	$(PY) -m graph.build --max-nodes 600

test:
	$(PY) -m pytest tests/ -q

sweep:
	$(PY) -m eval.depth_sweep --per-depth 8 --backend flan-t5

explain:
	$(PY) -m eval.truncation

report:
	$(PY) -m eval.report

serve:
	$(PY) -m uvicorn api.app:app --host 127.0.0.1 --port 8300

docker:
	docker compose up --build

all: graph test sweep explain report

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
