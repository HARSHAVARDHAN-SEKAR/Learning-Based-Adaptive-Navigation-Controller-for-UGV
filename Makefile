# One-command reproducibility. This IS the "reproducible" claim — a
# reviewer or recruiter should need nothing beyond `make bench`.

.PHONY: install test bench report diagram docker-build docker-run clean all

install:
	pip install -r requirements.txt

test:
	python3 tests/test_all_modules.py

bench: test
	python3 benchmarks/run_estimation.py
	python3 benchmarks/run_planners.py
	python3 benchmarks/run_full_flow.py
	python3 benchmarks/run_robustness.py

live:
	python3 benchmarks/live_simulation.py --controller $(or $(CONTROLLER),mpc)

report: bench
	python3 benchmarks/make_report.py
	python3 benchmarks/make_diagram.py

docker-build:
	docker build -t ugv-nav-research .

docker-run:
	docker run --rm \
		-v "$$(pwd)/benchmarks:/app/benchmarks" \
		ugv-nav-research

all: install report

clean:
	rm -rf benchmarks/plots/*.png benchmarks/*.csv paper_report.pdf
	find . -name "__pycache__" -exec rm -rf {} +
