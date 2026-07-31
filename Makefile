PY := .venv/bin/python

.PHONY: all index insider fund panel features paper test clean

all: index insider fund panel features

index:    ; $(PY) scripts/build_dataset.py index
insider:  ; $(PY) scripts/build_dataset.py insider
fund:     ; $(PY) scripts/reload_fund.py
panel:    ; $(PY) scripts/build_dataset.py panel
features: ; $(PY) scripts/make_features.py

paper:
	$(PY) scripts/make_charts.py

test:
	$(PY) -m pytest -q

# Drops derived data but keeps data/raw, so a rebuild does not re-download.
clean:
	rm -f data/*.duckdb data/features.parquet
