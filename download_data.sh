#!/usr/bin/env bash
# Download the three public benchmark datasets used in the paper into data/.
# After running this script, run `python preprocess_data.py` to produce the
# parquet files consumed by the pipeline.

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DATA_DIR"

echo "[adult] downloading from UCI ..."
curl -fsSL -o "$DATA_DIR/adult.data" \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
curl -fsSL -o "$DATA_DIR/adult.test" \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

echo "[compas] downloading from ProPublica ..."
curl -fsSL -o "$DATA_DIR/compas-scores-two-years.csv" \
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

echo "[bank marketing] downloading from UCI ..."
TMP_ZIP="$(mktemp -t bm-XXXXXX.zip)"
curl -fsSL -o "$TMP_ZIP" \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
unzip -p "$TMP_ZIP" bank-full.csv > "$DATA_DIR/bank-full.csv"
rm -f "$TMP_ZIP"

echo
echo "✓ raw files downloaded to $DATA_DIR"
echo "  Next: python preprocess_data.py"
