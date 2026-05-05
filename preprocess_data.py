"""
Preprocess raw benchmark datasets into the compact parquet files consumed by
the pipeline.

Reads raw Adult, COMPAS and Bank Marketing from ``data/`` and writes:

    data/adult_processed.parquet
    data/compas_processed.parquet
    data/bm_processed.parquet

Preprocessing follows standard steps from the fairness literature and is
summarised in Section 4.1 of the paper. See the per-dataset blocks below for
exact categorisation rules.

Usage:
    python preprocess_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


# ──────────────────────────────────────────────────────────────────────────
# Adult
# ──────────────────────────────────────────────────────────────────────────

def preprocess_adult() -> None:
    print("[adult] preprocessing ...")
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
    train = pd.read_csv(DATA_DIR / "adult.data", names=cols, na_values="?",
                        skipinitialspace=True)
    test = pd.read_csv(DATA_DIR / "adult.test", names=cols, na_values="?",
                       skipinitialspace=True)
    data = pd.concat([train, test], ignore_index=True)

    del data["fnlwgt"]
    data = data.dropna()

    data["age_cat"] = pd.cut(
        data["age"].astype(int),
        bins=[-np.inf, 25, 60, np.inf],
        labels=["<25", "25-60", ">60"], right=False,
    )
    data["capital_gain_cat"] = pd.cut(
        data["capital-gain"].astype(int),
        bins=[-np.inf, 5000, np.inf],
        labels=["≤5000", ">5000"],
    )
    data["hours_per_week_cat"] = pd.cut(
        data["hours-per-week"].astype(int),
        bins=[-np.inf, 40, 60, np.inf],
        labels=["<40", "40-60", ">60"], right=False,
    )
    data["workclass_cat"] = data["workclass"].apply(
        lambda x: "private" if x == "Private" else "non-private")
    high_education = {"Bachelors", "Some-college", "Masters",
                      "Doctorate", "Prof-school"}
    data["education_cat"] = data["education"].apply(
        lambda x: "high" if x in high_education else "low")
    married_status = {"Married-civ-spouse", "Married-spouse-absent",
                      "Married-AF-spouse"}
    data["marital_status_cat"] = data["marital-status"].apply(
        lambda x: "married" if x in married_status else "other")
    data["native_country_cat"] = data["native-country"].apply(
        lambda x: "US" if x == "United-States" else "non-US")
    data["race_cat"] = data["race"].apply(
        lambda x: "white" if x == "White" else "non-white")
    office_work = {"Adm-clerical", "Exec-managerial", "Prof-specialty",
                   "Sales", "Tech-support"}
    heavy_work = {"Craft-repair", "Farming-fishing", "Handlers-cleaners",
                  "Machine-op-inspct", "Transport-moving"}
    data["occupation_cat"] = data["occupation"].apply(
        lambda x: "office" if x in office_work
        else ("heavy-work" if x in heavy_work else "other"))

    for c in ("age_cat", "capital_gain_cat", "hours_per_week_cat",
              "occupation_cat"):
        data[c] = data[c].astype("object")

    data = data.rename(columns={"income": "target"})
    data["target"] = (data["target"] == ">50K").astype(int)

    keep = ["target", "sex", "age_cat", "capital_gain_cat",
            "hours_per_week_cat", "workclass_cat", "education_cat",
            "marital_status_cat", "native_country_cat", "race_cat",
            "occupation_cat"]
    data = data[keep]

    out = DATA_DIR / "adult_processed.parquet"
    data.to_parquet(out)
    print(f"[adult] wrote {out} ({len(data):,} rows)")


# ──────────────────────────────────────────────────────────────────────────
# COMPAS
# ──────────────────────────────────────────────────────────────────────────

def _bin_count(x: int) -> str:
    if x == 0:
        return "0"
    if x < 5:
        return "1-5"
    return ">5"


def preprocess_compas() -> None:
    print("[compas] preprocessing ...")
    raw = pd.read_csv(DATA_DIR / "compas-scores-two-years.csv")
    cols = ["sex", "race", "age_cat", "c_charge_desc", "c_charge_degree",
            "priors_count", "juv_fel_count", "juv_misd_count",
            "two_year_recid", "score_text", "decile_score",
            "days_b_screening_arrest", "is_recid"]
    raw = raw[cols].copy()

    mask = (
        (raw["days_b_screening_arrest"] <= 30)
        & (raw["days_b_screening_arrest"] >= -30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
        & (raw["c_charge_desc"] != "arrest case no charge")
    )
    raw = raw[mask].reset_index(drop=True)

    freq = raw["c_charge_desc"].value_counts()
    rare = set(freq[freq < 10].index)
    raw["c_charge_desc"] = raw["c_charge_desc"].apply(
        lambda x: "other" if x in rare else x).fillna("other")

    raw["priors"] = raw["priors_count"].apply(_bin_count)
    raw["juv_fel"] = raw["juv_fel_count"].apply(_bin_count)
    raw["juv_misd"] = raw["juv_misd_count"].apply(_bin_count)

    raw = raw.rename(columns={"two_year_recid": "target"})
    raw["race_cat"] = raw["race"].apply(
        lambda x: "caucasian" if x == "Caucasian" else "non-caucasian")

    keep = ["target", "sex", "race_cat", "age_cat", "c_charge_desc",
            "c_charge_degree", "priors", "juv_fel", "juv_misd"]
    data = raw[keep].copy()

    out = DATA_DIR / "compas_processed.parquet"
    data.to_parquet(out)
    print(f"[compas] wrote {out} ({len(data):,} rows)")


# ──────────────────────────────────────────────────────────────────────────
# Bank Marketing
# ──────────────────────────────────────────────────────────────────────────

def preprocess_bm() -> None:
    print("[bm] preprocessing ...")
    data = pd.read_csv(DATA_DIR / "bank-full.csv", delimiter=";")

    def _job(job: str) -> str:
        if job == "blue-collar":
            return "blue-collar"
        if job in ("management", "services"):
            return "management-service"
        return "other"

    data["job_cat"] = data["job"].apply(_job)
    data["balance_cat"] = pd.cut(
        data["balance"], bins=[-np.inf, 0, np.inf],
        labels=["0", ">0"], right=False)
    data["day_cat"] = pd.cut(
        data["day"], bins=[-np.inf, 15, np.inf],
        labels=["≤15", ">15"], right=False)
    data["duration_cat"] = pd.cut(
        data["duration"], bins=[-np.inf, 120, 600, np.inf],
        labels=["≤120", "121–600", ">600"], right=False)
    data["campaign_cat"] = pd.cut(
        data["campaign"], bins=[-np.inf, 1, 5, np.inf],
        labels=["≤1", "2–5", ">5"], right=False)
    data["pdays_cat"] = pd.cut(
        data["pdays"], bins=[-np.inf, 30, 180, np.inf],
        labels=["≤30", "31–180", ">180"], right=False)
    data["previous_cat"] = pd.cut(
        data["previous"], bins=[-np.inf, 0, 5, np.inf],
        labels=["0", "1–5", ">5"], right=False)
    data["age_cat"] = pd.cut(
        data["age"], bins=[-np.inf, 25, 60, np.inf],
        labels=["<25", "25-60", ">60"], right=False)
    data["age_cat"] = data["age_cat"].map(
        {"<25": "<25 or >60", ">60": "<25 or >60", "25-60": "25-60"})
    data["target"] = data["y"].map({"yes": 1, "no": 0})

    data = data.astype("object")
    keep = ["target", "age_cat", "marital", "education", "housing",
            "contact", "month", "duration_cat", "pdays_cat", "poutcome"]
    data = data[keep]

    out = DATA_DIR / "bm_processed.parquet"
    data.to_parquet(out)
    print(f"[bm] wrote {out} ({len(data):,} rows)")


# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    preprocess_adult()
    preprocess_compas()
    preprocess_bm()
    print("\n✓ preprocess_data done.")


if __name__ == "__main__":
    main()
