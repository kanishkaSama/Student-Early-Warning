"""Generate a SMALL SYNTHETIC dataset with the same files and columns as OULAD.

!!! The numbers this produces are fake. !!!
Use it only to test that the code runs (dry runs and automated tests).
Never report results computed on this data as project results.

Usage:
    python -m src.make_sample_data                  # writes to data/sample
    python -m src.make_sample_data --students 40 --out /tmp/oulad_sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.logging_utils import get_logger

logger = get_logger("make_sample_data")

MODULES = ["AAA", "BBB", "CCC"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
REGIONS = ["Scotland", "London Region", "Wales", "North Region", "South East Region"]
EDUCATION = ["A Level or Equivalent", "Lower Than A Level", "HE Qualification",
             "No Formal quals", "Post Graduate Qualification"]
IMD_BANDS = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%",
             "60-70%", "70-80%", "80-90%", "90-100%"]
ACTIVITY_TYPES = ["resource", "oucontent", "forumng", "quiz", "homepage", "url", "subpage"]
TMA_DAYS = [19, 54, 96, 131, 166, 208]


def generate(students_per_presentation: int = 120, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    courses, assessments, vle = [], [], []
    info, registration, submissions, clicks = [], [], [], []
    next_assessment_id, next_site_id, next_student_id = 1000, 500000, 10000

    for module in MODULES:
        for presentation in PRESENTATIONS:
            length = int(rng.choice([241, 262, 268, 269]))
            courses.append((module, presentation, length))

            assessment_ids = []
            for day in TMA_DAYS:
                assessments.append((module, presentation, next_assessment_id, "TMA",
                                    day, 100 / len(TMA_DAYS)))
                assessment_ids.append((next_assessment_id, day))
                next_assessment_id += 1
            # one exam with a missing date, like the real data
            assessments.append((module, presentation, next_assessment_id, "Exam", None, 100.0))
            next_assessment_id += 1

            sites = []
            for _ in range(25):
                vle.append((next_site_id, module, presentation,
                            rng.choice(ACTIVITY_TYPES), None, None))
                sites.append(next_site_id)
                next_site_id += 1
            sites = np.array(sites)

            for _ in range(students_per_presentation):
                student_id = next_student_id
                next_student_id += 1
                engagement = rng.normal()          # hidden "how engaged" score
                prev_attempts = int(rng.poisson(0.2))
                risk_logit = -0.6 - 1.6 * engagement + 0.5 * prev_attempts
                at_risk = rng.random() < 1 / (1 + np.exp(-risk_logit))

                if at_risk:
                    withdrawn = rng.random() < 0.55
                    result = "Withdrawn" if withdrawn else "Fail"
                    unregister_day = int(rng.integers(-20, length - 30)) if withdrawn else None
                else:
                    result = "Distinction" if engagement > 1.2 else "Pass"
                    unregister_day = None

                imd = None if rng.random() < 0.05 else str(rng.choice(IMD_BANDS))
                info.append((module, presentation, student_id, rng.choice(["M", "F"]),
                             rng.choice(REGIONS), rng.choice(EDUCATION), imd,
                             rng.choice(["0-35", "35-55", "55<="], p=[0.7, 0.28, 0.02]),
                             prev_attempts, int(rng.choice([30, 60, 90, 120])),
                             rng.choice(["N", "Y"], p=[0.9, 0.1]), result))
                reg_day = None if rng.random() < 0.01 else int(rng.integers(-150, 5))
                registration.append((module, presentation, student_id, reg_day, unregister_day))

                last_day = unregister_day if unregister_day is not None else length
                # engagement fades over time for students who will struggle
                days = np.arange(-10, last_day)
                fade = np.where(at_risk, np.clip(1 - days / 150, 0.15, 1), 1.0)
                p_active = np.clip(0.30 + 0.12 * engagement, 0.03, 0.85) * fade
                active_days = days[rng.random(len(days)) < p_active]
                for day in active_days:
                    for site in rng.choice(sites, size=int(rng.integers(1, 4)), replace=False):
                        clicks.append((module, presentation, student_id, int(site), int(day),
                                       int(rng.poisson(3) + 1)))

                for assessment_id, due in assessment_ids:
                    if due > last_day or rng.random() < (0.35 if at_risk else 0.05):
                        continue  # not submitted
                    late = int(rng.integers(1, 10)) if rng.random() < (0.3 if at_risk else 0.08) else 0
                    score = float(np.clip(rng.normal(68 + 12 * engagement, 12), 0, 100).round())
                    score_value = None if rng.random() < 0.01 else score
                    submissions.append((assessment_id, student_id, due - int(rng.integers(0, 5)) + late,
                                        0, score_value))

    return {
        "courses": pd.DataFrame(courses, columns=["code_module", "code_presentation",
                                                  "module_presentation_length"]),
        "assessments": pd.DataFrame(assessments, columns=["code_module", "code_presentation",
                                                          "id_assessment", "assessment_type",
                                                          "date", "weight"]),
        "vle": pd.DataFrame(vle, columns=["id_site", "code_module", "code_presentation",
                                          "activity_type", "week_from", "week_to"]),
        "studentInfo": pd.DataFrame(info, columns=["code_module", "code_presentation",
                                                   "id_student", "gender", "region",
                                                   "highest_education", "imd_band", "age_band",
                                                   "num_of_prev_attempts", "studied_credits",
                                                   "disability", "final_result"]),
        "studentRegistration": pd.DataFrame(registration, columns=[
            "code_module", "code_presentation", "id_student", "date_registration",
            "date_unregistration"]),
        "studentAssessment": pd.DataFrame(submissions, columns=[
            "id_assessment", "id_student", "date_submitted", "is_banked", "score"]),
        "studentVle": pd.DataFrame(clicks, columns=["code_module", "code_presentation",
                                                    "id_student", "id_site", "date",
                                                    "sum_click"]),
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        # Like the real files: missing values are written as "?"
        df.to_csv(out_dir / f"{name}.csv", index=False, na_rep="?")
    logger.info("Wrote SYNTHETIC sample (%s) to %s",
                ", ".join(f"{k}={len(v):,}" for k, v in tables.items()), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic OULAD-shaped sample.")
    parser.add_argument("--students", type=int, default=120,
                        help="students per course presentation (default 120)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out_dir = args.out or load_config()["paths"]["sample_data_dir"]
    write_tables(generate(args.students, args.seed), out_dir)


if __name__ == "__main__":
    main()
