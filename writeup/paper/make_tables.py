from __future__ import annotations

from pathlib import Path
import csv
import math
from typing import Iterable

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def format_num(x: str, digits: int = 3) -> str:
    try:
        val = float(x)
    except ValueError:
        return x
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def write_latex_table(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l r l r}\n")
        f.write("\\toprule\n")
        f.write("{}\\\\\n".format(" & ".join(header)))
        f.write("\\midrule\n")
        for row in rows:
            f.write("{}\\\\\n".format(" & ".join(row)))
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def build_overlap_table(out_csv: Path, out_tex: Path, n: int = 50) -> None:
    root = repo_root()
    vocab_path = root / "data" / "raw" / "openicpsr_wu2018_replication-pkg" / "vocab10K.csv"
    full_dir = root / "output" / "intermediate" / "lasso_full"
    pronoun_dir = root / "output" / "intermediate" / "lasso_pronoun"

    vocab = []
    with vocab_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab.append(row["word"])

    def load_words_and_coef(coef_path: Path, keep_path: Path) -> tuple[list[str], np.ndarray]:
        coef = np.loadtxt(coef_path)
        keep = np.loadtxt(keep_path).astype(int)
        words = [vocab[idx] for idx in keep]
        return words, coef

    words_full, coef_full = load_words_and_coef(
        full_dir / "coef_lasso_logit_full.txt", full_dir / "i_keep_columns.txt"
    )
    words_pronoun, coef_pronoun = load_words_and_coef(
        pronoun_dir / "coef_lasso_logit_pronoun.txt", pronoun_dir / "i_keep_columns.txt"
    )

    def top_words(words: list[str], coef: np.ndarray, direction: str) -> list[str]:
        if direction == "pos":
            idx = np.argsort(coef)[-n:]
        elif direction == "neg":
            idx = np.argsort(coef)[:n]
        else:
            raise ValueError("direction must be 'pos' or 'neg'")
        return [words[i] for i in idx]

    pos_full = set(top_words(words_full, coef_full, "pos"))
    neg_full = set(top_words(words_full, coef_full, "neg"))
    pos_pronoun = set(top_words(words_pronoun, coef_pronoun, "pos"))
    neg_pronoun = set(top_words(words_pronoun, coef_pronoun, "neg"))

    def overlap_stats(a: set[str], b: set[str]) -> tuple[int, int, float, list[str]]:
        inter = sorted(a & b)
        union = a | b
        jaccard = (len(inter) / len(union)) if union else 0.0
        return len(inter), len(union), jaccard, inter

    pos_overlap, pos_union, pos_jaccard, pos_words = overlap_stats(pos_full, pos_pronoun)
    neg_overlap, neg_union, neg_jaccard, neg_words = overlap_stats(neg_full, neg_pronoun)

    rows = [
        {
            "group": "Female-associated (pos.)",
            "n": n,
            "overlap": pos_overlap,
            "union": pos_union,
            "jaccard": round(pos_jaccard, 3),
            "examples": ", ".join(pos_words[:4]),
        },
        {
            "group": "Male-associated (neg.)",
            "n": n,
            "overlap": neg_overlap,
            "union": neg_union,
            "jaccard": round(neg_jaccard, 3),
            "examples": ", ".join(neg_words[:4]),
        },
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l r r r p{7cm}}\n")
        f.write("\\toprule\n")
        f.write("Group & N & Overlap & Jaccard & Example overlap\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                "{} & {} & {} & {} & {}\\\\\n".format(
                    latex_escape(row["group"]),
                    row["n"],
                    row["overlap"],
                    row["jaccard"],
                    latex_escape(row["examples"]),
                )
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main() -> None:
    root = repo_root()
    tables_dir = Path(__file__).resolve().parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1
    header1, rows1 = read_csv_rows(root / "output" / "tables" / "table1.csv")
    table1_rows = []
    for row in rows1:
        table1_rows.append(
            [
                latex_escape(row[0]),
                format_num(row[1]),
                latex_escape(row[2]),
                format_num(row[3]),
            ]
        )
    write_latex_table(
        tables_dir / "table1.tex",
        ["Female word", "ME", "Male word", "ME"],
        table1_rows,
    )

    # Table 2
    header2, rows2 = read_csv_rows(root / "output" / "tables" / "table2.csv")
    table2_rows = []
    for row in rows2:
        table2_rows.append(
            [
                latex_escape(row[0]),
                format_num(row[1]),
                latex_escape(row[2]),
                format_num(row[3]),
            ]
        )
    write_latex_table(
        tables_dir / "table2.tex",
        ["Female word", "ME (pronoun)", "Male word", "ME (pronoun)"],
        table2_rows,
    )

    # Extension overlap table
    build_overlap_table(
        root / "output" / "tables" / "overlap_full_vs_pronoun.csv",
        tables_dir / "table_overlap.tex",
        n=50,
    )


if __name__ == "__main__":
    main()
