#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_METHOD_ORDER = ["GCP", "CCCP", "SCCP"]

def _safe_get(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return default

def _fmt(x: float, nd: int = 3) -> str:
    # LaTeX-friendly numeric formatting
    if x != x:  # NaN
        return "--"
    return f"{x:.{nd}f}"

def extract_row(method: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the metrics we care about from the saved JSON.
    We assume the JSON structure like:
      - top-level has "GCP"/"CCCP"/"SCCP" dicts produced by eval_sets()
      - each has keys: coverage, avg_size, cov_tail, size_tail, cov_head, size_head,
                       avg_class_cov, worst_class_cov, std_class_cov,
                       avg_cluster_cov, worst_cluster_cov, std_cluster_cov
    """
    e = blob.get(method, {})
    return {
        "method": method,
        "cov": _safe_get(e, "coverage"),
        "size": _safe_get(e, "avg_size"),
        "covT": _safe_get(e, "cov_tail"),
        "szT": _safe_get(e, "size_tail"),
        "covH": _safe_get(e, "cov_head"),
        "szH": _safe_get(e, "size_head"),
        "avg_cls": _safe_get(e, "avg_class_cov"),
        "worst_cls": _safe_get(e, "worst_class_cov"),
        "std_cls": _safe_get(e, "std_class_cov"),
        "avg_clu": _safe_get(e, "avg_cluster_cov"),
        "worst_clu": _safe_get(e, "worst_cluster_cov"),
        "std_clu": _safe_get(e, "std_cluster_cov"),
    }

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def latex_escape(s: str) -> str:
    # minimal; method names typically safe
    return s.replace("_", r"\_")

def make_table(
    rows_by_block: List[Dict[str, Any]],
    caption: Optional[str],
    label: Optional[str],
    nd_cov: int = 3,
    nd_size: int = 1,
) -> str:
    """
    rows_by_block: list of blocks, each block is dict:
      {
        "title": "SCCP (tau=5)",
        "rows": [ {method,...}, ... ]
      }
    We produce a single LaTeX table with blocks separated by \midrule.
    """
    # Column spec: l + 12 numeric columns
    colspec = "l" + "c" * 12
    header = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\begin{{tabular}}{{{colspec}}}" "\n"
        r"\toprule" "\n"
        r"Method & cov & size & cov$_T$ & sz$_T$ & cov$_H$ & sz$_H$ & "
        r"avg\_cls & worst\_cls & std\_cls & avg\_clu & worst\_clu & std\_clu \\" "\n"
        r"\midrule" "\n"
    )

    body_lines: List[str] = []
    for bi, block in enumerate(rows_by_block):
        title = block.get("title")
        rows = block["rows"]

        if title:
            # A block title row spanning all columns
            body_lines.append(
                rf"\multicolumn{{13}}{{l}}{{\textbf{{{latex_escape(title)}}}}} \\"
            )

        for r in rows:
            body_lines.append(
                f"{latex_escape(r['method'])} & "
                f"{_fmt(r['cov'], nd_cov)} & {_fmt(r['size'], nd_size)} & "
                f"{_fmt(r['covT'], nd_cov)} & {_fmt(r['szT'], nd_size)} & "
                f"{_fmt(r['covH'], nd_cov)} & {_fmt(r['szH'], nd_size)} & "
                f"{_fmt(r['avg_cls'], nd_cov)} & {_fmt(r['worst_cls'], nd_cov)} & {_fmt(r['std_cls'], nd_cov)} & "
                f"{_fmt(r['avg_clu'], nd_cov)} & {_fmt(r['worst_clu'], nd_cov)} & {_fmt(r['std_clu'], nd_cov)} \\\\"
            )

        if bi != len(rows_by_block) - 1:
            body_lines.append(r"\midrule")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    if caption:
        footer += rf"\caption{{{caption}}}" "\n"
    if label:
        footer += rf"\label{{{label}}}" "\n"
    footer += r"\end{table}" "\n"

    return header + "\n".join(body_lines) + "\n" + footer

def infer_block_title(blob: Dict[str, Any], fallback: str) -> str:
    # Prefer something like "SCCP (tau=5, Kc=10)" if present
    tau = blob.get("tau", None)
    kc = blob.get("clusters", None)
    parts = []
    if tau is not None:
        parts.append(f"tau={tau}")
    if kc is not None:
        parts.append(f"Kc={kc}")
    if parts:
        return f"{fallback} ({', '.join(parts)})"
    return fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="One or more result JSON files saved by run_cp_from_npz.py --out")
    ap.add_argument("--methods", type=str, default="GCP,CCCP,SCCP", help="Comma-separated method order")
    ap.add_argument("--caption", type=str, default=None)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--out_tex", type=str, default=None, help="Write LaTeX to this file (or print to stdout)")
    ap.add_argument("--single_block", action="store_true",
                    help="If set, merge all JSONs into ONE block (no block titles / midrules). "
                         "Useful when JSON already corresponds to same setting.")
    ap.add_argument("--block_title_prefix", type=str, default="Results",
                    help="Prefix for each block title when multiple JSONs are provided.")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    json_paths = [Path(p) for p in args.json]

    blocks = []
    if args.single_block:
        # Only use the first JSON to populate the table
        blob = load_json(json_paths[0])
        rows = [extract_row(m, blob) for m in methods]
        blocks.append({"title": None, "rows": rows})
    else:
        for p in json_paths:
            blob = load_json(p)
            title = infer_block_title(blob, args.block_title_prefix)
            rows = [extract_row(m, blob) for m in methods]
            blocks.append({"title": title, "rows": rows})

    tex = make_table(blocks, caption=args.caption, label=args.label)

    if args.out_tex:
        Path(args.out_tex).write_text(tex, encoding="utf-8")
        print(f"[saved] {args.out_tex}")
    else:
        print(tex)

if __name__ == "__main__":
    main()
