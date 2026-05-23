"""Module 1: Enhanced error feedback for submit_sql failures.

When agent's SQL produces the correct shape but wrong values,
generate a meaningful diff so the agent can diagnose the issue.

Design: pure functions, no LLM dependency, testable with raw data.
"""

from typing import List, Tuple, Optional


def compute_value_diff(
    pred_rows: List[tuple],
    gt_rows: List[tuple],
    column_names: Optional[List[str]] = None,
    max_sample: int = 3,
    ordered: bool = False,
) -> str:
    """Compare predicted vs ground-truth result sets and return a human-readable diff.

    Args:
        pred_rows: Agent's query results (list of tuples).
        gt_rows:   Ground-truth query results (list of tuples).
        column_names: Column headers (from cursor.description).
        max_sample: Max number of mismatched rows to show.
        ordered: Whether row order matters.

    Returns:
        A concise diff string for the agent, or "" if results match.
    """
    if not pred_rows and not gt_rows:
        return ""
    if pred_rows is None:
        pred_rows = []
    if gt_rows is None:
        gt_rows = []

    pred_set = set(pred_rows) if not ordered else None
    gt_set = set(gt_rows) if not ordered else None

    parts = []
    n_pred = len(pred_rows)
    n_gt = len(gt_rows)
    n_cols_pred = len(pred_rows[0]) if pred_rows else 0
    n_cols_gt = len(gt_rows[0]) if gt_rows else 0

    # Column count mismatch
    if n_cols_pred != n_cols_gt:
        parts.append(f"Column count mismatch: expected {n_cols_gt}, got {n_cols_pred}.")
        return " ".join(parts)

    # Row count mismatch
    if n_pred != n_gt:
        parts.append(f"Row count mismatch: expected {n_gt} rows, got {n_pred} rows.")

    headers = column_names or [f"col{i}" for i in range(n_cols_gt)]

    if ordered:
        # Ordered comparison: show first N mismatched positions
        mismatches = []
        for i, (p, g) in enumerate(zip(pred_rows, gt_rows)):
            if p != g:
                mismatches.append((i, p, g))
            if len(mismatches) >= max_sample:
                break
        if mismatches:
            parts.append(f"{len(mismatches)}+ rows differ (showing first {len(mismatches)}):")
            for idx, p, g in mismatches:
                # Find which columns differ
                diffs = []
                for j, (pv, gv) in enumerate(zip(p, g)):
                    if pv != gv:
                        diffs.append(f"{headers[j]}: expected {_fmt(gv)}, got {_fmt(pv)}")
                parts.append(f"  Row {idx}: {'; '.join(diffs)}")
    else:
        # Unordered: show sample of rows in GT but not in pred
        missing = gt_set - pred_set
        extra = pred_set - gt_set

        if missing:
            sample = list(missing)[:max_sample]
            parts.append(f"{len(missing)} expected rows missing from your result (sample):")
            for row in sample:
                parts.append(f"  {_format_row(row, headers)}")

        if extra:
            sample = list(extra)[:max_sample]
            parts.append(f"{len(extra)} unexpected rows in your result (sample):")
            for row in sample:
                parts.append(f"  {_format_row(row, headers)}")

        # If sets are same size but different, highlight column-level pattern
        if missing and extra and len(missing) == len(extra) and len(missing) <= 20:
            col_diff = _detect_column_pattern(list(missing), list(extra), headers)
            if col_diff:
                parts.append(f"Pattern: differences concentrated in column(s): {col_diff}")

    return "\n".join(parts) if parts else ""


def _fmt(val, max_len: int = 30) -> str:
    """Format a single value for display."""
    s = repr(val)
    return s if len(s) <= max_len else s[:max_len] + "..."


def _format_row(row: tuple, headers: List[str]) -> str:
    """Format one row with column names."""
    pairs = [f"{h}={_fmt(v)}" for h, v in zip(headers, row)]
    return "{" + ", ".join(pairs) + "}"


def _detect_column_pattern(
    missing: List[tuple], extra: List[tuple], headers: List[str]
) -> Optional[str]:
    """Check if mismatches are concentrated in specific columns."""
    if not missing or not extra:
        return None

    n_cols = len(missing[0])
    col_diffs = [0] * n_cols

    # For each missing row, find closest extra row and see which cols differ
    for m_row in missing[:10]:
        best_match = None
        best_score = -1
        for e_row in extra:
            score = sum(1 for a, b in zip(m_row, e_row) if a == b)
            if score > best_score:
                best_score = score
                best_match = e_row
        if best_match:
            for j, (a, b) in enumerate(zip(m_row, best_match)):
                if a != b:
                    col_diffs[j] += 1

    # Columns that differ in >50% of compared rows
    threshold = max(1, len(missing[:10]) // 2)
    problem_cols = [headers[j] for j in range(n_cols) if col_diffs[j] >= threshold]
    return ", ".join(problem_cols) if problem_cols else None
