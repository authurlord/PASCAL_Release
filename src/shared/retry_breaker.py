"""Module 2: SQL retry breaker.

Tracks consecutive submit failures per task. When the agent repeats
the same error pattern N times, injects a strategy-switch hint.

Design: stateful tracker, no LLM dependency, testable standalone.
"""

from collections import defaultdict
from typing import Optional, Tuple


class RetryBreaker:
    """Track submit failure patterns and signal when strategy should change."""

    def __init__(
        self,
        max_same_error: int = 4,
        max_total_submits: int = 30,
        max_same_vdiff: int = 3,
    ):
        """
        Args:
            max_same_error: min consecutive same-category failures before the
                shape/values hint fires. Default 4 (was 3) — empirically 3 caused
                ~60% false-positive on hard_59 passes because agents legitimately
                repeat wrong_values during iterative refinement.
            max_total_submits: hard cap on submits per task before the
                "stop retrying" hint. Default 15 (was 10) — passing hard_59
                tasks frequently submit 10+ times.
            max_same_vdiff: min consecutive identical-value-diff failures
                before the strongest "truly stuck" hint. Kept at 3 — this
                signal has 0% false-positive rate on historical PASS traces.
        """
        self.max_same_error = max_same_error
        self.max_total_submits = max_total_submits
        self.max_same_vdiff = max_same_vdiff
        # task_id -> list of (pred_shape, gt_shape, error_category, vdiff_sig)
        self._history: dict = defaultdict(list)

    def record_failure(
        self, task_id: str, pred_shape: Tuple[int, int], gt_shape: Tuple[int, int],
        error_category: str = "unknown", vdiff_sig: int = 0
    ) -> Optional[str]:
        """Record a submit failure and return a hint if strategy should change.

        Args:
            task_id: Task identifier.
            pred_shape: (rows, cols) of agent's result.
            gt_shape: (rows, cols) expected.
            error_category: 'shape_mismatch', 'wrong_values', 'sql_error', 'unknown'.
            vdiff_sig: hash of the server-reported value-diff body. If non-zero
                and unchanged across N consecutive wrong_values submits, that's
                the strongest "truly stuck" signal (same SQL or SQL that produces
                identical wrong rows).

        Returns:
            Strategy hint string if breaker triggers, None otherwise.
        """
        sig = (pred_shape, gt_shape, error_category, vdiff_sig)
        self._history[task_id].append(sig)
        history = self._history[task_id]

        # Check total submits
        total = len(history)
        if total >= self.max_total_submits:
            return (
                f"You have submitted {total} times without success. "
                f"STOP repeating the same SQL shape — analyze the last failure "
                f"message carefully and try a DIFFERENT join, aggregation, or "
                f"filter. If the test-case error message mentions specific "
                f"columns or values, address those exactly. "
                f"DO NOT submit placeholder queries like 'SELECT 1' or 'SELECT NULL'. "
                f"If you genuinely don't know the answer, call ask_user for the "
                f"exact formula/business rule, then write a real query."
            )

        # Strongest signal: same value_diff repeated N times (wrong_values only).
        # Uses a tighter window (max_same_vdiff, default 3) because this signal
        # has 0% false-positive on historical PASS traces.
        vwin = history[-self.max_same_vdiff:]
        if len(vwin) >= self.max_same_vdiff:
            vcats = [s[2] for s in vwin]
            vdiffs = [s[3] for s in vwin]
            if (all(c == "wrong_values" for c in vcats)
                    and len(set(vdiffs)) == 1 and vdiffs[0] != 0):
                return (
                    f"You've submitted SQL with the **identical wrong rows** "
                    f"{self.max_same_vdiff} times in a row. You are not making progress. "
                    f"Re-read the task question and knowledge definitions carefully — "
                    f"the bug is in how you interpret the question (filter boundary? "
                    f"aggregation level? column choice?), not in SQL syntax. "
                    f"Try a fundamentally different formula or ask_user for one specific "
                    f"clarifying point."
                )

        # Weaker signal: same category repeated max_same_error times
        recent = history[-self.max_same_error:]
        if len(recent) < self.max_same_error:
            return None

        categories = [s[2] for s in recent]
        shapes = [s[0] for s in recent]

        if len(set(categories)) == 1 and len(set(shapes)) <= 2:
            cat = categories[0]
            if cat == "wrong_values":
                return (
                    f"Your SQL has the correct shape but wrong values "
                    f"({self.max_same_error} times). The issue is likely in "
                    f"your formula, JOIN condition, or WHERE filter — not the "
                    f"query structure. Try: (1) re-read the knowledge definition, "
                    f"(2) check your calculation formula, (3) verify JOIN keys."
                )
            elif cat == "shape_mismatch":
                return (
                    f"Your SQL has the wrong number of rows/columns "
                    f"({self.max_same_error} times). Try a fundamentally "
                    f"different approach: check your WHERE clause, GROUP BY, "
                    f"or whether you need a different JOIN type."
                )
            elif cat == "sql_error":
                return (
                    f"SQL execution error repeated {self.max_same_error} times. "
                    f"Inspect the error message carefully — is it a column name "
                    f"typo, a missing JOIN, or a type mismatch? Call explain_error "
                    f"or get_schema before submitting again."
                )

        return None

    def record_success(self, task_id: str):
        """Clear history on success."""
        self._history.pop(task_id, None)

    def get_submit_count(self, task_id: str) -> int:
        return len(self._history.get(task_id, []))

    def reset(self, task_id: str):
        self._history.pop(task_id, None)
