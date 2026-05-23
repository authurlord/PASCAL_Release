"""Module 3: Knowledge base pre-loader.

Resolves external_knowledge IDs from task data to actual definitions
and formats them for prompt injection.

Design: pure functions, testable with mock KB data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_knowledge_base(kb_path: str) -> Dict[str, dict]:
    """Load knowledge base entries keyed by ID.

    Supports both JSONL files and JSON files containing a list.
    """
    path = Path(kb_path)
    if not path.exists():
        return {}

    entries = {}
    text = path.read_text()

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for entry in data:
                eid = str(entry.get("id", entry.get("knowledge_id", "")))
                entries[eid] = entry
            return entries
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            eid = str(entry.get("id", entry.get("knowledge_id", "")))
            entries[eid] = entry
        except json.JSONDecodeError:
            continue

    return entries


def resolve_knowledge_ids(
    knowledge_ids: List[int],
    kb_entries: Dict[str, dict],
    visible_fields: Optional[List[str]] = None,
) -> List[dict]:
    """Resolve numeric knowledge IDs to their full definitions.

    Args:
        knowledge_ids: List of knowledge entry IDs from task data.
        kb_entries: Full KB dict from load_knowledge_base().
        visible_fields: Which fields to include (default: all).

    Returns:
        List of resolved knowledge entries.
    """
    if visible_fields is None:
        visible_fields = ["id", "knowledge", "description", "formula", "definition"]

    resolved = []
    for kid in knowledge_ids:
        entry = kb_entries.get(str(kid))
        if entry:
            visible = {k: entry[k] for k in visible_fields if k in entry}
            resolved.append(visible)
    return resolved


def format_knowledge_for_prompt(resolved: List[dict]) -> str:
    """Format resolved knowledge entries into a prompt-ready string.

    Args:
        resolved: List of knowledge dicts from resolve_knowledge_ids().

    Returns:
        Formatted string for prompt injection.
    """
    if not resolved:
        return ""

    parts = ["# Pre-loaded Domain Knowledge (from task metadata)"]
    for i, entry in enumerate(resolved, 1):
        name = entry.get("knowledge", entry.get("id", f"Entry {i}"))
        desc = entry.get("description", "")
        formula = entry.get("formula", entry.get("definition", ""))

        part = f"\n## {name}"
        if desc:
            part += f"\n{desc}"
        if formula and formula != desc:
            part += f"\nFormula/Definition: {formula}"
        parts.append(part)

    return "\n".join(parts)


def preload_task_knowledge(
    task_data: dict,
    kb_entries: Dict[str, dict],
) -> str:
    """One-call convenience: extract IDs from task data, resolve, format.

    Args:
        task_data: A single task dict (from the JSONL dataset).
        kb_entries: Full KB dict.

    Returns:
        Formatted knowledge string, or "" if no knowledge needed.
    """
    # external_knowledge field contains the IDs
    ek = task_data.get("external_knowledge", [])
    if isinstance(ek, str):
        try:
            ek = json.loads(ek)
        except (json.JSONDecodeError, TypeError):
            return ""

    if not isinstance(ek, list) or not ek:
        return ""

    # Filter to numeric IDs
    ids = [x for x in ek if isinstance(x, (int, float))]
    if not ids:
        return ""

    resolved = resolve_knowledge_ids(ids, kb_entries)
    return format_knowledge_for_prompt(resolved)
