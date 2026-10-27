# PASCAL browser replay demo

[Open demo](https://authurlord.github.io/PASCAL_Release/) · [Paper](https://aclanthology.org/2026.emnlp-main.465/) · [Method code](https://github.com/authurlord/PASCAL_Release/tree/main)

Static adaptation of the portable PASCAL demo, preserving task browsing, schema and KB inspection, answer keys, cached explanations and all 2,083 stored trajectories across 900 tasks. No server, model or database is required.

Choose a dataset and database, select a task, then click Replay anchor. The archive selector switches between PASCAL frozen, PASCAL submitted NVD and ReAct NVD. Replay speed supports 1×, 4× and Instant. Stop, task switching and archive switching cancel playback.

## Local preview

Serve this directory using any static web server, for example:

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open http://127.0.0.1:8080/. Serve over HTTP; opening index.html directly with file:// cannot fetch JSON.

## Rebuild

Unpack PASCAL_demo_portable_20260923.zip and install its replay dependencies. With that Python environment:

```bash
python export_site.py /path/to/PASCAL_demo_portable_20260923 /path/to/output
```

Keep `static-adapter.js` beside the exporter. It exports only replay-facing JSON and validates task counts, event counts and final outcomes against the original archives. Preserve README.md and DATA_SOURCES.md when rebuilding into a new directory.

Deployment: GitHub Pages from the root of this repository's `gh-pages` branch. The `main` branch contains method code.

[Data sources, version details and licenses](DATA_SOURCES.md).
