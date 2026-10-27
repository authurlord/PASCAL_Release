"""Export only the portable demo's replay-facing responses; never launch a model/DB.

Usage: <bundle>/.venv/bin/python export_site.py <unpacked-bundle> <output-directory>
The output can be served at either / or a GitHub Pages repository subpath.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

BUNDLE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
os.environ['DEMO_LIVE'] = '0'
os.environ.pop('PASCAL_FREE_MODE', None)
spec = importlib.util.spec_from_file_location('portable_demo', BUNDLE / 'BIRD-Interact-ADK/demo/app.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

def write(path, value):
    destination = OUT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

def payload(response):
    assert response.status_code == 200
    return json.loads(response.body)

async def main():
    counts = {}
    for split, tasks in demo._TASKS.items():
        schemas = set()
        for tid, task in tasks.items():
            assert re.fullmatch(r'[A-Za-z0-9_-]+', tid), tid
            details = payload(demo.task(split, tid))
            details.pop('initial_budget', None)
            analysis = payload(demo.analysis(split, tid))
            assert analysis.get('cached') and analysis.get('analysis_zh'), tid
            write(f'data/tasks/{split}/{tid}.json', {'task': details, 'kb': payload(demo.kb(split, tid)), 'analysis': analysis})
            db = task['selected_database']
            assert re.fullmatch(r'[A-Za-z0-9_-]+', db), db
            if db not in schemas:
                schema = payload(demo.schema(split, tid))
                assert 'error' not in schema, db
                if not schema['tables']:
                    schema['raw'] = demo._schema_text(split, db)
                write(f'data/schemas/{split}/{db}.json', schema)
                schemas.add(db)
        for archive in ['frozen', 'submitted', 'react']:
            write(f'data/lists/{archive}-{split}.json', payload(demo.tasks(split, archive)))
            records = demo._load_archive(split, archive)
            counts[f'{archive}-{split}'] = len(records)
            for tid, rec in records.items():
                response = await demo.replay(split, tid, delay=0, archive=archive)
                events = []
                async for chunk in response.body_iterator:
                    events.extend(json.loads(line[5:]) for line in chunk.splitlines() if line.startswith('data:'))
                assert events[-1]['type'] == 'demo_done', tid
                assert sum(e['type'] == 'tool_step' for e in events) == len(rec.get('tool_trajectory') or []), tid
                assert not any(e['type'] == 'error' for e in events), tid
                assert next(e for e in events if e['type'] == 'final')['state']['phase1_passed'] == bool(rec.get('phase1_passed')), tid
                write(f'data/replays/{archive}/{split}/{tid}.json', events)
    assert sum(counts.values()) == 2083, counts
    html = (BUNDLE / 'BIRD-Interact-ADK/demo/index.html').read_text(encoding='utf-8')
    html = html.replace('https://anonymous.4open.science/r/PASCAL_Release-733A', 'https://github.com/authurlord/PASCAL_Release')
    html = html.replace('src="/bird-interact.png"', 'src="bird-interact.png"')
    html = html.replace('</head>', '<meta name="description" content="Explore 900 BIRD-Interact tasks and replay 2,083 PASCAL and ReAct trajectories in your browser. EMNLP 2026.">\n<script src="static-adapter.js"></script>\n<style>header{flex-wrap:wrap;gap:8px}a.demo-link{color:var(--cyan);font-size:12px;text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:6px 9px}.webnote{font-size:12px;color:var(--mut);padding:7px 16px;border-bottom:1px solid var(--line)}@media(max-width:760px){body{height:auto;min-height:100vh}header .sub{display:none}main{display:block}#left,#mid,#right{width:100%!important;max-width:none!important}#left{max-height:40vh}#mid{max-height:65vh}.resizer{display:none}#right{min-height:50vh}#uspanel{width:min(350px,85vw)}}</style>\n</head>')
    html = html.replace('<span id="health"', '<a class="demo-link" href="https://aclanthology.org/2026.emnlp-main.465/" target="_blank" rel="noopener">Paper · ACL</a>\n<a class="demo-link" href="https://github.com/authurlord/PASCAL_Release" target="_blank" rel="noopener">Code</a>\n<label style="font-size:12px">Replay speed <select id="replaySpeed" aria-label="Replay speed"><option value="1">1×</option><option value="4">4×</option><option value="0">Instant</option></select></label>\n<span id="health"')
    html = html.replace('</header>', '</header>\n<div class="webnote">Browser replay · 存档回放：SQL results and model replies are recorded, not generated live. <a href="DATA_SOURCES.md" style="color:inherit">Data &amp; attribution</a></div>')
    html = html.replace('new EventSource(', 'new StaticReplay(')
    html = html.replace('let demoLive=false;', 'let listEpoch=0,selectionEpoch=0;\nlet demoLive=false;')
    html = html.replace('function changeArchive(value){', 'function changeArchive(value){\n  ++selectionEpoch; $("#fsm").classList.remove("on"); $("#stats").classList.remove("on");')
    html = html.replace('async function loadList(sp){', 'async function loadList(sp){\n  const requestEpoch=++listEpoch;')
    html = html.replace('if(split!==sp){', 'if(split!==sp){\n    ++selectionEpoch; $("#fsm").classList.remove("on"); $("#stats").classList.remove("on");')
    html = html.replace('if(!tasks[sp].length)tasks[sp]=await(await fetch("/api/tasks?split="+sp)).json();', 'if(!tasks[sp].length){const result=await(await fetch("/api/tasks?split="+sp)).json(); if(requestEpoch!==listEpoch)return; tasks[sp]=result;}\n  if(requestEpoch!==listEpoch)return;')
    html = html.replace('async function selectTask(id){', 'async function selectTask(id){\n  const requestEpoch=++selectionEpoch; $("#fsm").classList.remove("on"); $("#stats").classList.remove("on");')
    html = html.replace('async function loadSchema(id){', 'async function loadSchema(id){\n  const requestEpoch=selectionEpoch;')
    html = html.replace('async function loadKB(id){', 'async function loadKB(id){\n  const requestEpoch=selectionEpoch;')
    html = html.replace('if(sel!==id)return;', 'if(sel!==id||requestEpoch!==selectionEpoch)return;')
    html = html.replace('if(!tbl.length){w.innerHTML=`<div class="lbl">${esc(t2("schema"))} —</div>`;return;}', 'if(!tbl.length){w.innerHTML=`<div class="lbl">${esc(t2("schema"))}</div><pre style="white-space:pre-wrap">${esc(s.raw||"—")}</pre>`;return;}')
    html = html.replace('离线回放 / replay ready', 'browser replay ready')
    html = html.replace('backend error', 'data unavailable — reload to retry')
    html = html.replace('stream closed before completion — services may have dropped; press Run again', 'stored trajectory could not be loaded — please retry Replay')
    html = html.replace('Generate 中文 analysis', 'Read cached 中文 analysis').replace('生成中文解析', '读取中文解析')
    html = html.replace('Press “Run agent” (live vLLM) or “Replay anchor” (stored).', 'Press “Replay anchor” to watch the stored trajectory.')
    html = html.replace('点击「运行智能体」（实时 vLLM）或「回放标准轨迹」（已存）。', '点击「回放标准轨迹」查看已存档的工具调用与结果。')
    html = re.sub(r'<button id="runBtn".*?</button>', '<a class="demo-link" href="https://github.com/authurlord/PASCAL_Release#quick-start" target="_blank" rel="noopener">Live setup ↗</a>', html)
    html = re.sub(r'<button class="ghost" style="font-size:11px;padding:3px 8px" onclick="refreshSample.*?</button>', '<span style="font-size:11px">archived sample</span>', html)
    # Update inherited help so the static site's instructions match its controls.
    html = html.replace('applyLang();health().then(()=>loadList("lite"));', '''HELP.en = `## Browser replay demo
1. Select a trajectory set: PASCAL frozen, PASCAL submitted NVD, or ReAct NVD.
2. Select lite_300 or full_600, expand a database group, and choose a task. Use All to include failures.
3. Click Replay anchor to play recorded tool calls, SQL results and user clarifications. Choose 1×, 4× or Instant speed. Stop cancels playback.
4. Inspect schema tables, archived samples and the knowledge base. Answer key opens reference SQL and cached Chinese explanations.
5. Switch archive, task or split to explore another run. ReAct has 290 lite trajectories and no full trajectories.
This website replays stored results; it does not call a model or execute SQL. Live setup links to the code repository.
The 2,083 trajectories include different runs of the same 900 tasks. Historical submitted full and ReAct lite sets have 7 and 10 missing records respectively.`;
HELP.zh = `## 浏览器回放演示
1. 选择轨迹集：PASCAL frozen、PASCAL submitted NVD 或 ReAct NVD。
2. 选择 lite_300 或 full_600，展开数据库分组并选择任务。切到 All 可查看失败案例。
3. 点击回放，逐步查看存档中的工具调用、SQL 结果和用户澄清。可选 1×、4× 或 Instant 速度；Stop 停止播放。
4. 查看 schema、表格存档样例和知识库；答案解析可查看参考 SQL 与缓存中文解析。
5. 切换轨迹集、任务或数据划分即可查看另一条记录。ReAct 只有 290 条 lite 轨迹，没有 full 轨迹。
网页不调用模型、不执行 SQL；Live setup 链接提供实时运行的安装方法。
2,083 条轨迹是同一组 900 道任务的不同运行记录；历史 submitted full 和 ReAct lite 分别缺少 7 条和 10 条记录。`;
applyLang();health().then(()=>loadList("lite")).catch(error=>{document.querySelector('#list').textContent=error.message;});''')
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('static-adapter.js'), OUT / 'static-adapter.js')
    shutil.copy2(BUNDLE / 'BIRD-Interact-ADK/demo/bird-interact.png', OUT / 'bird-interact.png')
    shutil.copy2(BUNDLE / 'LICENSE', OUT / 'LICENSE')
    (OUT / '.nojekyll').touch()
    write('manifest.json', {'source_bundle': 'PASCAL_demo_portable_20260923.zip', 'tasks': {k: len(v) for k, v in demo._TASKS.items()}, 'trajectories': counts, 'total_trajectories': sum(counts.values()), 'paper': 'https://aclanthology.org/2026.emnlp-main.465/'})
    print(json.dumps({'tasks': 900, 'trajectories': counts, 'total': sum(counts.values()), 'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}, indent=2))

asyncio.run(main())
