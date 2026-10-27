/* Static data transport for the original PASCAL portable demo UI. */
(() => {
  const nativeFetch = window.fetch.bind(window);
  const base = new URL('.', document.baseURI);
  const cache = new Map();
  async function read(path, signal) {
    const response = await nativeFetch(new URL(path, base), {signal});
    if (!response.ok) throw new Error(`Could not load stored data (${response.status}). Please retry.`);
    return response.json();
  }
  function cached(path) {
    if (!cache.has(path)) {
      const promise = read(path).catch(error => {cache.delete(path); throw error;});
      cache.set(path, promise);
      if (cache.size > 24) cache.delete(cache.keys().next().value);
    }
    return cache.get(path);
  }
  const response = value => new Response(JSON.stringify(value), {headers: {'Content-Type': 'application/json'}});
  window.fetch = async (input, init) => {
    if (typeof input !== 'string' || !input.startsWith('/api/')) return nativeFetch(input, init);
    const url = new URL(input, location.origin);
    const route = url.pathname.slice(5);
    const split = url.searchParams.get('split') || 'lite';
    const archive = url.searchParams.get('archive') || 'frozen';
    const id = url.searchParams.get('id');
    if (!['lite', 'full'].includes(split) || !['frozen', 'submitted', 'react'].includes(archive)) throw new Error('Unknown dataset');
    if (route === 'health') return response({ok: true, live_enabled: false, active_split: '', agent_model: '900 tasks · 2,083 stored trajectories'});
    if (route === 'tasks') return response(await cached(`data/lists/${archive}-${split}.json`));
    if (route === 'has_replay') {
      const groups = await cached(`data/lists/${archive}-${split}.json`);
      return response({available: groups.some(group => group.tasks.some(task => task.instance_id === id && task.has_replay))});
    }
    if (['task', 'schema', 'kb', 'analysis'].includes(route)) {
      const task = await cached(`data/tasks/${split}/${encodeURIComponent(id)}.json`);
      if (route === 'schema') return response(await cached(`data/schemas/${split}/${encodeURIComponent(task.task.database)}.json`));
      return response(task[route]);
    }
    throw new Error('This action requires the portable demo with a live backend.');
  };
  window.StaticReplay = class {
    constructor(input) {
      this.controller = new AbortController();
      this.closed = false;
      const url = new URL(input, location.origin);
      const split = url.searchParams.get('split');
      const archive = url.searchParams.get('archive');
      const id = url.searchParams.get('id');
      if (url.pathname !== '/api/replay' || !['lite', 'full'].includes(split) || !['frozen', 'submitted', 'react'].includes(archive)) {
        queueMicrotask(() => this.onerror?.(new Error('Replay only'))); return;
      }
      this.play(`data/replays/${archive}/${split}/${encodeURIComponent(id)}.json`);
    }
    async play(path) {
      try {
        const events = await read(path, this.controller.signal);
        for (const event of events) {
          if (this.closed) return;
          this.onmessage?.({data: JSON.stringify(event)});
          if (this.closed) return;
          // Yield even at instant speed so Stop/task switching can cancel a long replay.
          if (event.type === 'tool_step') {
            const speed = Number(document.querySelector('#replaySpeed')?.value || 1);
            await new Promise(resolve => {this.resume = resolve; this.timer = setTimeout(resolve, speed === 0 ? 0 : 350 / speed);});
            this.resume = null;
          }
        }
      } catch (error) {
        if (!this.closed && error.name !== 'AbortError') this.onerror?.(error);
      }
    }
    close() {
      this.closed = true;
      this.controller.abort();
      clearTimeout(this.timer);
      this.resume?.();
    }
  };
})();
