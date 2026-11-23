// A* на сетке с 8-связностью, октильная эвристика. grid[y][x] = 1 — препятствие.
export function astar(grid, start, goal, { heuristic = true } = {}) {
  const H = grid.length, W = grid[0].length;
  const idx = (x, y) => y * W + x;
  const g = new Float64Array(W * H).fill(Infinity);
  const came = new Int32Array(W * H).fill(-1);
  const closed = new Uint8Array(W * H);
  const h = (x, y) => {
    if (!heuristic) return 0;
    const dx = Math.abs(x - goal[0]), dy = Math.abs(y - goal[1]);
    return dx + dy + (Math.SQRT2 - 2) * Math.min(dx, dy);
  };
  // двоичная куча
  const heap = [];
  const push = (n) => {
    heap.push(n);
    let i = heap.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (heap[p].f <= heap[i].f) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; }
  };
  const pop = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l].f < heap[m].f) m = l;
        if (r < heap.length && heap[r].f < heap[m].f) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]]; i = m;
      }
    }
    return top;
  };
  if (!grid[start[1]] || grid[start[1]][start[0]] || grid[goal[1]][goal[0]]) return { path: null, expanded: 0, cost: Infinity };
  g[idx(...start)] = 0;
  push({ x: start[0], y: start[1], f: h(...start) });
  let expanded = 0;
  const dirs = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
  while (heap.length) {
    const cur = pop();
    const ci = idx(cur.x, cur.y);
    if (closed[ci]) continue;
    closed[ci] = 1;
    expanded++;
    if (cur.x === goal[0] && cur.y === goal[1]) break;
    for (const [dx, dy, c] of dirs) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H || grid[ny][nx]) continue;
      if (dx && dy && (grid[cur.y][nx] || grid[ny][cur.x])) continue; // без срезания углов
      const ni = idx(nx, ny);
      const ng = g[ci] + c;
      if (ng < g[ni]) { g[ni] = ng; came[ni] = ci; push({ x: nx, y: ny, f: ng + h(nx, ny) }); }
    }
  }
  const gi = idx(...goal);
  if (!Number.isFinite(g[gi])) return { path: null, expanded, cost: Infinity };
  const path = [];
  for (let c = gi; c !== -1; c = came[c]) path.push([c % W, Math.floor(c / W)]);
  path.reverse();
  return { path, expanded, cost: g[gi] };
}

export function lineFree(grid, a, b) {
  // проверка прямой видимости (с шагом 0.25 клетки)
  const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 4);
  for (let i = 0; i <= n; i++) {
    const t = i / (n || 1);
    const x = Math.round(a[0] + (b[0] - a[0]) * t), y = Math.round(a[1] + (b[1] - a[1]) * t);
    if (grid[y]?.[x]) return false;
  }
  return true;
}

// «Натягивание нити»: удаляем лишние узлы, сохраняя прямую видимость
export function smoothPath(grid, path) {
  if (!path || path.length < 3) return path;
  const out = [path[0]];
  let i = 0;
  while (i < path.length - 1) {
    let j = path.length - 1;
    while (j > i + 1 && !lineFree(grid, path[i], path[j])) j--;
    out.push(path[j]);
    i = j;
  }
  return out;
}

export const pathLen = (p) => (p ? p.reduce((s, q, i) => (i ? s + Math.hypot(q[0] - p[i - 1][0], q[1] - p[i - 1][1]) : 0), 0) : Infinity);

// Улучшение порядка обхода точек: 2-opt по матрице расстояний (точка 0 — база, фиксирована)
export function twoOpt(order, D) {
  const len = (o) => o.reduce((s, v, i) => (i ? s + D[o[i - 1]][v] : 0), 0) + D[o[o.length - 1]][o[0]];
  let best = order.slice(), bestL = len(best), improved = true;
  while (improved) {
    improved = false;
    for (let i = 1; i < best.length - 1; i++)
      for (let k = i + 1; k < best.length; k++) {
        const cand = best.slice(0, i).concat(best.slice(i, k + 1).reverse(), best.slice(k + 1));
        const l = len(cand);
        if (l < bestL - 1e-9) { best = cand; bestL = l; improved = true; }
      }
  }
  return best;
}
