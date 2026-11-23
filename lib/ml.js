import { rng } from './math';

// ---------- CART (Gini) и Random Forest ----------
function gini(counts, n) {
  let s = 1;
  for (const c of counts) s -= (c / n) ** 2;
  return s;
}
function leaf(y, idx, K) {
  const p = new Array(K).fill(0);
  for (const i of idx) p[y[i]]++;
  return { leaf: true, p: p.map((c) => c / (idx.length || 1)), n: idx.length };
}
export function buildTree(X, y, idx, { depth = 0, maxDepth = 5, minLeaf = 3, mtry = null, r = Math.random, K = 3, imp = null } = {}) {
  const nF = X[0].length;
  const base = leaf(y, idx, K);
  if (depth >= maxDepth || idx.length < 2 * minLeaf || base.p.some((v) => v === 1)) return base;
  const parentCounts = base.p.map((v) => v * idx.length);
  const parentG = gini(parentCounts, idx.length);
  let feats = [...Array(nF).keys()];
  if (mtry) { for (let i = feats.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [feats[i], feats[j]] = [feats[j], feats[i]]; } feats = feats.slice(0, mtry); }
  let best = null;
  for (const f of feats) {
    const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
    const L = new Array(K).fill(0), R = parentCounts.slice();
    for (let k = 0; k < s.length - 1; k++) {
      L[y[s[k]]]++; R[y[s[k]]]--;
      const nl = k + 1, nr = s.length - nl;
      if (nl < minLeaf || nr < minLeaf || X[s[k]][f] === X[s[k + 1]][f]) continue;
      const g = (nl * gini(L, nl) + nr * gini(R, nr)) / s.length;
      if (!best || g < best.g) best = { g, f, thr: (X[s[k]][f] + X[s[k + 1]][f]) / 2 };
    }
  }
  if (!best || best.g >= parentG - 1e-9) return base;
  if (imp) imp[best.f] += (parentG - best.g) * idx.length;
  const li = idx.filter((i) => X[i][best.f] <= best.thr), ri = idx.filter((i) => X[i][best.f] > best.thr);
  const opts = { depth: depth + 1, maxDepth, minLeaf, mtry, r, K, imp };
  return { f: best.f, thr: best.thr, n: idx.length, p: base.p, left: buildTree(X, y, li, opts), right: buildTree(X, y, ri, opts) };
}
export function treeProba(node, x) {
  while (!node.leaf) node = x[node.f] <= node.thr ? node.left : node.right;
  return node.p;
}
export function trainForest(X, y, { nTrees = 25, maxDepth = 6, K = 3, seed = 1 } = {}) {
  const r = rng(seed);
  const imp = new Array(X[0].length).fill(0);
  const trees = [];
  const mtry = Math.max(1, Math.round(Math.sqrt(X[0].length)));
  for (let t = 0; t < nTrees; t++) {
    const boot = Array.from({ length: X.length }, () => Math.floor(r() * X.length));
    trees.push(buildTree(X, y, boot, { maxDepth, mtry, r, K, imp }));
  }
  const s = imp.reduce((a, b) => a + b, 0) || 1;
  return { trees, importance: imp.map((v) => v / s) };
}
export function forestProba(forest, x) {
  const K = forest.trees[0].p.length;
  const p = new Array(K).fill(0);
  for (const t of forest.trees) treeProba(t, x).forEach((v, k) => (p[k] += v / forest.trees.length));
  return p;
}
export const argmax = (p) => p.indexOf(Math.max(...p));

export function confusion(yTrue, yPred, K) {
  const M = Array.from({ length: K }, () => new Array(K).fill(0));
  yTrue.forEach((t, i) => M[t][yPred[i]]++);
  return M;
}

// ---------- Линейный SVM (Pegasos, hinge loss) ----------
export function trainSVM(X, y, { lambda = 0.01, iters = 4000, seed = 2 } = {}) {
  const r = rng(seed);
  const d = X[0].length;
  let w = new Array(d).fill(0), b = 0;
  for (let t = 1; t <= iters; t++) {
    const i = Math.floor(r() * X.length);
    const eta = 1 / (lambda * t);
    const m = y[i] * (w.reduce((s, wj, j) => s + wj * X[i][j], 0) + b);
    w = w.map((wj) => wj * (1 - eta * lambda));
    if (m < 1) { w = w.map((wj, j) => wj + eta * y[i] * X[i][j]); b += eta * y[i] * 0.1; }
  }
  return { w, b };
}

// ---------- Нечёткая логика: треугольные/трапециевидные функции принадлежности ----------
export function trap(x, a, b, c, d) {
  if (x <= a || x >= d) return (x <= a && a === b) || (x >= d && c === d) ? 1 : 0;
  if (x < b) return (x - a) / (b - a);
  if (x <= c) return 1;
  return (d - x) / (d - c);
}
