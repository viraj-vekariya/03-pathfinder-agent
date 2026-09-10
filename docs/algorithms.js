/* Graph algorithms, reimplemented in JavaScript so the demo runs entirely in the browser.
 *
 * These mirror graph/algorithms.py exactly, and tools/check_js_matches_python.py asserts
 * they produce identical install orders on the real 411-package graph. Two independent
 * implementations agreeing is a much stronger correctness claim than either alone.
 *
 * Both are ITERATIVE for the same reason as the Python: real dependency chains reach
 * depths that overflow a recursive implementation's stack.
 */

function reverseGraph(edges, nodes){
  const r = {}; nodes.forEach(n => r[n] = []);
  for(const [u, vs] of Object.entries(edges)) for(const v of vs){ (r[v] = r[v] || []).push(u); }
  return r;
}

function closureFrom(edges, root){
  // BFS-induced subgraph: everything installing `root` pulls in, transitively.
  const seen = new Set([root]); const queue = [root];
  while(queue.length){
    const n = queue.shift();
    for(const m of (edges[n] || [])) if(!seen.has(m)){ seen.add(m); queue.push(m); }
  }
  return seen;
}

function installOrder(edges, root){
  // Kahn's algorithm over the REVERSED closure. The reversal is the subtlety: edges point
  // at what a package DEPENDS ON, so sorting them directly yields dependents before
  // dependencies - exactly backwards for an install order.
  const nodeSet = closureFrom(edges, root);
  const nodes = [...nodeSet];
  const sub = {};
  nodes.forEach(n => sub[n] = (edges[n] || []).filter(m => nodeSet.has(m)));
  const rev = reverseGraph(sub, nodes);

  const indeg = {}; nodes.forEach(n => indeg[n] = 0);
  for(const n of nodes) for(const m of rev[n]) indeg[m]++;

  // Tie-break exactly as the Python does, and the exact form matters. Python keeps a FIFO
  // queue and appends each newly-ready BATCH sorted among itself; it does not re-sort the
  // whole queue. Re-sorting globally turns it into a priority queue and yields a DIFFERENT
  // (still valid) topological order - which is how the first version of this file diverged
  // from the Python on all ten sample packages while producing correct-looking output.
  const queue = nodes.filter(n => indeg[n] === 0).sort();
  const order = [];
  while(queue.length){
    const n = queue.shift(); order.push(n);
    const newly = [];
    for(const m of (rev[n] || [])) if(--indeg[m] === 0) newly.push(m);
    newly.sort();
    for(const m of newly) queue.push(m);
  }
  return order.length === nodes.length ? order : null;   // null => the closure is cyclic
}

function depthOf(edges, root){
  // The LONGEST dependency chain from `root` - a DP over a topological order, matching
  // graph/algorithms.py:longest_path_dag.
  //
  // NOT BFS levels. BFS gives the SHORTEST path to each node, and the first version of
  // this file used it: it silently under-reported depth for any package reachable by both
  // a short and a long route, which is most of them. Longest path is NP-hard in general
  // but linear on a DAG, which is why the cycle check has to come first.
  const nodeSet = closureFrom(edges, root);
  const nodes = [...nodeSet];
  const sub = {};
  nodes.forEach(n => sub[n] = (edges[n] || []).filter(m => nodeSet.has(m)));

  const indeg = {}; nodes.forEach(n => indeg[n] = 0);
  for(const n of nodes) for(const m of sub[n]) indeg[m]++;
  const queue = nodes.filter(n => indeg[n] === 0);
  const topo = [];
  while(queue.length){
    const n = queue.shift(); topo.push(n);
    for(const m of sub[n]) if(--indeg[m] === 0) queue.push(m);
  }

  const depth = {}; nodes.forEach(n => depth[n] = 0);
  for(const n of topo) for(const m of sub[n]) depth[m] = Math.max(depth[m], depth[n] + 1);
  const max = nodes.reduce((a, n) => Math.max(a, depth[n]), 0);
  return {max, level: depth};
}

function findCycles(edges, nodes){
  // Iterative Tarjan. Returns every strongly connected component with more than one node,
  // plus self-loops. Tarjan rather than "DFS and remember the stack" because it finds ALL
  // components in one pass.
  let idx = 0;
  const index = {}, low = {}, onStack = {}, stack = [], out = [];
  for(const root of nodes){
    if(index[root] !== undefined) continue;
    const work = [[root, 0]];
    index[root] = low[root] = idx++; stack.push(root); onStack[root] = true;
    while(work.length){
      const frame = work[work.length - 1];
      const node = frame[0];
      const nbrs = edges[node] || [];
      if(frame[1] < nbrs.length){
        const nxt = nbrs[frame[1]++];
        if(index[nxt] === undefined){
          index[nxt] = low[nxt] = idx++; stack.push(nxt); onStack[nxt] = true;
          work.push([nxt, 0]);
        } else if(onStack[nxt]){
          low[node] = Math.min(low[node], index[nxt]);
        }
      } else {
        work.pop();
        if(work.length){
          const parent = work[work.length - 1][0];
          low[parent] = Math.min(low[parent], low[node]);
        }
        if(low[node] === index[node]){
          const comp = [];
          for(;;){ const w = stack.pop(); onStack[w] = false; comp.push(w); if(w === node) break; }
          if(comp.length > 1 || (edges[node] || []).includes(node)) out.push(comp.sort());
        }
      }
    }
  }
  return out.sort((a, b) => a.length - b.length);
}

if(typeof module !== "undefined") module.exports = {installOrder, closureFrom, depthOf, findCycles};
