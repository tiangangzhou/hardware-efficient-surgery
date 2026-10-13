// Replay the animation's own schedulers on its embedded models (run by extract_timings.py).
// Input (stdin): the JSON of animation.compiler_inputs().
// Output (stdout): {rows, verified} with one row of Table S2 per (lift, method, AOD pairs):
// X, Z and nonparallel X+Z round times, and for memory / algebraic fibre surgery the
// parallel X/Z time per round, T_XZ^(r) / r with r = 8, 10, 20 (all in ms); `verified`
// counts the memory and fibre schedules that pass verify.cjs.
const fs = require('fs'), vm = require('vm'), verify = require('./verify.cjs');
const input = JSON.parse(fs.readFileSync(0, 'utf8')), ctx = {};
vm.createContext(ctx);
for (const code of [input.runtime, input.decoders, input.parallel]) vm.runInContext(code, ctx);
const ROUNDS = {9: 8, 15: 10, 33: 20};
const rows = [];
let verified = 0;
const check = (schedule, model) => { verify(schedule, model); verified++; return schedule; };

for (const m of ctx.unpackModels(input.lp).models) {
  if (m.mode !== 'memory' && m.mode !== 'fibre') continue;
  for (const sector of ['X', 'Z']) check(ctx.AodParallel.serial(m, sector), m);
  const t = check(ctx.AodParallel.serial(m, 'all'), m).sectorTimesUs;
  const parallel = check(ctx.AodParallel.compile(m, ROUNDS[m.lift], 0), m);
  rows.push({lift: m.lift, method: m.mode === 'fibre' ? 'algebraic' : 'memory', pairs: m.pairsPerCode,
    X_ms: t.X / 1000, Z_ms: t.Z / 1000, XZ_ms: t.all / 1000,
    XZ_parallel_ms: parallel.total / ROUNDS[m.lift] / 1000});
}
const hybrid = ctx.unpackHybrid(input.hybrid);
for (const g of hybrid.graph.models) {
  for (const bare of hybrid.bare.models.filter(b => b.lift === g.lift)) {
    const all = ctx.GraphHybrid.compile(g, bare, 'all', bare.pairs);
    rows.push({lift: g.lift, method: g.method, pairs: bare.pairs,
      X_ms: ctx.GraphHybrid.select(all, 'X').total / 1000,
      Z_ms: ctx.GraphHybrid.select(all, 'Z').total / 1000,
      XZ_ms: all.total / 1000, XZ_parallel_ms: ''});
  }
}
process.stdout.write(JSON.stringify({rows, verified}) + '\n');
