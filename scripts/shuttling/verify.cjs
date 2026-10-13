// Independent physical checks of one compiled memory or fibre-surgery schedule; throws on
// the first violation.  `model` is the schedule, `source` the animation model it implements.
//  - moves: rest-to-rest duration tau(L) = 2 sqrt(L / a_max), a_max = 5500 m/s^2 (L in um,
//    times in us), within the schedule, each atom at most once;
//  - gates: 1 us matchings within the schedule, at 2 um docking distance when replayed;
//  - no atom or AOD pair is in two operations at once;
//  - every check edge of `source` is applied once per basis and round;
//  - atoms return to their home fibre slots (single round) or labelled homes (parallel).
// Parallel X/Z schedules over r rounds (`model.rounds`) must also read out each round after
// its gates, start the next round after reset, keep check assignments fixed within a round,
// keep X/Z rounds non-interleaved on every data atom, restore homes after the last readout
// and actually overlap X and Z.
const assert = require('assert/strict');
const EPS = 1e-5;

function verify(model, source) {
  const busy = new Map();  // 'atom:i' or 'pair:j' -> [{t, end}]
  const reserve = (id, t, end) => { if (!busy.has(id)) busy.set(id, []); busy.get(id).push({t, end}); };
  const events = [], edges = new Map(), dataRounds = new Map();
  for (const move of model.moves) {
    const [t, duration, dx, dy, pair, atoms] = move;
    assert(Math.abs(duration - 2 * Math.sqrt(Math.hypot(dx, dy) / 5500) * 1000) < EPS, 'Transport duration');
    assert(t >= -EPS && t + duration <= model.total + EPS, 'Motion within schedule');
    assert(new Set(atoms).size === atoms.length, 'Unique moving atoms');
    reserve('pair:' + pair, t, t + duration);
    for (const id of atoms) reserve('atom:' + id, t, t + duration);
    events.push({t: t + duration, order: 0, move});
  }
  for (const gate of model.gates) {
    assert(Math.abs(gate.end - gate.t - 1) < EPS, 'Gate duration');
    assert(gate.t >= -EPS && gate.end <= model.total + EPS, 'Gate within schedule');
    assert(new Set(gate.edges.flat()).size === 2 * gate.edges.length, 'Gate is a matching');
    for (const id of gate.edges.flat()) reserve('atom:' + id, gate.t, gate.end);
    events.push({t: gate.t, order: 1, gate});
    const key = gate.basis + ':' + (gate.round || 1);
    if (!edges.has(key)) edges.set(key, []);
    edges.get(key).push(...(gate.logicalEdges || gate.edges).map(e => e.join(':')));
    if (model.rounds) for (const [, data] of gate.edges) {
      if (!dataRounds.has(data)) dataRounds.set(data, []);
      dataRounds.get(data).push({t: gate.t, rank: 2 * (gate.round - 1) + (gate.basis === 'Z' ? 1 : 0)});
    }
  }
  for (const [id, ops] of busy) {
    ops.sort((a, b) => a.t - b.t || a.end - b.end);
    for (let i = 1; i < ops.length; i++) assert(ops[i - 1].end <= ops[i].t + EPS, 'Busy ' + id);
  }

  // Replay the moves (applied at their end) and check the docking distance of every gate.
  const positions = model.atoms.map(a => a.slice(0, 2));
  events.sort((a, b) => Math.abs(a.t - b.t) < EPS ? a.order - b.order : a.t - b.t);
  for (const {move, gate} of events) {
    if (move) for (const id of move[5]) { positions[id][0] += move[2]; positions[id][1] += move[3]; }
    else for (const [a, b] of gate.edges) {
      const distance = Math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]);
      assert(Math.abs(distance - 2) < EPS, 'Gate docking');
    }
  }
  if (model.rounds) {
    positions.forEach((p, i) => assert(Math.hypot(p[0] - model.atoms[i][0], p[1] - model.atoms[i][1]) < EPS, 'Final labelled homes'));
  } else {  // atoms (x, y, kind, module, fibre, ...) may permute within their fibre
    const slots = new Map(), final = new Map();
    const add = (map, atom, point) => { const key = atom.slice(2, 5).join(':'); if (!map.has(key)) map.set(key, []); map.get(key).push(point.join(':')); };
    model.atoms.forEach((atom, i) => { add(slots, atom, atom.slice(0, 2)); add(final, atom, positions[i]); });
    for (const [key, points] of slots) assert.deepEqual(final.get(key).sort(), points.sort(), 'Returned to same fibre slots');
  }
  for (const [key, applied] of edges) {
    const expected = source.gates.filter(g => g.basis === key[0]).flatMap(g => g.edges.map(e => e.join(':')));
    assert.deepEqual(applied.sort(), expected.sort(), 'Per-round check incidence');
  }
  if (!model.rounds) return;

  assert.equal(model.readouts.length, 2 * model.rounds);
  assert.equal(edges.size, 2 * model.rounds);
  for (const basis of ['X', 'Z']) for (let r = 1; r <= model.rounds; r++) {
    const read = model.readouts.filter(x => x.basis === basis && x.round === r);
    assert.equal(read.length, 1);
    const gates = model.gates.filter(g => g.basis === basis && g.round === r);
    assert(Math.max(...gates.map(g => g.end)) <= read[0].t + EPS, 'Readout after all check gates');
    if (r > 1) {
      const previous = model.readouts.find(x => x.basis === basis && x.round === r - 1);
      assert(Math.min(...gates.map(g => g.t)) >= previous.end - EPS, 'Next round after reset');
    }
    const assignment = new Map(model.roundAssignments.find(x => x.basis === basis && x.round === r).checks);
    for (const g of gates) g.edges.forEach((edge, j) => assert.deepEqual(Array.from(edge),
      [assignment.get(g.logicalEdges[j][0]), g.logicalEdges[j][1]], 'Fixed check assignment during round'));
  }
  for (const ops of dataRounds.values()) {
    ops.sort((a, b) => a.t - b.t);
    for (let i = 1; i < ops.length; i++) assert(ops[i].rank >= ops[i - 1].rank, 'Noninterleaved X/Z round order on data');
  }
  assert(model.pipeline.finalRestoreStart >= Math.max(...model.readouts.map(r => r.end)) - EPS, 'Restore after readout');
  assert(model.parallelWitness, 'Actual X/Z overlap');
}
module.exports = verify;
