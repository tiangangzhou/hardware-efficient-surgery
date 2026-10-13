# Parallel algebraic surgery for qLDPC codes with constant shuttling depth

Code and data for

> Tian-Gang Zhou, Boren Gu, Jens Eisert, Chen Zhao,
> *Parallel algebraic surgery for qLDPC codes with constant shuttling depth*.

**Interactive shuttling animation:**
[lp-surgery-aod-polynomials.html](https://tiangangzhou.github.io/hardware-efficient-surgery/lp-surgery-aod-polynomials.html)

## Layout

```
src/                         reusable code
  algebraic_qec/             group algebra F2[x]/(x^l+1), lifted-product codes, CRT kernels,
                             canonical generators of the logical fibres, auxiliary (tube) factors,
                             exhaustive distance search and the paper's three instances
                             (examples/surgery_instances.py)
  qldpc_surgery/             CSS surgery on top of qldpc: graph-based gadgets (Graph-WY gauging,
                             Graph-IGND edge expansion), bridges for joint measurements,
                             Stim circuits for single / joint / batch PPMs,
                             Relay-BP + BP-LSD decoder and its batched two-stage driver
scripts/                     one script per result of the paper
  codes/                     code constructions and distances (Supplemental Material)
  shuttling/                 shuttling times and qubit counts (Fig. 3, Table S2)
  ler/l33/, ler/l9/          logical error rates at l = 33 (Fig. 4, code capacity) and l = 9
data/                        frozen inputs and published results (see data/README.md)
figures/                     the paper's figures
tests/
lp-surgery-aod-polynomials.html   the animation (served by GitHub Pages)
```

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Some scripts need extra tools: Node.js (shuttling times), a C++17 compiler `c++`
(exhaustive distance search) and GAP with QDistRnd (optional distance estimates).

## Reproducing the paper

Run from the repository root with `uv run python <script>`. Unless noted, each
script takes seconds and checks or prints the published numbers.

| Result | Script |
| --- | --- |
| Three instances: merged codes, ranks, readouts, exact distances 8 and 10 | `scripts/codes/verify_instances.py --exhaustive` |
| Worked examples l = 9, 15 (CRT packets, generators, adapters) | `scripts/codes/worked_examples.py` |
| CRT-packet selection, table of CRT unions for l = 33 | `scripts/codes/crt_packets.py` |
| Algebraic adapters with unequal lifts | `scripts/codes/adapters.py` |
| QDistRnd distance table | `scripts/codes/qdistrnd.py summarize` |
| Table S2 (shuttling and gate times) | `scripts/shuttling/extract_timings.py` |
| Graph-WY / Graph-IGND gadgets | `scripts/shuttling/build_graph_gadgets.py` |
| Fig. 3 qubit counts / figure | `scripts/shuttling/compute_space_overhead.py`, `scripts/shuttling/plot_fig3.py` |
| Fig. 4 (l = 33 LER by basis) | `scripts/ler/l33/plot_fig4.py` |
| Effective distances 15.9 and 22.8 | `scripts/ler/l33/fit_effective_distance.py` |
| Table of circuit-level failures by decoder stage | `scripts/ler/l33/circuit_error_table.py` |
| l = 9 LER figure | `scripts/ler/l9/plot_figure.py` |
| Code-capacity table | `scripts/ler/l33/summarize_code_capacity.py` |

The figure and table scripts read the published samples in `data/`. New samples
are drawn with `scripts/ler/l33/run_circuits.py`, `scripts/ler/l9/run_circuits.py` and
`scripts/ler/l33/run_code_capacity.py` (Relay-BP with BP-LSD fallback, settings in
`qldpc_surgery.decoding`), or on a cluster with `scripts/ler/l33/run_batched.py`, which
runs the two decoder stages as separate jobs per shard; for example

```bash
uv run python scripts/ler/l33/run_circuits.py --p-list 0.003 --shots 100 --out l33.csv
uv run python scripts/ler/l33/plot_fig4.py --stats l33.csv --out l33.pdf
```

The published data took hundreds of core-days on a cluster. The shuttling
animation is the source of all timings: its embedded JavaScript schedulers are
replayed by `scripts/shuttling/extract.cjs`.

## Tests

```bash
uv run pytest -m "not slow"
```

## Citation

If you use this code, please cite the paper above.

## License

MIT, see [LICENSE](LICENSE).
