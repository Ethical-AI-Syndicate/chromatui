# ChromaTUI Conformance Matrix (2026-02-23)

This matrix compares the requested architecture and algorithm contract against the current implementation.

Status legend:

- PASS: Implemented with concrete code path and tests
- PARTIAL: Implemented in simplified or incomplete form
- MISSING: No concrete implementation found

## 1) Architecture Conformance

| Area | Status | Evidence | Notes |
|---|---|---|---|
| Input layer (`TerminalSession` -> `Event`) | PASS | `crates/chromatui-core/src/backend.rs:49`, `crates/chromatui-core/src/backend.rs:97`, `crates/chromatui-core/src/types.rs:50` | Raw crossterm events map into typed `Event`. |
| Runtime loop (`update(Event) -> Cmd`) | PASS | `crates/chromatui-core/src/runtime.rs:5`, `crates/chromatui-core/src/runtime.rs:124`, `crates/chromatui-core/src/runtime.rs:133` | Model update contract exists and is exercised in loop. |
| Cmd -> effects execution | PASS | `crates/chromatui-core/src/runtime.rs:147`, `crates/chromatui-runtime/src/lib.rs:569` | Effect dispatcher implemented in both core and deterministic runtime layers. |
| Subscriptions stream | PASS | `crates/chromatui-runtime/src/lib.rs:666`, `crates/chromatui-runtime/src/lib.rs:727` | Runtime now multiplexes subscription-derived tick events with source events. |
| View -> frame -> diff -> presenter -> ANSI | PASS | `crates/chromatui-runtime/src/lib.rs:119`, `crates/chromatui-render/src/diff.rs:38`, `crates/chromatui-render/src/presenter.rs:12` | Frame is converted to `Buffer`, diffed as `BufferDiff`, then presented to ANSI bytes. |
| Output layer (`TerminalWriter`) | PASS | `crates/chromatui-core/src/backend.rs:198`, `crates/chromatui-core/src/backend.rs:227` | Inline and alt-screen output modes implemented. |
| One-writer rule | PASS | `crates/chromatui-core/src/backend.rs:6`, `crates/chromatui-core/src/backend.rs:205`, `crates/chromatui-core/src/backend.rs:280` | Enforced via global atomic claim with unit tests. |

## 2) "Alien Artifact" Algorithm Conformance

| Algorithm | Status | Evidence | Notes |
|---|---|---|---|
| Bayesian fuzzy scoring (command palette) | PASS | `crates/chromatui-algorithms/src/lib.rs:1`, `crates/chromatui-widgets/src/lib.rs:662` | Bayesian scorer is integrated through a widget-level command palette with evidence ledgers. |
| Bayesian hint ranking (keybinding hints) | PASS | `crates/chromatui-algorithms/src/lib.rs:921`, `crates/chromatui-widgets/src/lib.rs:572` | Bayesian utility/cost/VOI ranker with hysteresis is implemented and wired through widget-level keybinding hint APIs. |
| Bayesian diff strategy selection | PASS | `crates/chromatui-runtime/src/lib.rs:236`, `crates/chromatui-runtime/src/lib.rs:237` | Selector uses beta posterior with dirty-scan cost model and p95 risk-aware decisioning. |
| Bayesian capability detection (terminal caps probe) | PASS | `crates/chromatui-core/src/backend.rs:183`, `crates/chromatui-core/src/backend.rs:213` | Capability probe now combines env/size/background evidence through log-BF posteriors. |
| Dirty-span interval union | PASS | `crates/chromatui-algorithms/src/lib.rs:760`, `crates/chromatui-algorithms/src/lib.rs:792`, `crates/chromatui-algorithms/src/lib.rs:818` | Dirty span tracking and span merge logic implemented. |
| Summed-area table tile skip | PASS | `crates/chromatui-runtime/src/lib.rs:150`, `crates/chromatui-algorithms/src/lib.rs:853` | Frame pipeline computes SAT-backed density and uses it to force redraw for dense-tile frames. |
| Fenwick tree virtualization support | PASS | `crates/chromatui-widgets/src/lib.rs:612`, `crates/chromatui-widgets/src/lib.rs:632` | Virtual height model uses Fenwick prefix sums for fast estimated offsets. |
| Bayesian height prediction + conformal bounds | PASS | `crates/chromatui-algorithms/src/lib.rs:1624`, `crates/chromatui-widgets/src/lib.rs:612` | Bayesian mean predictor with Welford variance and conformal intervals is implemented and used by widget virtual height modeling. |
| BOCPD resize regime detection | PASS | `crates/chromatui-algorithms/src/lib.rs:190`, `crates/chromatui-runtime/src/lib.rs:319`, `crates/chromatui-runtime/src/lib.rs:343` | BOCPD-like detector and runtime coalescer integration are present. |
| Bayes-factor evidence ledger (resize) | PASS | `crates/chromatui-runtime/src/lib.rs:395`, `crates/chromatui-runtime/src/lib.rs:409` | Resize LBF now uses explicit likelihood-ratio evidence with exponential inter-arrival models. |
| VOI sampling for expensive ops | PASS | `crates/chromatui-runtime/src/lib.rs:141`, `crates/chromatui-runtime/src/lib.rs:145` | Runtime now uses VOI sampler to trigger expensive diff sampling under uncertainty/interval rules. |
| E-process anytime-valid testing | PASS | `crates/chromatui-runtime/src/lib.rs:599`, `crates/chromatui-runtime/src/lib.rs:602` | Runtime integrates e-process wealth into sequential alerting decisions. |
| Conformal alerting | PASS | `crates/chromatui-runtime/src/lib.rs:605`, `crates/chromatui-runtime/src/lib.rs:608` | Runtime emits per-frame conformal alert flags from residual thresholds. |
| Mondrian conformal frame-time risk gating | PASS | `crates/chromatui-algorithms/src/lib.rs:1207`, `crates/chromatui-runtime/src/lib.rs:542` | Bucketed conformal gate with fallback hierarchy is implemented and integrated into runtime frame evidence. |
| CUSUM control charts | PASS | `crates/chromatui-runtime/src/lib.rs:598`, `crates/chromatui-runtime/src/lib.rs:602` | Runtime uses CUSUM and combines it with e-process evidence for dual gating. |
| CUSUM hover stabilizer | PASS | `crates/chromatui-algorithms/src/lib.rs:1489`, `crates/chromatui-widgets/src/lib.rs:607` | Hover CUSUM stabilizer is implemented and wired through widget hover-target tracking APIs. |
| Damped spring dynamics | PASS | `crates/chromatui-algorithms/src/lib.rs:702`, `crates/chromatui-algorithms/src/lib.rs:712`, `crates/chromatui-algorithms/src/lib.rs:730` | Spring now supports stable large-`dt` subdivision via bounded-step integration. |
| Easing curves + stagger distributions | PASS | `crates/chromatui-algorithms/src/lib.rs:1456`, `crates/chromatui-algorithms/src/lib.rs:1485` | Easing-based stagger with deterministic xorshift jitter and clamping is implemented. |
| Sine pulse attention cue | PASS | `crates/chromatui-algorithms/src/lib.rs:1436` | Dedicated half-cycle sine pulse primitive is implemented. |
| Perceived luminance probe | PASS | `crates/chromatui-core/src/backend.rs:194`, `crates/chromatui-core/src/backend.rs:239` | Luminance probe is wired into terminal capability detection and dark-mode preference inference. |
| Jain fairness input guard | PASS | `crates/chromatui-algorithms/src/lib.rs:1335`, `crates/chromatui-runtime/src/lib.rs:580` | Jain fairness metric and guard exist; runtime now evaluates fairness and yields when guard triggers. |

## 3) Test Evidence

Validation command run:

- `cargo test` in `/home/mike/chromatui`

Observed result:

- Workspace tests pass across all crates (`chromatui-*`) with zero failures.

Important limitation:

- Automated tests and trace captures provide strong evidence, but physical terminal emulator variance (font rendering, compositor latency, platform-specific escape handling) should still be checked in target deployment environments.

## 4) Interactive Visual Validation Evidence

Validation command run:

- `cargo run -p chromatui-runtime --example visual_validation`

Observed evidence (terminal trace capture):

- `VISUAL_VALIDATION_V1`
- `inline_ansi=\r\x1b[1;1Hframe-01        \x1b[2;1H                \n\r\x1b[1;1Hframe-02        \x1b[2;1H                \n\r\n`
- `alt_ansi=\x1b[1;1Hframe-01        \x1b[2;1H                `
- `inline_report=step=1 ... strategy=FullRedraw ... step=2 ... strategy=FullRedraw ... step=3 ... strategy=DirtyRow ...`
- `alt_report=step=1 ... strategy=FullRedraw ... step=2 ... event=Resize(16, 2) strategy=DirtyRow ...`
- `capabilities=alt:0.8638,truecolor:0.6000,dark:0.5000,prefers_dark:true`

Evidence source:

- `crates/chromatui-runtime/examples/visual_validation.rs`

## 5) Benchmark and Proof Evidence

Validation commands run:

- `cargo bench -p chromatui-render --no-run`
- `cargo bench -p chromatui-render --bench diff_bench -- --sample-size 10`

Observed benchmark evidence:

- `diff/identical_100x50   time: [179.05 us 181.27 us 183.00 us]`
- `diff/sparse_5pct_100x50 time: [242.69 us 247.55 us 255.01 us]`
- `diff/dense_100x50       time: [233.12 us 245.82 us 253.44 us]`

Proof-oriented and deterministic validation additions:

- `crates/chromatui-render/src/no_flicker_proof.rs` includes randomized theorem checks for diff completeness.
- `crates/chromatui-core/src/runtime.rs` includes 16-byte packed-cell compile-time size assertion (`PackedCell16`).
- `crates/chromatui-render/src/presenter.rs` includes OSC-8 hyperlink encoding tests for linked runs.

Additional subsystem evidence:

- Modal stack + time-travel recorder: `crates/chromatui-widgets/src/lib.rs`
- Synchronized output brackets (DEC 2026): `crates/chromatui-core/src/backend.rs`
- PID/PI, MPC objective, Count-Min, W-TinyLFU, PAC-Bayes, scheduling math, and expanded FX equations: `crates/chromatui-algorithms/src/lib.rs`
