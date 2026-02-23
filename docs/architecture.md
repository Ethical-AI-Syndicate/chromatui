# ChromaTUI Architecture and "Alien Artifact" Algorithms

This document is the canonical architecture and algorithm reference for ChromaTUI.

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                 INPUT LAYER                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ TerminalSession (crossterm)                                                  │
│   └─ raw terminal events  ->  Event (chromatui-core)                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                        |
                                        v
┌──────────────────────────────────────────────────────────────────────────────┐
│                                RUNTIME LOOP                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Program / Model (chromatui-runtime)                                          │
│   update(Event) -> (Model', Cmd)                                             │
│   Cmd -> Effects                                                             │
│   Subscriptions -> Event stream (tick / io / resize / ...)                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                        |
                                        v
┌──────────────────────────────────────────────────────────────────────────────┐
│                               RENDER KERNEL                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ view(Model) -> Frame -> Buffer -> BufferDiff -> Presenter -> ANSI            │
│                 (cell grid)    (minimal)       (encode bytes)                │
└──────────────────────────────────────────────────────────────────────────────┘
                                        |
                                        v
┌──────────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT LAYER                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ TerminalWriter                                                               │
│   inline mode (scrollback-friendly)  |  alt-screen mode (classic)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Frame Pipeline

1. Input: `TerminalSession` reads `Event`
2. Model: `update()` returns `Cmd` for side effects
3. View: `view()` renders into `Frame`
4. Buffer: `Frame` writes cells into a 2D buffer
5. Diff: `BufferDiff` computes minimal changes
6. Presenter: emits ANSI with state tracking
7. Writer: enforces one-writer rule, flushes output

This loop must remain deterministic and flicker-free.

## Bayesian Fuzzy Scoring (Command Palette)

Posterior odds are computed as:

```text
P(relevant | evidence) / P(not_relevant | evidence)
  = [P(relevant) / P(not_relevant)] * Product_i BF_i

BF_i = P(evidence_i | relevant) / P(evidence_i | not_relevant)
```

Prior odds by match type:

- Exact: 99:1
- Prefix: 9:1
- Word-start: 4:1
- Substring: 2:1
- Fuzzy: 1:3

Evidence factors:

- Word boundary bonus (`BF ~= 2.0`)
- Position bonus (`BF ~ 1/position`)
- Gap penalty (`BF < 1.0`)
- Tag match bonus (`BF ~= 3.0`)
- Length factor (`BF ~ 1/length`)

Every result should include an explainable evidence ledger.

## Bayesian Hint Ranking (Keybinding Hints)

```text
U_i ~ Beta(alpha_i, beta_i)
E[U_i] = alpha_i / (alpha_i + beta_i)
VOI_i = sqrt(Var(U_i))

V_i = E[U_i] + w_voi * VOI_i - lambda * C_i
```

Use hysteresis: only swap hint `i` over `j` if `V_i - V_j > epsilon`.

## Bayesian Diff Strategy Selection

Change-rate posterior:

```text
p ~ Beta(alpha, beta)
Prior: alpha0 = 1, beta0 = 19

alpha <- alpha * decay + N_changed
beta  <- beta  * decay + (N_scanned - N_changed)
decay = 0.95
```

Cost model:

```text
Cost = c_scan * cells_scanned + c_emit * cells_emitted

FullDiff:   c_row * H + c_scan * D * W + c_emit * p * N
DirtyRow:   c_scan * D * W + c_emit * p * N
FullRedraw: c_emit * N
```

Decision: choose `argmin(E[Cost_*])`. Use conservative `p95(p)` when posterior variance is high.

## Bayesian Capability Detection (Terminal Caps Probe)

```text
log BF = ln(P(data | feature) / P(data | not_feature))

logit P(feature | evidence)
  = logit P(feature) + Sum_i log BF_i
```

Then:

```text
P = 1 / (1 + exp(-logit))
```

## Sparse and Hierarchical Diff Algorithms

### Dirty-Span Interval Union

For row `y`, dirty spans are `S_y = union_k [x0_k, x1_k)`.

Scan cost is `Sum_y |S_y|`.

### Summed-Area Table (Tile-Skip Diff)

```text
SAT(x,y) = A(x,y)
         + SAT(x-1,y) + SAT(x,y-1) - SAT(x-1,y-1)
```

Tile sums are O(1), so empty tiles can be skipped deterministically.

## Fenwick Tree for Virtualized Lists

```text
sum(i) = Sum_{k=1..i} a_k
update(i, d): for (j=i; j<=n; j+=j&-j) tree[j] += d
query(i):     for (j=i; j>0; j-=j&-j)  sum += tree[j]
```

Enables O(log n) prefix-sum updates and queries.

## Bayesian Height Prediction + Conformal Bounds

```text
Prior:     mu ~ N(mu0, sigma0^2 / kappa0)
Posterior: mu_n = (kappa0*mu0 + n*x_bar) / (kappa0 + n)

Conformal interval:
[mu_n - q_{1-alpha}, mu_n + q_{1-alpha}]
```

Track variance online with Welford's algorithm.

## BOCPD for Resize Coalescing

Observation model on inter-arrival times:

- Steady: Exponential(mean ~= 200 ms)
- Burst: Exponential(mean ~= 20 ms)

Run-length recursion uses geometric hazard:

```text
H(r) = 1 / lambda_hazard,  lambda_hazard = 50
```

Classify regime by burst posterior:

- `p_burst > 0.7`: Burst
- `p_burst < 0.3`: Steady
- otherwise: Transitional

## Bayes-Factor Evidence Ledger (Resize Coalescer)

```text
LBF = log10(P(evidence | apply_now) / P(evidence | coalesce))
```

Interpretation:

- `LBF > 0`: apply now
- `LBF < 0`: coalesce
- `|LBF| > 1`: strong
- `|LBF| > 2`: decisive

## Value-of-Information (VOI) Sampling

```text
p ~ Beta(alpha, beta)

VOI = variance_before - E[variance_after]
sample iff (max_interval exceeded) OR (VOI * value_scale >= sample_cost)
```

Default tuning:

- `prior_alpha=1.0`
- `prior_beta=9.0`
- `max_interval_ms=1000`
- `min_interval_ms=100`
- `sample_cost=0.08`

## E-Process (Anytime-Valid Testing)

```text
W_t = W_{t-1} * (1 + lambda_t * (X_t - mu0))
```

Guarantee:

```text
P(exists t: W_t >= 1/alpha) <= alpha
```

## Conformal Alerting

```text
R_t = |observed_t - predicted_t|
q   = quantile_{(1-alpha)(n+1)/n}(R_1, ..., R_n)

P(R_{n+1} <= q) >= 1 - alpha
```

## Mondrian Conformal Frame-Time Risk Gating

Residuals and upper bound:

```text
r_t = y_t - y_hat_t
y_hat_t^+ = y_hat_t + q_{1-alpha}(|r|)
```

Risk triggers when `y_hat_t^+ > budget`.

Bucket fallback:

1. `(mode, diff, size)`
2. `(mode, diff)`
3. `(mode)`
4. global default

## CUSUM Control Charts

```text
S_t = max(0, S_{t-1} + (X_t - mu0) - k)
alert when S_t > h
```

Dual alerting policy:

- alert iff (CUSUM detects and e-process confirms)
- or e-process alone exceeds `1/alpha`

## CUSUM Hover Stabilizer

```text
S_t = max(0, S_{t-1} + d_t - k)
switch if S_t > h
```

Suppresses boundary jitter while preserving intentional hover switches.

## Damped Spring Dynamics (Animation)

```text
F = -k(x - x*) - c v
x'' + c x' + k(x - x*) = 0

critical damping: c_crit = 2 * sqrt(k)
```

Use semi-implicit Euler and subdivide large `dt` for stability.

## Easing Curves and Stagger

```text
ease_in(t)  = t^2
ease_out(t) = 1 - (1 - t)^2
ease_in_out(t):
  2t^2                    if t < 0.5
  1 - (-2t + 2)^2 / 2     otherwise

offset_i = D * ease(i / (n - 1))
```

Optional deterministic jitter uses xorshift with clamping.

## Sine Pulse Attention Cue

```text
p(t) = sin(pi * t),  t in [0, 1]
```

Produces smooth `0 -> 1 -> 0` emphasis.

## Perceived Luminance

```text
Y = 0.299R + 0.587G + 0.114B
```

Feeds dark/light classification for terminal defaults.

## Input Fairness Guard (Jain Index)

```text
F(x1, ..., xn) = (Sum x_i)^2 / (n * Sum x_i^2)
```

Interpretation:

- `F = 1.0`: perfect fairness
- `F = 1/n`: one source dominates

Intervention trigger:

```text
if input_latency > threshold OR F < 0.8:
    force_coalescer_yield()
```

## Core Algorithms and Data Structures

ChromaTUI is not treated as a generic widget toolkit. It is engineered around terminal-specific cost models, deterministic update loops, and low-overhead data structures.

Use `docs/conformance-matrix.md` as the source of truth for what is currently implemented as `PASS` versus still under hardening.

### Math-Driven Performance

The operating principle is to spend small, bounded compute on principled decisions that avoid larger downstream costs (extra scans, ANSI bytes, redundant renders, unstable ranking, or jitter).

## Additional Math Systems (Reference Catalog)

### Presenter Cost Modeling (Cursor/Byte Economy)

```text
cost = c_scan * N_scan + c_emit * N_emit
```

Choose emission strategy by minimizing byte and cursor movement cost (sparse runs vs merged runs).

### BOCPD Run-Length Posterior + Hazard (Core Form)

```text
H(r) = 1/lambda

P(r_t = 0 | x_1:t)
  proportional to Sum_r P(r_{t-1}=r) * H(r) * P(x_t | r)

P(r_t = r | x_1:t)
  proportional to P(r_{t-1}=r-1) * (1 - H(r-1)) * P(x_t | r)
```

### GRAPA (Adaptive Betting Fraction)

```text
lambda_{t+1} = lambda_t + eta * gradient_lambda log(W_t)
```

Used to adapt e-process sensitivity while preserving anytime-valid behavior.

### PID / PI Frame Pacing

```text
u_t = Kp * e_t + Ki * Sum e_t + Kd * Delta e_t
```

PI is the default practical controller in most terminal workloads.

### MPC Evaluation Objective

```text
min_{u_t:t+H} Sum_{k=0..H} |y_{t+k} - y*|^2 + rho * |u_{t+k}|^2
```

MPC is evaluated as a benchmark control strategy against PI to validate tradeoffs.

### Count-Min Sketch (Approximate Frequency)

```text
f_hat(x) = min_j C[j, h_j(x)]
```

### PAC-Bayes Calibration (Bound Tightening)

```text
E[err] <= err_bar + KL(q || p) / (2n)
```

### Scheduling Math (Smith's Rule + Aging)

```text
priority = w/r + a * wait
```

Improves throughput while preventing starvation.

## Visual FX Math (Reference)

These equations are the deterministic reference for visual effects modules.

| Effect | Core Equation | Output |
|---|---|---|
| Metaballs | `F(x,y)=Sum_i r_i^2/((x-x_i)^2+(y-y_i)^2), render F>=tau` | Smooth blob fields |
| Plasma | `v=(1/6) Sum_{k=1..6} sin(phi_k(x,y,t))` | Interference bands |
| Gray-Scott | `du/dt=Du Lap(u)-u v^2+F(1-u)`, `dv/dt=Dv Lap(v)+u v^2-(F+k)v` | Reaction-diffusion patterns |
| Clifford attractor | `x_{t+1}=sin(a y_t)+c cos(a x_t)`, `y_{t+1}=sin(b x_t)+d cos(b y_t)` | Chaotic filaments |
| Mandelbrot/Julia | `z_{n+1}=z_n^2+c` | Fractal boundaries |
| Lissajous/Harmonograph | `x=A sin(a t+delta)`, `y=B sin(b t+phi)` | Phase curves |
| Flow field | `v=(cos 2piN, sin 2piN)`, `p_{t+1}=p_t+v dt` | Particle ribbons |
| Wave interference | `I(x,t)=Sum_i sin(k_i |x-s_i| - omega_i t)` | Ripple superposition |
| Spiral galaxy | `r=a e^{b theta}`, `theta(t)=theta_0+omega t` | Spiral starfields |
| Spin lattice (LLG) | `dS/dt=-SxH-alpha Sx(SxH)` | Domain dynamics |

## Cell Layout and Diff Mechanics

### 16-Byte Cell Target

The renderer is designed around a 16-byte logical cell budget to optimize cache locality and equality checks.

```text
CellContent (4B) | fg (4B) | bg (4B) | attrs (2B) | link_id (2B)
```

### Block-Based Diff

Diffing is row-first with fast-skip on identical rows and coalesced runs for changed segments.

```text
for each row:
  if rows_equal(old, new): continue
  scan blocks
  coalesce adjacent changes into runs
```

## Runtime Architecture Pattern (Elm-Style)

ChromaTUI uses an update/view command loop with explicit side effects and subscriptions.

```text
Event -> update(Model) -> Cmd
Model -> view() -> Frame -> BufferDiff -> Presenter
Cmd -> Effects -> Event stream
```

## Safety and Correctness Standards

- Prefer `#![forbid(unsafe_code)]` in core runtime/render crates.
- Use saturating/checked arithmetic for coordinates and dimensions.
- Preserve no-flicker guarantees through deterministic frame boundaries and synchronized output behavior.
- Keep proof-oriented regression tests for diff completeness, dirty-tracking soundness, and deterministic presenter output.

## Performance and Validation Notes

- Dirty-row tracking is O(1) on mutation and reduces scan work in sparse frames.
- Grapheme pooling and deduplication reduce repeated width/cell overhead.
- Synchronized output mode prevents partial-frame visibility.
- Use benchmark and snapshot suites to verify throughput, correctness, and render stability over time.
