# nanoACE DEVLOG

A running log of design decisions and their rationale. The *why* matters as much as
the *what* — this file exists so a human or coding agent can reconstruct the reasoning
behind the code without spelunking git history.

Reference: Chang et al. (2025), *Amortized Probabilistic Conditioning for Optimization,
Simulation and Inference* (AISTATS 2025). Paper markdown lives in `paper/`.

---

## 2026-06-06 — Prior representation: candidate redesign (idea, not yet implemented)

The shipped prior encoder is the histogram→MLP (`Tokens.prior: [B, T, prior_bins]`,
`prior_embed = _mlp(prior_bins, ...)`). It works but is the wrong default for nanoACE's
scope: it spends a 64-dim, discretization-noisy vector to encode what is almost always two
numbers, and it drags in scale coupling — `Variable.prior_range`, per-variable `prior_bins`,
and the global-`prior_bins` validation in `ACE.__init__` exist only to pin the grid. It is
also what made the Gaussian toy harder, which is why runtime priors were removed from that
example. These are candidate replacements, recorded so the decision is not relitigated from
scratch. **None is implemented yet; the histogram is still what is in the code.**

- **Preferred candidate: a rescaled Beta prior token.** Encode a continuous latent's prior
  as a Beta on the latent's bounded range, user-facing `(mean, SD)`. This fits the toys
  specifically because their latents are bounded-uniform by construction (`MU_RANGE`,
  `LOGSIG_RANGE`, `LOG_LENGTHSCALE_RANGE`, `LOG_OUTPUTSCALE_RANGE`), so a bounded prior
  matches the true support and the no-information case is exactly `Beta(1,1) = uniform`
  (today's generative prior). Two numbers buy uniform, Gaussian-like bumps, skew (`α≠β`),
  and edge-peaked/U-shaped priors (`α,β<1`).
- **Parameterization: user `(mean, SD)`, internal `(μ, ν)`, location embedded like a value.**
  User interface `(mean, SD)` for interpretability; sample/store internally as `(μ, ν)` with
  concentration `ν = α + β` — constraint-free, since every `μ∈(0,1), ν>0` is a valid Beta.
  For the embedding, feed the prior **mean directly** (rescaled to the latent's range / the
  `[-1,1]` convention), ideally through the existing `value_embed` so a prior's location
  enters exactly like an observed value, and carry the **concentration** as a separate
  channel. `log ν` is the natural concentration coordinate (a positive multiplicative scale
  spanning ≈2 to 1000); **`logit μ` is not** worth it — `μ` is a bounded location with no
  positivity or scale argument, and `d/dμ logit(μ) = 1/(μ(1−μ))` over-resolves the
  degenerate endpoints while compressing the mid-range where informative priors usually sit.
  (For the value-limit variant below, carry concentration as the `σ`-gated term instead of
  raw `log ν`.) Recover shape params when needed: with variance `v`, `ν = μ(1−μ)/v − 1`,
  `α = μν`, `β = (1−μ)ν`, valid iff `σ² < μ(1−μ)` (the Bernoulli-SD cap); clamp/validate the
  user's `(μ, σ)` against that bound.
- **Coordinate convention (avoid space confusion).** Three spaces: the user specifies
  `(mean, SD)` in the latent's **original** range `[a,b]`; the **Beta math** (recover
  `α,β`/`ν`, check validity `SD² < μ(1−μ)`) lives in **unit `[0,1]`**; the **embedding** uses
  the `[-1,1]` convention. Maps are affine (width `w = b−a`): to unit, `μ_u = (m−a)/w`,
  `SD_u = SD/w`; to `[-1,1]`, `m_± = 2μ_u − 1`, `SD_± = 2·SD_u` (mean through the affine map,
  SD times the slope). Crucially **`ν` is coordinate-free** — convert `SD ↔ ν` *once* in unit
  space, then carry `(location, ν)` and only ever transform the *location*; `ν` never changes
  (equivalently `SD_±² = (1−m_±²)/(ν+1)`). The prior mean in `[-1,1]` lands in the same
  coordinate as a VALUE token for that latent, which is what makes the `value_embed` reuse
  coherent. Worked example: `[a,b]=[-2,8]`, user `mean=2, SD=2` → unit `μ=0.4, SD=0.2` (not
  `0.5/0.25`) → `Beta(2,3)`, `ν=5` → `[-1,1]` location `−0.2`, SD `0.4`.
- **Known-value as a limit, done honestly.** An earlier claim that a prior token "collapses
  to a VALUE token" as spread→0 is *not* enforced by the current representation: the spread
  MLP has no zero limit, and the additive `mode_embed(PRIOR)` vs `mode_embed(VALUE)` is a
  constant offset that never closes. To make it real you must (a) gate the spread so it
  vanishes by construction — e.g. payload `value_embed(mean) + σ·h(mean, σ)`, feeding `σ`
  not `log σ` so the input stays bounded — and (b) treat a known latent as the zero-spread /
  `ν→∞` prior token rather than a separately-moded VALUE token, so "value" *is* the boundary
  of "prior." With Beta this falls out naturally: `ν→∞` is a spike at `μ`. Even then this is
  representation continuity only; the model behaves like exact conditioning in the limit only
  if training visits small spreads.
- **Discrete latents stay separate.** Beta covers continuous bounded latents. The discrete
  `kernel` latent's runtime prior is a categorical pmf — a small "K-probabilities" prior
  token, not a Beta.
- **Training implication.** Sample a Beta hyperprior `(α, β)`, draw the true latent from it,
  build the prior token, NLL as usual. This is much simpler than the paper's
  MoG/geometric-K/Dirichlet prior-generating machinery (already slated for slimming) and is
  the actual distribution the model would amortize over.
- **Default hyperprior over `(μ, ν)`.** The prior-generating mixture, defined in unit
  `(μ, ν)` space so it is range-agnostic and reused across all continuous latents:
  - `1/3`: `μ = 0.5, ν = 2` → exactly `Beta(1,1)` (uniform; the uninformative case).
  - `1/6`: `μ ~ U(0,1), ν ~ U(0.1, 2)` → broad / skewed / U-shaped (`ν<1`).
  - `1/2`: `μ ~ U(0,1), log ν ~ U(log 2, log 1000)` → concentrated, up to near-spike.

  Clamp `μ` to `(ε, 1−ε)` so `α,β > 0`. Weights and bounds are tunable. The heavy `1/2`
  mass on concentrated priors (with `ν` to 1000) is deliberate — it is where prior
  conditioning matters most and where the `ν→∞ ≈ known-value` limit is learned. The `1/6`
  U-shaped band is the most exotic slice and the first dial to turn if it hurts (raise
  `ν_min`, e.g. 0.1→0.5, or lower the weight). Note `log ν ~ U(2, 1e3)` taken literally
  overflows (`e^1000`); the intended range is `log ν ~ U(log 2, log 1000)`.
- **ACEP always emits prior tokens; "no prior" = uniform.** The architecture does not (and
  need not) enforce "absent prior token ≡ uniform prior," so do not train over a
  present/absent mixture. The prior-conditioning model (ACEP) always carries one prior token
  per continuous latent, using `Beta(1,1)` when there is no information. This is benign here
  because the latents are bounded-uniform by construction, so the uniform token is the honest
  "no extra info" prior rather than the hard amortization burden that arbitrary histograms
  were. Prior-free ACE and always-prior ACEP are then simply two variants with fixed token
  layouts (supersedes the earlier idea of a separate `prior_prob` present/absent knob).
- **Fallbacks if more expressiveness is wanted.** Plain moments `(mean, log_std)` if even
  Beta is more than needed (but it cannot express bounded support or skew, and a Gaussian
  encoder leaks past the latent range); a small fixed quantile vector (e.g. 5/25/50/75/95th
  percentiles) for asymmetric unimodal priors; a handful of orthogonal-basis coefficients
  (Hermite/Legendre) only if arbitrary/multimodal priors become an explicit demo. All three
  still beat a 64-bin histogram on compactness and trainability.
- **What adopting any of these deletes.** `Tokens.prior` shrinks from `[B, T, prior_bins]`
  to a 2–3 vector; `Variable.prior_range`, per-variable `Variable.prior_bins`, and the
  `prior_bins != cfg.prior_bins` check all go away (a fixed small vector is never ragged);
  "no prior" becomes "do not emit a PRIOR token" instead of an encoded uniform histogram.

---

## 2026-06-06 — GP-1D executable example

- **Standalone GP file.** `gp1d.py` owns the GP regression example end to end:
  online function sampling, command-line training/evaluation, checkpoint helpers, and a
  fixed diagnostic plot. It follows the same local-file pattern as `gaussian_toy.py`
  rather than introducing a generic training framework.
- **Latents.** The example uses two continuous latents, `log_lengthscale` and
  `log_outputscale`, plus one discrete `kernel` latent with four classes: RBF,
  Matern-1/2, Matern-3/2, and periodic. This exercises the shared continuous MDN head,
  the shared `Kmax` categorical head, and discrete value embeddings when a kernel latent
  is occasionally revealed as context.
- **Sampler.** GP functions are generated online. Kernel matrices and Cholesky sampling
  run on CPU float64, then sampled tensors move to the selected ACE device. There is no
  sharded data layer yet; add one only when a real GP training run needs saved pools.
  Function values are not clipped: with `log_outputscale` up to `log(1.0)`, sampled `y`
  values can naturally leave the loose `[-1, 1]` convention used by the embedders. Treat
  that as a calibration caveat, not a bug. The default Cholesky jitter is `1e-5`; increase
  it if clustered x-locations ever make Matern-1/2 or periodic kernel matrices unstable.
- **Diagnostic.** GP-1D now computes a numerical grid oracle for the fixed context.
  For each kernel and each `log_lengthscale`/`log_outputscale` grid point, it evaluates
  the GP marginal likelihood of the observed context, applies uniform-prior quadrature
  weights, and normalizes over the full kernel-by-hyperparameter grid. The diagnostic
  reports per-kernel integrated marginal likelihood deltas, the kernel posterior,
  continuous latent marginals, and the posterior predictive moments formed by mixing
  conditional GP predictives over the grid. This is not a closed-form analytic posterior;
  it is an oracle up to grid resolution, the bounded hyperparameter ranges, and the
  Cholesky jitter used by the sampler. The fixed context locations are irregular and
  clustered, including nearby pairs/triples, because sparse evenly spaced points do not
  say much about local roughness and make kernel/lengthscale inference mostly guesswork.
- **Current retained GP artifact.** The retained `artifacts/gp1d.pt` / `artifacts/gp1d.png`
  pair is a 100k-step online run. On the fixed diagnostic, the generating kernel is
  Matern-3/2, but the numerical oracle gives posterior mass roughly RBF 0.70,
  Matern-1/2 0.02, Matern-3/2 0.28, periodic 0.00; the checkpoint gives roughly RBF
  0.74, Matern-1/2 0.02, Matern-3/2 0.24, periodic 0.00. The oracle-vs-ACE kernel KL is
  about 0.01, and the predictive RMSEs are similar. The current artifact is useful for
  checking the machinery, but it should not be treated as evidence that the training
  setup has converged in general.

---

## 2026-06-06 — Gaussian toy implementation

- **Core implementation.** `ace.py` now contains the current source-of-truth
  implementation: `Variable`, `Tokens`, `Batch`, separated context/target transformer
  blocks, shared continuous MDN head, shared masked `Kmax` categorical head, prediction
  object, target NLL loss, and `sample_ar`.
- **Executable Gaussian toy file.** `gaussian_toy.py` owns the Gaussian example end to end:
  online batch generation, command-line training/evaluation, oracle math, checkpoint
  helpers, and plotting. There is no separate generic command-line wrapper because that
  wrapper would still depend on the Gaussian file to be understood. Reusable grid query
  helpers live in `diagnostics.py`.
- **Gaussian diagnostic.** `gaussian_toy.py` trains the Gaussian toy online under fixed latent priors,
  compares ACE marginals to an analytic grid posterior, and includes an autoregressive
  two-latent joint diagnostic. The Gaussian example no longer injects random runtime prior
  tokens: amortizing over arbitrary prior histograms made the small Gaussian example much
  harder than the posterior-learning behavior this example is meant to inspect. Runtime
  prior tokens remain in the core ACE representation for examples that explicitly train prior
  conditioning. The AR diagnostic averages the two
  factorizations `p(mu)p(log_sigma|mu)` and `p(log_sigma)p(mu|log_sigma)` in probability
  space so the displayed joint posterior is not tied to one arbitrary order. The
  deterministic evaluation batch is defined in `gaussian_toy.py`: three observed `y` values,
  plus latent truth for printed diagnostics. Keeping that batch constant makes plots
  comparable across runs without a separate evaluation-case selector.
  The plot includes a posterior predictive panel for a new `y`: the analytic curve is the
  posterior mixture `sum p(mu, log_sigma | D) Normal(y | mu, sigma)`, not a Gaussian
  plug-in approximation.
  Training now mixes in latent-value context cases (`--latent-context-prob`, default
  `0.25`): one latent is sometimes revealed as a `VALUE` token and the other remains a
  target. Without that, AR conditional queries are syntactically valid but
  out-of-distribution for the toy model.
- **Artifacts are optional and regenerable.** `gaussian_toy.py` can save a diagnostic plot
  and a lightweight checkpoint under `artifacts/`, but these are convenience outputs
  rather than load-bearing repository assets. `--eval-only --load-checkpoint` verifies a
  saved checkpoint.
- **Environment.** The project currently pins the PyTorch CUDA wheel tested on this
  workstation: `torch==2.11.0+cu128`, PyTorch CUDA runtime 12.8, and the RTX 4060 Laptop
  GPU.

---

## 2026-06-06 — Initial design

### Vision

- **What it is.** A single-purpose, legible implementation of ACE that a coding agent
  can read end-to-end and extend. The optimization target is the *source* — legibility
  and extensibility — not a packaged runtime artifact (pretrained weights, a stable
  calling contract, an MCP tool). That packaging is outside the current scope, not an
  anti-goal: a clean `condition().predict()` surface is part of legible source anyway,
  so doing this well keeps that path open rather than foreclosing it.
- **What "nano" means.** Minimal *concepts* and maximal *locality* — not minimal lines,
  and not a trivial toy. nanoACE should be legible **and real** (you can train real
  models with it), exactly as nanoGPT is small/readable yet genuinely trains GPT-2.
  The bar a feature must clear is conceptual surface, not compute.

### Guiding principles

- **Paper-faithful starting point, not paper-bound math.** Start from the paper where it
  preserves ACE's core idea, but treat the equations as an existence proof rather than a
  constraint. The invariants are the conditioning semantics: variables as tokens, data /
  latent / prior information in one context set, target tokens that request predictive
  distributions, and a type-agnostic `dist.log_prob` training path. Attention details,
  output-head parameterization, prior features, latent transforms, and training mixtures
  can change when a tweak is simpler, more robust, or makes the conditioning interface
  clearer.
- **Tiebreaker when simplicity vs. performance conflict.** Simplicity is the default. A
  feature *can* buy its way in if it improves predictions or real-training throughput —
  but it pays rent in legibility (inline shape contracts, an obvious why). Large
  experiment-management complexity (clusters, multi-seed pools) is outside this
  repository's scale target.
- **"naïve" is not a synonym for "legible."** Evaluate case by case (see Attention).

### Model / architecture

- **Discrete in the core from day one.** A `Variable` schema (type, cardinality; ~8
  lines) travels with each token. Discrete value embedding uses an embedding matrix
  (`E_val`); discrete output uses one shared categorical head with `Kmax =
  max(cardinality)`. At distribution construction, mask logits beyond the target
  variable's cardinality and read only local labels `0..k-1`. Rationale: the paper's
  headline is "unifies discrete + continuous," and this keeps the code path shared rather
  than introducing per-variable heads.
- **Heads return distribution objects** (`.log_prob` / `.sample`). The training loss is
  then `-dist.log_prob(z)` for *everything*, so the loop is type-agnostic — the GMM vs.
  categorical distinction never reaches `train.py`.
- **Embedder (paper-shaped default).**
  - data:         `f_x(x) + f_val(y) + e_data`
  - latent:       `f_val(θ_l) + e_l`
  - prior-latent: `f_prob(p_l) + e_l`
  - target value branch replaced by the learned unknown embedding `e_?`.
  - discrete `f_val` → embedding matrix `E_val`.
- **Attention: separated context self-attn + target→context cross-attn. No mask.**
  This *is* the paper's choice (§3.2, `O(N²+NM)`), and it's both cheaper and more
  legible than the alternative. The "naïve" single masked stream materializes the full
  `(N+M)²` matrix and masks ~half away, *and* buries the key invariant ("targets read
  context, never each other") inside a fiddly block mask. The separated version needs no
  mask — the conditioning structure falls out of which sets you pass in:
  ```
  ctx = ctx + self_attn(q=ctx, kv=ctx)    # N×N
  tgt = tgt + cross_attn(q=tgt, kv=ctx)   # M×N
  ```
- **Block: standard pre-LN default.** Use the stable transformer form
  `x = x + attn(LN(x))`; `x = x + mlp(LN(x))`. No scalar-channel plumbing. If training
  stability argues for a different normalization placement, change the block rather than
  preserving paper/formula fidelity for its own sake.
- **Single MDN head, allowed to be slightly stronger.** Default to a shared
  `MLP(D, hidden) → Linear(3K)` → (raw weight, mean, log-std), with
  softmax/identity/softplus. *Not* K separate per-component MLPs + a global raw bias
  (that's more apparatus than the idea needs). A bare `Linear(D, 3K)` is acceptable only
  if the toy posterior still behaves.
- **Prior: keep the runtime-conditioning idea, not the exact representation.** The
  histogram→MLP encoder (`f_prob`, N_bins grid) is the first implementation because it is
  direct and visible. If moments, CDF features, learned basis coefficients, or another
  compact prior summary is simpler or trains better, use that. The elaborate prior
  *generative* process (80% MoG / geometric-K / Dirichlet / three mean-std configs) is
  slimmed to a simple sampler; it demonstrates injection just as well.
- **Latent loss weight: a single scalar `latent_weight`** — not the
  `((n_total − ½(max_ctx+min_ctx))/n_latent)^T` formula + `T` grid search (a
  benchmark-tuning artifact).
- **Cheap wins taken by default:** log-space latent prediction; multi-task loss balancing.

### Interface / token representation

- **Canonical internal object: two token sets.** The model receives a small `Batch` with
  `context: Tokens` and `target: Tokens`. Context tokens are known values or latent priors;
  target tokens are queries whose true values may be present for loss. Use padding masks
  for ragged batches, but no directional attention mask: conditioning direction comes from
  `ctx = self_attn(ctx)` and `tgt = cross_attn(tgt, ctx)`.
- **Shape contract.** The first code pass should implement this directly:
  ```python
  Batch(
      variables: list[Variable],
      context: Tokens,
      target: Tokens,
  )

  Tokens(
      var_id: LongTensor[B, T],
      x: FloatTensor[B, T, x_dim],
      value: FloatTensor[B, T],
      value_index: LongTensor[B, T],
      prior: FloatTensor[B, T, n_bins],
      mode: LongTensor[B, T],   # VALUE | PRIOR | QUERY
      mask: BoolTensor[B, T],
  )
  ```
  Target truth sits in `value` / `value_index` for loss, but the embedder ignores it
  when `mode == QUERY`.
- **`Tokens` is padded, explicit, and boring.** Fields:
  - `var_id: LongTensor[B, T]` indexes the `Variable` schema.
  - `x: FloatTensor[B, T, x_dim]` stores data covariates; latent tokens use zeros.
  - `value: FloatTensor[B, T]` stores transformed continuous values or dummy zeros.
  - `value_index: LongTensor[B, T]` stores discrete labels or dummy zeros.
  - `prior: FloatTensor[B, T, n_bins]` stores latent-prior histograms, zeros otherwise.
  - `mode: LongTensor[B, T]` is `VALUE`, `PRIOR`, or `QUERY`.
  - `mask: BoolTensor[B, T]` marks active padded tokens.
  Target truth lives in `value` / `value_index`; the embedder ignores it when
  `mode == QUERY`, while the loss uses it where `mask` is true.
- **`Variable` owns semantics.** Each variable has `name`, `kind` (`data` / `latent`),
  value type (`continuous` / `discrete`), optional discrete `cardinality`, transform
  (`identity`, `log`, maybe `logit`), and optional prior-grid range/bin count. Prior-grid
  ownership lives on latent variables, not in a global config.
- **Heads are hidden behind a prediction object.** `ACE.forward(batch)` returns an object
  with `.log_prob(batch.target)`, `.sample(batch.target, ...)`, and `.mean(batch.target)`.
  Continuous targets use the shared MDN path. Discrete targets use the shared
  `Kmax`-logit categorical path, masking invalid logits by `cardinality[var_id]`.
  The training loop should not branch on variable type.
- **Autoregression is a helper, not architecture.** Joint samples come from
  `sample_ar(model, batch, order=None)`, which repeatedly predicts one target,
  samples it, appends it to context as a `VALUE` token, and continues. The base model stays
  a diagonal prediction map. With `order=None`, the helper randomizes target order by
  default; pass an explicit order for deterministic probes.
- **Continuous head is shared.** Use one shared MDN head for all continuous variables;
  `var_id` / variable embeddings tell the model which scalar is being predicted. Do not
  add per-variable continuous heads unless a concrete task shows shared capacity is the
  bottleneck.
- **Discrete value embeddings use local labels plus offsets.** Each discrete variable's
  labels are local integers `0..k-1`. For context value embeddings, use one global
  embedding table with per-variable offsets so labels from different variables do not
  collide. For target predictions, use the shared `Kmax` head described above.
- **One `n_bins` initially.** Latent variables own their prior ranges, but the first
  implementation uses a global prior-bin count to avoid ragged prior tensors and nuisance
  padding. Different bin counts can wait until a real example needs them.
- **Inference targets may omit truth.** It is valid to pass target tokens with dummy
  `value` / `value_index` for prediction-only calls. `.log_prob(target)` is only called
  when target truth is present; example/train code are responsible for that distinction.
- **Loss weighting stays minimal.** Start with `data_weight` and `latent_weight` scalars,
  then average per active target token. No benchmark-derived formulas or per-task grids.
- **Device movement lives on the data objects.** Add `Tokens.to(device)` and
  `Batch.to(device)` immediately so training scripts do not accumulate tensor
  plumbing.

### Tighten before coding

- **Implement in dependency order.** Build `ace.py` and the Gaussian toy path first, prove
  the interface with `gaussian_toy.py`, then add GP-1D save/load shards. Do not build a generic
  sharded data layer before the model and toy diagnostic work.
- **Keep `Variable` boring.** The schema should be explicit and local: `name`, `kind`
  (`data` / `latent`), value type (`continuous` / `discrete`), cardinality for discrete
  variables, and optional transform metadata. Avoid clever token abstractions until the
  examples force them.
- **Restrict priors to latents initially.** ACE's useful runtime-prior story is about
  task-relevant latent variables. Do not generalize prior tokens to arbitrary data values
  until there is a concrete example that needs it.
- **Let the Gaussian toy be the first correctness oracle.** It should check that posterior
  moments and autoregressive two-latent predictions are sane before GP-1D adds Cholesky
  and dataset plumbing.
- **Use archived experiments selectively.** If a gitignored `temp/` directory is present,
  treat it as external experiment code. Reuse ideas only when they fit nanoACE directly:
  rectangular attention, CPU float64 GP sampling, and simple train-loop habits are useful;
  cache provenance, Slurm, prefetch, reeval schemas, and large diagnostic suites are out
  of scope.
- **Short verification runs stay loose.** The Gaussian toy oracle should catch broken
  heads and obviously wrong posterior moments, but it should not become a brittle
  strict-quality gate around stochastic training.
- **Checkpoint artifact deferred.** A tiny toy `state_dict` can ship only after the toy
  trains reliably and the blob is genuinely small. Design the code so retraining is the
  fallback; do not design the repository around a checkpoint.

### Data layer

- **Native path: generate → save → train** (the nanoGPT `prepare`→`train` pattern).
  Decoupled (`data.py` writes, `train.py` reads), inspectable (the dataset is an artifact
  you can open), reproducible (not an RNG side-effect of the loop).
- **Dataset format: minimal `.pt` shards plus a tiny manifest.** Shards store tensors in
  the canonical `Batch`/`Tokens` shape or enough arrays to reconstruct it. The manifest
  records schema version, variable definitions, normalization/transform metadata, and
  shard list. Avoid cache fingerprints, optimizer provenance, or resume-compatibility
  matrices.
- **Sharded pool on disk** so it scales past RAM. This is the regime that actually
  produced results at scale (a large finite pool, a few passes), not a toy shortcut.
- **Deterministic `(seed, pass)` reconstruction** of `n` and the context/target split;
  independent of batch size.
- **Shuffle is fixed to "both"** (permute shard order + rows within shards), seeded by
  `(seed, pass)`. No `shuffle_mode` enum — drops the field from any manifest and removes
  one axis from resume checks.
- **Online generation is not a mode.** Generators (`gaussian_toy`, `gp1d`, ...) are pure
  functions returning a batch. "Online" means calling the generator inside the training
  loop; it needs no config flag. The boundary is the generator function itself, so
  `train.py` does not need separate online and cached code paths.
- **Scaling happens at generation.** Examples should keep values roughly around
  `[-1,1]`, because the embedders/head bias assume that scale. This is not a hard bound:
  stochastic observations can have tails outside the range, and calibration issues should
  be checked before adding clipping or rescaling.

### Training / ops

- **Keep:** checkpoint + simple resume; cosine LR; Adam; grad clip; GPU model with a
  **CPU float64 data/oracle split** (Cholesky on the Win/WSL2 GPU path can trip the host
  watchdog).
- **Drop prefetch.** Generate→save moved the expensive Cholesky *off* the training loop
  (it now runs once, in the save step), so prefetch no longer earns its keep. `train.py`
  reads shards synchronously. **Leave a code comment** explaining this, so the absence
  reads as a decision, not an oversight. (If the GPU ever starves on H2D, add
  pinned-memory + a copy stream as a later, local change.)
- **Cut entirely (large experiment management):** Slurm/HPC layer; finite-pool caching machinery
  beyond the simple sharded save (manifest, DGP-hash fingerprint, shuffle enum);
  multi-axis resume-provenance guards; schema-migration `reeval`; mechanistic
  probes / partial-R² / causal interventions (those served a separate research question).

### Examples / scope

- **Example 1: Gaussian (`mu`, `sigma`) toy.** Two latents, fixed latent priors, analytic
  Bayesian posterior on a grid, and a matching posterior predictive diagnostic. Runs
  online and remains small; longer local runs can save a checkpoint/plot for inspection.
  - **An optional pretrained checkpoint is useful** because it would make `gaussian_toy.py`
    instant and provide a stable regression anchor. Distribution is deferred until the
    trained artifact is small and worth keeping. See Open questions.
- **Example 2: GP-1D regression.** Lengthscale/outputscale as continuous latents,
  **plus kernel selection as a discrete latent** (this is what exercises the discrete
  path so it isn't dead code). Use a **deliberately diverse kernel set** (RBF, Matern,
  periodic, ...) so the sampled functions look genuinely different and classification is
  meaningful. The first implementation is online in `gp1d.py`; use the generate/save/pool
  path only if longer training makes online Cholesky sampling the bottleneck.
- **"Add a third task" is the canonical user/agent extension** — nano ships exactly two.
  Generality lives in the *architecture* (free: attention over a token stream), not in
  the *example surface*.

### Layout

- `ace.py`         — config → embedder → block → forward → heads → loss   (THE file)
- `diagnostics.py` — reusable grid-query helpers for marginal/AR diagnostics
- `gaussian_toy.py` — executable Gaussian toy: generator, training, oracle, eval, checkpoint, plot
- `gp1d.py`        — executable GP-1D example: generator, training, eval, checkpoint, plot
- `data.py`        — generators + save/load (future sharded saved-pool path)
- `train.py`       — loop + resume (future sharded saved-pool path)
- Dependencies: **torch only** in the core; plotting imports are isolated to `gaussian_toy.py`.
  Match the pinned workstation-tested stack unless there is a specific reason to change it:
  `torch==2.11.0+cu128` via `https://download.pytorch.org/whl/cu128` on the RTX 4060
  Laptop GPU. Single dataclass config, no config framework.

### Open questions

- **Correctness oracle as an automated test?** The Gaussian toy's analytic grid posterior
  is kept as a first-class *diagnostic* (`gaussian_toy.py` computes/shows predicted-vs-true
  moments). Whether it also becomes an automated check is undecided — leaning at most a
  loose check that catches catastrophic breakage, not a strict R²-threshold quality
  gate (brittle against stochastic training).
- **Online gen as a one-flag option** — resolved: not a flag; it's the generator-as-
  function (see Data layer).
- **How to distribute optional checkpoints** (deferred — decide once there is a real
  trained artifact). Options on the table: a tiny toy checkpoint committed in-repo (keeps
  the Gaussian example instant + offline; sub-MB, no Git LFS, plain `state_dict`); a larger model
  hosted on HF and lazy-downloaded (natural home for the multi-MB GP model); or both
  (two-tier). Principles to preserve for any option: core stays **torch-only** (any
  HF fetch is an optional extra, never required to read/run/train); the model ships
  config + seed so it's regenerable, not a mystery binary; and it's an *example
  artifact*, not the packaged "runtime product" non-goal.
