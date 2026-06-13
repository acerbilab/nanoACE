# EFFICIENT AUTOREGRESSIVE INFERENCE FOR TRANSFORMER PROBABILISTIC MODELS - Appendix

---

## A METHOD DETAILS

This appendix spells out the modules used in Eq. (3), the single block- sparse attention mask that implements requirements (R1)- (R4), and the exact procedures for autoregressive sampling and one- pass joint log- likelihood evaluation.

## A.1 MODULES AND NOTATION

Our method uses three sets of tokens: context \(\mathcal{C}\) , buffer \(\mathcal{B}\) , and targets \(\mathcal{T}\) , of sizes \(N,K,M\) , respectively. Throughout this paper, let

\[\mathbf{E}_{x}:\mathcal{X}\to \mathbb{R}^{d},\quad \mathbf{E}_{y}:\mathcal{Y}\to \mathbb{R}^{d},\quad \mathbf{a}:\{1,\ldots ,K\}\to \mathbb{R}^{d}\]

denote learned embeddings for inputs, outputs, and buffer positions. In addition, we introduce role embeddings that indicate token type, denoted by \(e_{\mathrm{ctx}}^{\mathrm{role}}\) , \(e_{\mathrm{buf}}^{\mathrm{role}}\) , and \(e_{\mathrm{tgt}}^{\mathrm{role}}\) for context, buffer, and target tokens, respectively.

Context encoder \(\mathbf{r}_{\mathcal{C}}\) . Given context pairs \(\mathcal{C} = \{({\bf x}_{n},y_{n})\}_{n = 1}^{N}\) , construct context tokens: \(e_{n}^{\mathrm{ctx}} =\) \(\mathbf{E}_{x}(\mathbf{x}_{n}) + \mathbf{E}_{y}(y_{n}) + e_{\mathrm{ctx}}^{\mathrm{role}}\) , process them with bidirectional MHSA (no positional embeddings), and cache per- layer keys/values:

\[\{\mathbf{K}\mathbf{V}_{\mathcal{C}}^{\ell}\}_{\ell = 1}^{L} = \mathbf{r}_{\mathcal{C}}(\mathcal{C})\quad (\mathrm{computed~once};\mathrm{immutable}).\]

Buffer encoder \(\mathbf{r}_{\mathcal{B}}\) . For a buffer prefix \(\mathcal{B}_{1:k} = \{({\bf x}_{j}^{*},y_{j}^{*})\}_{j = 1}^{k}\) , form tokens \(e_{j}^{\mathrm{buf}} = \mathbf{E}_{x}(\mathbf{x}_{j}^{*}) + \mathbf{E}_{y}(y_{j}^{*}) + \mathbf{a}(j) + e_{\mathrm{buf}}^{\mathrm{role}}\) , then apply strictly causal MHSA on \(\{e_{j}^{\mathrm{buf}}\}_{j\leq k}\) so that each token is restricted to attend only to earlier tokens in the sequence, and in addition, each token performs cross- attention to the cached context \(\{\mathbf{K}\mathbf{V}_{\mathcal{C}}^{\ell}\}\) . This yields per- layer \(\mathbf{K}\mathbf{V}_{\mathcal{B}_{1:k}}^{\ell}\) that we update incrementally at inference:

\[\{\mathbf{K}\mathbf{V}_{\mathcal{B}_{1:k}}^{\ell}\}_{\ell = 1}^{L} = \mathbf{r}_{\mathcal{B}}\left(\mathcal{B}_{1:k},\mathbf{r}_{\mathcal{C}}(\mathcal{C})\right).\]

Target decoder \(\mathbf{r}_{\mathrm{tgt}}\) and prediction head. For a target input \(\mathbf{x}_{m}^{*}\) we create a query token \(e_{m}^{\mathrm{tgt}} =\) \(\mathbf{E}_{x}(\mathbf{x}_{m}^{*}) + e_{\mathrm{tgt}}^{\mathrm{role}}\) . The target decoder \(\mathbf{r}_{\mathrm{tgt}}\) performs a single cross- attention from \(e_{m}^{\mathrm{tgt}}\) to the concatenated keys/values of the context cache \(\{\mathbf{K}\mathbf{V}_{\mathcal{C}}^{\ell}\}\) and the visible buffer prefix \(\{\mathbf{K}\mathbf{V}_{\mathcal{B}_{1:v_{m}}}^{\ell}\}\) , followed by normalization and an MLP:

\[\mathbf{h}_{m} = \mathbf{r}_{\mathrm{tgt}}\Big(e_{m}^{\mathrm{tgt}},\Big[\{\mathbf{K}\mathbf{V}_{\mathcal{C}}^{\ell}\} ,\{\mathbf{K}\mathbf{V}_{\mathcal{B}_{1:v_{m}}}^{\ell}\} \Big]\Big),\qquad \phi_{m} = \psi (\mathbf{h}_{m}),\]

where \(\psi\) is the distribution head (e.g., the mixture- of- Gaussian head).

## A.2 TRAINING MASK THAT IMPLEMENTS (R1)- (R4)

We concatenate tokens as \([\mathcal{C},\mathcal{B},\mathcal{T}]\) with sizes \(N,K\) , and \(M\) , respectively, and use one block- sparse attention mask consisting of the following five unmasked sections (everything else is masked):

(1) Self-attention within context. Context tokens attend bidirectionally to other context tokens. Context never attends to buffer or targets (context is read-only outside this block).

(2) Buffer reads context (cross-attention). Each buffer token can read (attend to) all context tokens. This lets the buffer incorporate task information from the cached context while keeping the context cache immutable.

(3) Causal self-attention within the buffer. Within the buffer itself, attention is strictly causal: a buffer token at position \(j\) can only read earlier buffer positions \(< j\) (no future reads). This encodes the autoregressive dependency among realized targets.

(4) Targets read context (cross-attention). Each target query can read the entire cached context. There are no edges between targets.

(5) Targets read buffer (prefix only, cross-attention). Each target query can read only a visible prefix of the buffer. The visible prefix length for target \(m\) is \(v_{m}\) : training (teacher forcing): we set \(v_{m} = 0\) for \(50\%\) of targets and sample \(v_{m} \sim \mathrm{Uniform}\{1,\ldots ,K\}\) for the rest (the curriculum);

> **Image description.** A technical diagram consisting of two side-by-side block charts, illustrating different attention mask structures for a "Transformer diagonal prediction-map." Both charts use stacked horizontal blocks to represent computational components, with a vertical axis labeled "Q" (Query) and a horizontal axis labeled "K/V" (Key/Value).
>
> The image is divided into two panels:
>
> **Left Panel: Transformer diagonal prediction-map training mask**
> This panel shows a simpler structure, divided into two main horizontal blocks:
> 1.  **Top Block:** Light yellow/cream color, labeled "Context self-attn."
> 2.  **Bottom Block:** Dark orange/brown color, labeled "Target-context cross-attn."
>
> **Right Panel: Transformer diagonal prediction-map training mask w/ AR buffer**
> This panel shows a more complex structure, incorporating an autoregressive (AR) buffer, and is divided into four distinct horizontal blocks:
> 1.  **Top Block:** Light yellow/cream color, labeled "Context self-attn."
> 2.  **Second Block:** Medium orange color, labeled "Buffer-context cross-attn."
> 3.  **Third Block:** Darker orange/brown color, labeled "Buffer self-attn."
> 4.  **Bottom Block:** Darkest brown/black color, labeled "Target-buffer cross-attn."
>
> Both panels maintain the same overall dimensions and axis labels, visually demonstrating how the introduction of the AR buffer (the right panel) adds two intermediate computational layers (Buffer-context cross-attn and Buffer self-attn) between the context and the target processing.

<center>Figure A1: Block-sparse attention masks with and without an autoregressive buffer. Left: a diagonal prediction-map transformer (e.g., TNP/PFN): the context attends to itself and each target reads the entire context. Right: our buffered variant inserts an autoregressive memory \(B\) between context and targets, adding three blocks: (i) buffer reads context (ii) causal self-attention within buffer (iii) target reads varying number of elements from start of buffer, depending on curriculum. </center>

Algorithm 1 Autoregressive sample generation for \(K\) targets

Require: Context \(\mathcal{C} = \{(x_{n},y_{n})\}_{n = 1}^{N}\) , target inputs \(\{x_{k}^{*}\}_{k = 1}^{K}\)

1: \(\{\mathrm{KV}_{\mathcal{C}}^{\ell}\} \leftarrow \mathbf{r}_{\mathcal{C}}(\mathcal{C})\) \(\mathcal{O}(N^{2})\) ; cached

2: Initialize \(\{\mathrm{KV}_{\mathcal{B}_{1:0}}^{\ell}\}\) \(\triangleright\) empty buffer cache

3: for \(k = 1\) to \(K\) do

4: \(\mathbf{h}_{k}\leftarrow \mathbf{r}_{\mathrm{tgt}}\left(\mathbf{E}_{x}(x_{k}^{*}) + e_{\mathrm{tgt}}^{\mathrm{role}},\left[\{\mathrm{KV}_{\mathcal{C}}^{\ell}\} ,\{\mathrm{KV}_{\mathcal{B}_{1:k - 1}}^{\ell}\} \right]\right)\)

5: Sample \(y_{k}^{*}\sim p_{\theta}(\cdot ;\psi (\mathbf{h}_{k}))\)

6: Append \((x_{k}^{*},y_{k}^{*})\) ; update \(\{\mathrm{KV}_{\mathcal{B}_{1:k}}^{\ell}\}\) (strictly causal)

7: end for

8: return \(\{y_{k}^{*}\}_{k = 1}^{K}\)

sampling: at step \(k\) , the active query sees the realized prefix \(k - 1\) ; one- pass joint log- likelihood: packed queries use \(v_{m} = m - 1\) to recover the autoregressive chain in a single forward pass.

All other connections are masked: context never reads buffer or targets; targets never read targets; and buffer never reads targets. This single pattern implements the four requirements from the main text—immutable context, strictly causal buffer, unidirectional flow out of context, and target access to (context + visible buffer). See Fig. A1 for the diagram.

Complexity. Under this mask, a full prediction pass costs \(\mathcal{O}(N^{2} + NK + K^{2})\) attention operations per layer: one- time \(\mathcal{O}(N^{2})\) for \(\mathcal{C}\) , \(\mathcal{O}(NK)\) for reads from \(\mathcal{C}\) , and \(\mathcal{O}(K^{2})\) for causal buffer self- attention. This replaces the \(\mathcal{O}(K(N + K)^{2})\) cost of naive AR re- encoding. Packing \(B\) target orders in parallel (for order averaging) isolates the \(B\) buffer sets while sharing the context cache, yielding \(\mathcal{O}(N^{2} + B(NK + K^{2}))\) .

## A.3 ALGORITHMS FOR AUTOREGRESSIVE SAMPLING AND LOG-LIKELIHOOD EVALUATION

We include here the pseudocode for the main procedures used in our method. Algorithm 1 details the autoregressive sampling procedure, and Algorithm 2 presents the joint likelihood evaluation.

Algorithm 2 Joint log- likelihood evaluation for \(K\) targets

Require: Context \(\mathcal{C} = \{(x_{n},y_{n})\}_{n = 1}^{N}\) , ordered targets \(\{(x_{k}^{*},y_{k}^{*})\}_{k = 1}^{K}\)

1: \(\{\mathrm{KV}_{\mathcal{C}}^{\ell}\} \leftarrow \mathbf{r}_{\mathcal{C}}(\mathcal{C})\) \(\triangleright \mathcal{O}(N^{2})\) ; cached

2: Build all \(K\) buffer tokens; compute \(\{\mathrm{KV}_{\mathcal{B}_{1:K}}^{\ell}\}\) under causal mask

3: Build target queries \(\{\mathbf{E}_{x}(x_{k}^{*}) + e_{\mathrm{tgt}}^{\mathrm{role}}\}_{k = 1}^{K}\)

4: Mask: target \(k\) sees \(\mathcal{B}_{1:k - 1}\) and all of \(\mathcal{C}\)

5: Compute \(\{\log p_{k}\}_{k = 1}^{K}\)

6: return \(\sum_{k = 1}^{K}\log p_{k}\)

## B TRANSFORMER NEURAL PROCESS BASELINES DETAILS

We summarize the baseline transformer neural process (TNP) variants used in our comparisons, following Nguyen & Grover (2022). Architectural hyperparameters appear in Appendix D.1.

## B.1 TNP-D

This model takes as input a context set \(\{({\bf x}_{n},y_{n})\}_{n = 1}^{N}\) and a target set \(\{{\bf x}_{m}^{*}\}_{m = 1}^{M}\) . Similar to Appendix A, the context embeddings \(e_{m}^{\mathrm{ctx}}\) are processed with bidirectional MHSA with no positional encodings. Each target is decoded by:

\[\mathbf{h}_{m} = \mathbf{r}_{\mathrm{tgt}}\big(e_{m}^{\mathrm{tgt}}, \mathbf{r}_{\mathcal{C}}(\mathcal{C})\big), \qquad \phi_{m} = \psi (\mathbf{h}_{m}),\]

where \(\psi\) is the distribution head (Gaussian as in the original paper; we primarily use a mixture of Gaussians). The left panel of Fig. A1 shows the training mask for TNP- D. This model is trained via maximum likelihood estimation of independent targets given a fixed context set.

At deployment, the decoding can be independent or autoregressive, yielding TNP- D- Ind and TNP- D- AR methods. TNP- D- Ind decodes all targets independently in a single pass. It is fast (context and targets encoded once), but cannot capture dependencies between targets.

TNP- D- AR decodes targets sequentially, appending each sampled \(({\bf x}_{m}^{*},y_{m}^{*})\) to the context. This captures joint structure but requires re- encoding the growing set at each step. TNP- D- Ind is invariant to target order; TNP- D- AR is order- sensitive, so we approximate the predictive distribution by averaging over multiple target orderings.

## B.2 TNP-ND

This model encodes the context set once and decodes all targets simultaneously by parameterizing a joint multivariate Gaussian distribution over the outputs. The embedder and transformer backbone are identical to those of TNP- D- Ind:

\[\mathbf{h}_{m} = \mathbf{r}_{\mathrm{tgt}}\big(e_{m}^{\mathrm{tgt}}, \mathbf{r}_{\mathcal{C}}(\mathcal{C})\big).\]

Then the joint distribution is obtained via

\[\phi = \psi_{N D}(\mathbf{h}_{1},\ldots ,\mathbf{h}_{M}),\]

where \(\psi_{N D}\) is the multivariate Gaussian head that outputs both a mean vector and valid covariance matrix. The mean is produced per target, and a lightweight self- attention head over the set of targets yields fixed- width embeddings that are transformed into a valid covariance factor. This design supports a variable number of targets and is invariant to target order.

The training optimizes the joint multivariate Gaussian likelihood of the target points. At inference, the joint samples and log- likelihood are computed in a single pass. This model is invariant to the order of target points.

## B.3 TNP-A

The key difference between this model and TNP- D is the attention mechanism on the target set. This model processes three sets: the context \(\{({\bf x}_{n},y_{n})\}_{n = 1}^{N}\) , the target \(\{{\bf x}_{m}^{*}\}_{m = 1}^{M}\) , and the observed

target \(\{({\bf x}_{m}^{*},y_{m}^{*})\}_{m = 1}^{M}\) . To differentiate, we denote the embeddings of \(\{({\bf x}_{m}^{*},y_{m}^{*})\}_{m = 1}^{M}\) by \(\{e_{m}^{y,\mathrm{tgt}}\}\) . Similar to TNP- D, the context embeddings attend to each other. For the target set, each \(e_{m}^{\mathrm{tgt}}\) attends to the context and the previous observed target embeddings \(e_{< m}^{y,\mathrm{tgt}}\) . Likewise, the observed target embeddings attend to context and previous target embeddings (Fig. 2 of Nguyen & Grover 2022).

The target causal mask allows TNP- A to model the joint likelihood simultaneously in one single pass, assuming the observations are given (e.g., for training and test log- likelihood evaluations). For prediction generation, however, each sampled target needs to be re- encoded and attended for the generation of next targets, yielding a sequential re- encoding procedure. The causal mask on the target set is sensitive to the target order, and thus the final likelihood is an average over multiple random permutations. Note that this model processes duplicated target set- \(\{-{\bf x}_{m}^{*}\}_{m = 1}^{M}\) and an observed sequence \(\{({\bf x}_{m}^{*},y_{m}^{*})\}_{m = 1}^{M}\) ; this creates significant computational overhead in both the training and the inference, particularly when the target set is large (see e.g. Appendix C and Figs. A7 to A9).

Compared to our method, TNP- A can be viewed as TNP- D with a 'frozen buffer' \(\{({\bf x}_{m}^{*},y_{m}^{*})\}_{m = 1}^{M}\) of size \(K = M\) containing the observed targets. For likelihood evaluation where all sets are processed in one shot, the behavior of TNP- A and our approach are analogous, with the set \(\{({\bf x}_{m}^{*},y_{m}^{*})\}_{m = 1}^{M}\) serving a role similar to our buffer. However, for AR sampling, TNP- A repeatedly re- encodes the full context and target sets after each sampled \(y_{m}^{*}\) , whereas our method dynamically adapts to new samples. Moreover, since TNP- A does not afford a dynamic- size target set decoupled from the 'in- context' targets, training is much more expensive than our method (see Fig. 1 in the main text).

## C COMPUTATIONAL EFFICIENCY DETAILS

This section provides additional empirical results to support the efficiency claims made in the main paper. We present an analysis of performance scaling with batch size, an ablation study of our custom kernel, a comparison against unoptimized open- source baselines, and further ablations on training time. In all subsequent plots, the absence of a data point for a given method indicates that the experiment failed due to an out- of- memory (OOM) error for that specific configuration.

## C.1 SCALING WITH BATCH SIZE

To analyze the effect of batch size \(B\) , we provide expanded results for autoregressive sampling and joint log- likelihood evaluation in Fig. A2 and Fig. A3, respectively. These plots show the wall- clock time as a function of the number of context points \(N\) for various batch sizes. The results confirm that our method's performance advantage over autoregressive baselines like TNP- A is consistent and often widens as the context and batch size increase.

> **Image description.** A set of four line graphs arranged horizontally, titled "Sample generation time (M=16)," which compares the computational efficiency of five different methods across varying batch sizes ($B$) and sequence lengths ($N$).
>
> The graphs share a common structure:
> *   **Y-axis:** Labeled "Time (s)," this axis uses a logarithmic scale, ranging from $10^{-3}$ to $10^0$.
> *   **X-axis:** Labeled "$N$," this axis represents the sequence length and features discrete values: 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Located at the bottom left, the legend identifies five methods using distinct colors and markers:
>     *   "Ours" (Blue line with 'x' markers)
>     *   "TNP-D-Ind" (Orange line with square markers)
>     *   "TNP-D-AR" (Green line with triangle markers)
>     *   "TNP-A" (Purple line with diamond markers)
>     *   "TNP-ND" (Red line with circle markers)
>
> The four panels are distinguished by the batch size ($B$) they represent:
> 1.  **Panel 1:** $B = 128$
> 2.  **Panel 2:** $B = 256$
> 3.  **Panel 3:** $B = 512$
> 4.  **Panel 4:** $B = 1024$
>
> **Data Trends and Performance:**
> In all four panels, the generation time for every method increases as the sequence length $N$ increases, indicating a positive correlation between sequence length and computational time.
>
> Comparing the methods across all panels:
> *   The "Ours" method (blue line) consistently demonstrates the lowest sample generation time, performing better than all other methods regardless of the batch size $B$.
> *   The "TNP-D-Ind" method (orange line) generally shows the second-lowest generation time.
> *   The remaining methods ("TNP-D-AR," "TNP-A," and "TNP-ND") exhibit higher generation times. The "TNP-A" method (purple line) typically shows the highest generation time among the tested methods.
> *   As the batch size $B$ increases (moving from the left panel $B=128$ to the right panel $B=1024$), the overall generation time for all methods increases, and the performance gap between the fastest method ("Ours") and the slowest methods tends to widen.

<center>Figure A2: Autoregressive sampling time (log scale) versus context size \(N\) for an expanded range of batch sizes \(B\) . </center>

> **Image description.** A composite line graph consisting of four panels, titled "Log-likelihood evaluation time (log scale) versus context size $N$," which illustrates the computational time required for joint log-likelihood evaluation across different batch sizes ($B$) and context sizes ($N$).
>
> The overall figure uses a logarithmic scale for the Y-axis, labeled "Time (s)," ranging from $10^{-3}$ to $10^0$. The X-axis, labeled "$N$," represents the context size and is marked with values 32, 64, 128, 256, 512, and 1024.
>
> A legend at the bottom identifies five methods, each represented by a distinct color and marker:
> *   Ours (Blue)
> *   TNP-D-Ind (Orange)
> *   TNP-D-AR (Green)
> *   TNP-A (Purple)
> *   TNP-ND (Red)
>
> The four panels are organized by increasing batch size ($B$):
>
> 1.  **Panel $B=128$:** Shows the evaluation time for a batch size of 128. All methods exhibit an increasing trend in time as the context size $N$ increases. The "Ours" method (blue line) consistently achieves the lowest evaluation time across all values of $N$. The lines are relatively close together, indicating a smaller performance gap between methods at this batch size.
> 2.  **Panel $B=256$:** Displays the results for a batch size of 256. The overall time values are higher than in the $B=128$ panel. The relative performance order remains consistent, with "Ours" being the fastest and "TNP-D-Ind" being the slowest.
> 3.  **Panel $B=512$:** Shows the evaluation time for a batch size of 512. The time values are significantly higher, reflecting the increased computational load. The performance advantage of the "Ours" method becomes more pronounced compared to the other baseline methods.
> 4.  **Panel $B=1024$:** Represents the largest batch size shown. This panel exhibits the highest overall time values. The separation between the performance curves is the most distinct here, with the "Ours" method maintaining the lowest time across the entire range of $N$, demonstrating a substantial performance gain as the batch size increases.
>
> In summary, the graph visually demonstrates that while all methods experience an increase in evaluation time as the context size $N$ grows, the "Ours" method consistently maintains the lowest time complexity across all tested batch sizes ($B=128$ to $B=1024$).

<center>Figure A3: Joint log-likelihood evaluation time (log scale) versus context size \(N\) for an expanded range of batch sizes \(B\) . </center>

## C.2 IMPACT OF CUSTOM TRITON KERNEL

To isolate the contribution of our custom attention kernel, we compare the sampling time of our method with and without this optimization. The kernel is designed to accelerate a key computational step: the cross- attention between the batched target embeddings (batch size \(B\) ) and the concatenation of a batched buffer cache with a shared context cache (batch size 1). A naive implementation would explicitly expand the context cache tensor \(B\) times to match the batch dimension before the attention operation. This "expand" operation is memory- bandwidth intensive and creates a large, redundant intermediate tensor.

Our custom Triton kernel avoids this bottleneck by fusing the expansion and attention computations. The kernel loads the single context cache into fast SRAM and reuses it for each item in the batch, calculating the attention on- the- fly without ever materializing the full expanded tensor in slower global memory. As shown in Fig. A4, this memory- centric optimization provides a substantial speedup that grows with the batch size \(B\) .

> **Image description.** A multi-panel line graph titled "Sample generation time ($M=16$)" displays the relationship between sampling time and context size ($N$) for four different batch sizes ($B$). The graph uses a logarithmic scale for the vertical axis, representing time in seconds (s).
>
> **General Structure and Axes:**
> The image consists of four distinct panels arranged horizontally, each labeled with a specific batch size ($B$): $B=128$, $B=256$, $B=512$, and $B=1024$.
> *   **Y-axis (Vertical):** Labeled "Time (s)", it is scaled logarithmically, showing major ticks at $10^{-2}$ and $10^{-1}$.
> *   **X-axis (Horizontal):** Labeled "$N$", it represents the context size and features discrete values: 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Located at the bottom center, it identifies two data series: a solid blue line labeled "Ours (w Triton)" and a light blue/cyan line labeled "Ours (w/o Triton)".
>
> **Data Trends by Panel:**
> The visual data demonstrates how the performance difference between the two methods changes as the batch size ($B$) increases.
>
> 1.  **Panel $B=128$:** Both the "w Triton" and "w/o Triton" lines are relatively flat across all values of $N$. The blue line ("w Triton") is slightly lower than the light blue line ("w/o Triton"), indicating a minor speedup.
> 2.  **Panel $B=256$:** Similar to the first panel, both lines remain relatively flat. The "w Triton" line continues to be slightly faster than the "w/o Triton" line.
> 3.  **Panel $B=512$:** A noticeable divergence begins. While the "w Triton" line remains relatively flat and low, the "w/o Triton" line starts to trend upward, particularly as $N$ increases beyond 256.
> 4.  **Panel $B=1024$:** This panel shows the most significant difference. The "w/o Triton" line exhibits a steep, increasing trend, rising sharply as $N$ increases, reaching the highest time values on the graph. In contrast, the "w Triton" line remains relatively flat and significantly lower, demonstrating the most substantial performance gain compared to the other panels.
>
> Overall, the visual pattern shows that the performance advantage of the "Ours (w Triton)" method grows dramatically as both the batch size ($B$) and the context size ($N$) increase.

<center>Figure A4: Ablation study for autoregressive sampling, comparing our method with and without the custom Triton kernel across different context and batch sizes. </center>

## C.3 COMPARISON TO OPEN-SOURCE BASELINES

To demonstrate the fairness of our primary comparisons, we benchmark our optimized baseline implementations against their standard, publicly available versions. The results for sampling and likelihood evaluation are shown in Fig. A5 and Fig. A6. Our optimized baselines are consistently \(3 - 10 \times\) faster than their standard counterparts. This confirms that our method's performance gains are due to fundamental algorithmic advantages, not an unfair comparison against unoptimized code.

> **Image description.** A multi-panel line graph titled "Sample generation time ($M=16$)" that compares the time taken for various autoregressive sampling methods across different batch sizes ($B$) and input sizes ($N$).
>
> The figure consists of four distinct panels arranged horizontally, each representing a different batch size ($B$): $B=128$, $B=256$, $B=512$, and $B=1024$.
>
> **Axes and Scale:**
> *   The vertical Y-axis, labeled "Time (s)", uses a logarithmic scale, ranging from $10^{-3}$ to $10^1$.
> *   The horizontal X-axis, labeled "$N$", represents the input size and is consistent across all panels, showing values of 32, 64, 128, 256, 512, and 1024.
>
> **Data Series and Legend:**
> A legend positioned below the first panel identifies eight distinct data series, each represented by a unique color and line style (solid for "base," dashed for "compiled"):
> *   TNP-D-Ind (base) and TNP-D-Ind (compiled) (Orange)
> *   TNP-D-AR (base) and TNP-D-AR (compiled) (Teal)
> *   TNP-A (base) and TNP-A (compiled) (Purple)
> *   TNP-D (base) and TNP-D (compiled) (Light Blue)
>
> **Visual Trends:**
> In all four panels, the general trend shows that the sample generation time increases as the input size $N$ increases (moving from left to right along the X-axis). Furthermore, for every method, the "compiled" versions (represented by dashed lines) consistently show lower time values compared to their corresponding "base" versions (represented by solid lines), indicating performance improvements from compilation. The relative performance ranking of the different methods shifts slightly as the batch size $B$ increases.

<center>Figure A5: Comparison of our optimized baseline implementations against standard open-source versions for autoregressive sampling. </center>

> **Image description.** A multi-panel line graph displaying the log-likelihood evaluation time ($M = 16$) across four different batch sizes. The graph is composed of four horizontally arranged panels, each representing a specific batch size ($B$).
>
> **General Structure and Axes:**
> *   **Y-Axis:** Labeled "Time (s)", this axis uses a logarithmic scale, ranging from $10^{-3}$ to $10^1$.
> *   **X-Axis:** Labeled "$N$", this axis represents the context size and is consistent across all panels, showing values from 32 to 1024.
> *   **Panel Labels:** The four panels are labeled at the top with the batch size ($B$): $B = 128$, $B = 256$, $B = 512$, and $B = 1024$.
>
> **Data Series (Legend):**
> The graph tracks four distinct data series, differentiated by color and method:
> 1.  **TNP-A (base):** Represented by a light green/teal line.
> 2.  **TNP-A (compiled):** Represented by a darker green/teal line.
> 3.  **TNP-ND (base):** Represented by a light orange/yellow line.
> 4.  **TNP-ND (compiled):** Represented by a darker orange/red line.
>
> **Visual Trends and Patterns:**
> *   **Scaling with N:** In all four panels, the time required increases as the number of target points ($N$) increases, showing an upward trend for all methods.
> *   **Method Comparison:** The TNP-A methods (green/teal lines) consistently require significantly more time than the TNP-ND methods (orange/red lines) across all batch sizes and values of $N$.
> *   **Compilation Effect:** Within both the TNP-A and TNP-ND groups, the "compiled" versions (darker lines) consistently show lower time values than their respective "base" versions (lighter lines), indicating performance improvement through compilation.
> *   **Batch Size Impact:** As the batch size ($B$) increases from 128 to 1024 (moving from left to right), the overall time values for all methods generally increase, and the performance gap between the TNP-A and TNP-ND methods appears to widen.

<center>Figure A6: Comparison of our optimized baseline implementations against standard open-source versions for joint log-likelihood evaluation. </center>

## C.4 TRAINING TIME SCALING

We further analyze the scaling of training step time with respect to the number of target points \(M\) for different batch sizes. Figs. A7 to A9 show this relationship for batch sizes of 64, 128, and 256, respectively. The results show that as the context, target, or batch size increases, TNP- A becomes increasingly expensive to train relative to all other methods.

> **Image description.** A multi-panel line graph titled "Training step time (forward + backward, B=64)" displays the relationship between training time and the number of target points ($M$) for different values of $N$. The image consists of three identical line graphs arranged horizontally, each representing a different value of $M$.
>
> The common axes for all three panels are:
> *   **Y-axis:** Labeled "Time (s)", this axis uses a logarithmic scale, ranging from $10^{-1}$ to $10^1$.
> *   **X-axis:** Labeled "$N$", this axis displays discrete values: 32, 64, 128, 256, 512, and 1024.
>
> The three panels are labeled by the value of $M$:
> 1.  The left panel is titled "$M=128$".
> 2.  The middle panel is titled "$M=256$".
> 3.  The right panel is titled "$M=512$".
>
> A legend located below the graphs identifies four methods, each represented by a distinct line and marker:
> *   "Ours" (Blue line with 'x' markers)
> *   "TNP-D-Ind" (Orange line with square markers)
> *   "TNP-A" (Purple line with diamond markers)
> *   "TNP-ND" (Brown/Dark Orange line with circle markers)
>
> Visually, the graphs demonstrate that for all methods, the training time increases as the value of $N$ increases. Across all three panels ($M=128, 256, 512$), the "TNP-A" method (purple diamonds) consistently exhibits the highest training time. Conversely, the "Ours" method (blue 'x') consistently shows the lowest training time, indicating the most efficient performance. The "TNP-D-Ind" and "TNP-ND" methods occupy intermediate positions in terms of training time. The relative performance ranking of the four methods remains consistent across all three panels, regardless of the value of $M$.

<center>Figure A7: Training step time vs. number of target points \(M\) for batch size \(B = 64\) . </center>

> **Image description.** A multi-panel line graph titled "Training step time vs. number of target points M for batch size B = 128," which compares the training efficiency of four different models across varying numbers of steps (N) and target points (M).
>
> The image consists of three distinct panels arranged horizontally, each representing a different value for the number of target points (M).
>
> **Common Elements:**
> *   **Y-axis:** Labeled "Time (s)," this axis uses a logarithmic scale, ranging from $10^{-1}$ (0.1 seconds) to $10^1$ (10 seconds).
> *   **X-axis:** Labeled "$N$," this axis represents the number of steps and features discrete values: 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Located in the bottom left of the first panel, the legend identifies four data series:
>     *   "Ours" (represented by a blue line)
>     *   "TNP-D-Ind" (represented by an orange line)
>     *   "TNP-A" (represented by a pink/magenta line)
>     *   "TNP-ND" (represented by a brown/dark orange line)
>
> **Panel Details:**
> 1.  **Left Panel:** Labeled "$M = 128$."
> 2.  **Middle Panel:** Labeled "$M = 256$."
> 3.  **Right Panel:** Labeled "$M = 512$."
>
> **Visual Trends and Data Patterns:**
> In all three panels, the training time generally increases as the number of steps ($N$) increases, showing a positive correlation between $N$ and time.
>
> *   **TNP-A Performance:** The pink line representing "TNP-A" is consistently the highest across all three panels and all values of $N$, indicating that this model requires the longest training time.
> *   **Fastest Models:** The blue line ("Ours") and the orange line ("TNP-D-Ind") consistently occupy the lowest positions on the graph, demonstrating the fastest training times.
> *   **Relative Performance:** The lines for "Ours," "TNP-D-Ind," and "TNP-ND" are clustered closely together, particularly at lower values of $N$, suggesting similar performance among these three methods.
> *   **Impact of M:** As the number of target points ($M$) increases from 128 to 512, the overall training time for all models increases, and the gap between the fastest models (Ours/TNP-D-Ind) and the slowest model (TNP-A) appears to widen slightly.

<center>Figure A8: Training step time vs. number of target points \(M\) for batch size \(B = 128\) . </center>

> **Image description.** A composite figure consisting of three comparative line graphs, arranged horizontally, illustrating the "Training step time" across different model configurations as a function of input size $N$. All three panels share a common logarithmic Y-axis representing Time (s) and a common X-axis representing the input size $N$.
>
> The three panels are differentiated by the number of target points $M$ and the batch size $B$:
> 1.  **Left Panel:** Labeled "$M=128$", representing a batch size $B=128$.
> 2.  **Middle Panel:** Labeled "$M=256$", representing a batch size $B=256$.
> 3.  **Right Panel:** Labeled "$M=512$", representing a batch size $B=512$.
>
> **Shared Axes and Data Representation:**
> *   **Y-axis (Time (s)):** The vertical axis is logarithmic, ranging from $10^{-1}$ to $10^1$.
> *   **X-axis ($N$):** The horizontal axis represents the input size $N$, with discrete values marked at 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Each panel uses four distinct lines to represent four models:
>     *   "Ours" (Blue line)
>     *   "TNP-D-Ind" (Orange line)
>     *   "TNP-A" (Pink/Magenta line)
>     *   "TNP-ND" (Brown/Dark Orange line)
>
> **Visual Trends and Comparisons:**
> In all three panels, the training time for all four models increases as the input size $N$ increases.
>
> *   **TNP-A Performance:** Across all three configurations ($M=128, 256, 512$), the "TNP-A" model consistently exhibits the highest training step time, showing a significantly steeper and higher trajectory compared to the other models.
> *   **Other Models:** The "Ours," "TNP-D-Ind," and "TNP-ND" models generally cluster together, maintaining much lower training times. In the $M=128$ and $M=256$ panels, "Ours" and "TNP-D-Ind" appear to have the lowest training times.
> *   **Scaling:** As the number of target points $M$ increases from 128 to 512, the overall training time for all models increases, with the gap between the high-cost "TNP-A" and the lower-cost models widening proportionally.
>
> The figure visually demonstrates that the training time of the "TNP-A" model is substantially higher than the other methods, regardless of the batch size or the number of target points $M$.

<center>Figure A9: Training step time vs. number of target points \(M\) for batch size \(B = 256\) . </center>

## C.5 IMPACT OF ATTENTION PATTERNS ON TRAINING SPEED

A key difference between the baseline models is their compatibility with modern, efficient attention implementations. The causal attention mask required by TNP- A during training is incompatible with kernels like FlashAttention, forcing it to use PyTorch's standard, but slower, "math" attention backend. In contrast, models like TNP- D and ours can leverage these faster kernels.

In Appendix B, we discussed the duplicated processing of TNP- A on the target set, which incurs significant computational overhead. To determine if TNP- A's slow training is fundamental to its architecture or merely an artifact of this kernel incompatibility, we conduct a controlled ablation. We disable FlashAttention for all methods, forcing a fair comparison on the same standard PyTorch attention backend. The results, shown in Figs. A10 to A12, are unequivocal. Even on a level playing field, TNP- A's training time is orders of magnitude slower than all other methods. This confirms that its high computational cost is an inherent consequence of its autoregressive design, not just an implementation detail.

> **Image description.** A multi-panel line graph titled "Training step time (forward + backward), $B=64$," which displays the relationship between training time and the context size $N$ for three different values of $M$. The graph is composed of three side-by-side panels, each representing a different value of $M$.
>
> The common elements across all three panels are:
> *   **Y-axis:** Labeled "Time (s)," this axis uses a logarithmic scale, ranging from $10^{-1}$ to $10^1$.
> *   **X-axis:** Labeled "$N$," this axis represents the context size and features discrete values: $32, 64, 128, 256, 512,$ and $1024$.
> *   **Legend:** Located in the bottom left of the first panel, the legend identifies four methods using distinct colors and line styles:
>     *   "Ours" (Blue line)
>     *   "TNP-D-Ind" (Orange line)
>     *   "TNP-A" (Pink/Magenta line)
>     *   "TNP-ND" (Dark Orange/Brown line)
>
> The three panels are labeled according to the value of $M$:
> 1.  **Left Panel:** Labeled "$M=128$".
> 2.  **Middle Panel:** Labeled "$M=256$".
> 3.  **Right Panel:** Labeled "$M=512$".
>
> **Data Trends and Performance:**
> In all three panels, the training time for all four methods increases as the context size $N$ increases, demonstrating a positive correlation between context length and computational time.
>
> *   **Relative Performance:** The "Ours" method (blue line) consistently exhibits the lowest training time across all values of $N$ and $M$, indicating superior efficiency.
> *   **Comparison of Baselines:** The other three methods (TNP-D-Ind, TNP-A, and TNP-ND) generally show higher training times than "Ours." TNP-A (pink line) typically maintains the highest training time, especially as $N$ approaches $1024$.
> *   **Impact of $M$:** While the overall trend is consistent, the absolute time values shift across the panels, reflecting the change in the number of target points $M$. The performance gap between "Ours" and the other methods appears to widen as $N$ increases, regardless of the value of $M$.

<center>Figure A10: Training step time vs. number of target points \(M\) using the standard PyTorch attention backend (FlashAttention disabled). Batch size \(B = 64\) . </center>

> **Image description.** A multi-panel line graph consisting of three separate plots arranged horizontally, illustrating the relationship between training step time and context size ($N$) for different methods. The overall title for all three panels is "Training step time (forward + backward), B = 128."
>
> The graphs share a common structure:
> *   **Y-axis:** Labeled "Time (s)," this axis uses a logarithmic scale, ranging from $10^{-1}$ to $10^1$.
> *   **X-axis:** Labeled "$N$," this axis represents the context size and features discrete values: 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** A legend at the bottom of each panel identifies four methods: "Ours" (blue line), "TNP-D-Ind" (orange line), "TNP-A" (pink/magenta line), and "TNP-ND" (brown/dark orange line).
>
> The three panels are differentiated by the number of target points ($M$):
> 1.  **Left Panel:** Labeled "$M = 128$."
> 2.  **Middle Panel:** Labeled "$M = 256$."
> 3.  **Right Panel:** Labeled "$M = 512$."
>
> In all three panels, the visual trend shows that the training step time increases as the context size $N$ increases, with all lines sloping upwards. The "Ours" method (blue line) consistently maintains the lowest position on the graph across all $M$ values and $N$ sizes, indicating the fastest training time. Conversely, the "TNP-A" method (pink/magenta line) consistently shows the highest training time. The performance of "TNP-D-Ind" and "TNP-ND" falls between "Ours" and "TNP-A." The overall visual pattern demonstrates that the training time scales with $N$ for all methods, but the "Ours" method exhibits superior efficiency compared to the baselines.

<center>Figure A11: Training step time vs. number of target points \(M\) using the standard PyTorch attention backend (FlashAttention disabled). Batch size \(B = 128\) . </center>

> **Image description.** A multi-panel line graph consisting of three separate plots arranged horizontally, illustrating the relationship between training step time and the number of target points (N) for different batch sizes (M). The graphs are technical in nature, comparing the performance of four different methods.
>
> **General Structure and Axes:**
> All three panels share identical axes and scales.
> *   **Y-axis:** Labeled "Time (s)", representing the training step time, and uses a logarithmic scale ranging from $10^{-1}$ to $10^1$.
> *   **X-axis:** Labeled "N", representing the number of target points, and also uses a logarithmic scale with marked values at 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Located beneath the middle panel, it identifies four methods using distinct colors and markers:
>     *   "Ours" (Blue line with circular markers)
>     *   "TNP-D-Ind" (Orange line with square markers)
>     *   "TNP-A" (Pink/Magenta line with triangular markers)
>     *   "TNP-ND" (Darker orange/brown line with diamond markers)
>
> **Panel Details:**
> The three panels are differentiated by the batch size (M) used in the experiment:
> 1.  **Left Panel:** Labeled "$M = 128$".
> 2.  **Middle Panel:** Labeled "$M = 256$". This panel also features the centered title "Training step time (forward + backward, B = 256)".
> 3.  **Right Panel:** Labeled "$M = 512$".
>
> **Data Trends and Visual Patterns:**
> In all three panels, the general trend for all four methods is that the training step time increases as the number of target points (N) increases.
> *   The "Ours" method (blue line) consistently shows the lowest training time across all values of N and all batch sizes (M), indicating the fastest performance.
> *   The other three methods ("TNP-D-Ind," "TNP-A," and "TNP-ND") show progressively higher training times, with "TNP-A" (pink line) generally exhibiting the highest time, especially at larger values of N.
> *   The relative performance gap between the methods appears to widen as the batch size (M) increases from 128 to 512.

<center>Figure A12: Training step time vs. number of target points \(M\) using the standard PyTorch attention backend (FlashAttention disabled). Batch size \(B = 256\) . </center>

## C.6 MEMORY USAGE

Figure A13 reports peak GPU memory consumption during autoregressive sampling as a function of context size \(N\) across different batch sizes \(B\) . Our method maintains consistently low memory usage across all configurations, requiring \(6 - 7 \times\) less VRAM than TNP- D- AR and TNP- A at large context sizes \((N = 1024)\) . This efficiency stems from our fixed- size buffer mechanism: while autoregressive baselines must cache representations that grow with context size and batch size, our method only caches buffer representations of size \(K\) , independent of the batch. TNP- D- Ind and

TNP- ND show lower memory usage but, as demonstrated in the main text, cannot capture complex predictive dependencies.

> **Image description.** A multi-panel line graph, Figure A13, illustrating the relationship between Peak GPU memory usage (VRAM) and the number of context points ($N$) for five different models across four varying batch sizes ($B$).
>
> The image is composed of four distinct panels arranged horizontally, each representing a different batch size: $B=128$, $B=256$, $B=512$, and $B=1024$.
>
> **Common Graph Elements:**
> *   **Y-axis:** Labeled "Peak VRAM (GB)," this axis uses a logarithmic scale, ranging from $10^{-2}$ (0.01 GB) to $10^1$ (10 GB).
> *   **X-axis:** Labeled "$N$," this axis represents the number of context points and is marked with values: 32, 64, 128, 256, 512, and 1024.
> *   **Legend:** Located at the bottom, the legend identifies five models using distinct colors and line styles:
>     *   Ours (Blue line)
>     *   TNP-D-Ind (Orange line)
>     *   TNP-D-AR (Green line)
>     *   TNP-A (Purple line)
>     *   TNP-ND (Brown/Dark Orange line)
>
> **Data Trends and Comparisons:**
> *   **General Trend:** In all four panels, the memory usage for every model increases as the number of context points ($N$) increases.
> *   **Model Efficiency:** The "Ours" model (blue line) consistently demonstrates the lowest Peak VRAM usage across all tested values of $N$ and $B$.
> *   **Baseline Comparison:** Among the baseline models, the TNP-ND (brown/dark orange line) generally maintains the lowest memory usage, though it is significantly higher than the "Ours" model. The other baselines (TNP-D-Ind, TNP-AR, and TNP-A) show higher memory consumption than TNP-ND, particularly as $N$ grows.
> *   **Effect of Batch Size:** As the batch size $B$ increases from 128 (left panel) to 1024 (right panel), the overall memory usage for all models increases, indicating that larger batches require more GPU memory. However, the relative performance ranking of the models remains consistent across all batch sizes.
>
> The visual data strongly suggests that the "Ours" method scales efficiently, maintaining a substantially lower memory footprint compared to the expressive autoregressive baselines (TNP-AR, TNP-A, TNP-D-Ind) and even the TNP-ND model.

<center>Figure A13: Peak GPU memory usage during sampling as a function of context points \(N\) for different batch sizes \(B\) . Our method scales efficiently due to its fixed buffer size, using substantially less memory than expressive autoregressive baselines. </center>

## D EXPERIMENTAL DETAILS

## D.1 MODEL CONFIGURATION

In our paper, we use MLP to map context pairs, buffer pairs, or target points to tokens. Then a transformer is applied to the sequence of tokens. We used mixture- of- Gaussian (GMM) head as our main head distribution (more expressive than a single Gaussian head, as demonstrated in Appendix E). In general, we train all models (except the tabular model; see Appendix D.4 for details) with the following settings.

## Training configurations.

- Optimizer: Adam with learning rate \(1 \times 10^{-4}\) (unless stated otherwise), \(\beta = (0.9, 0.999)\) , no weight decay. For TNP w/buffer, we use the same settings, but apply weight decay of 0.01 for stability.

- Scheduler: Cosine schedule with warmup; warmup ratio 0.1 for all experiments. For TNP w/buffer, we use a warmup ratio of 0.05.

- Training loop: 32 epochs.

Embedder. We use a 3- layer MLP with 256 hidden layer dimension and 128 output dimension. There is a skip connection between the first linear layer and the MLP output.

Transformer backbone. This has 6 layers of transformer encoder modules, each with a multihead attention of 4 heads and dimension 128 followed by an MLP feedforward of 2 layers, dimension \(128 \rightarrow 256 \rightarrow 128\) . This is the transformer attending context, buffer, and target set (Appendix A and Appendix B).

Prediction head. Note first that different distribution heads involve individual parameterization structures. Therefore, another layer of distribution- specific NNs is required to process the above transformer outputs. This NN module is considered part of the distribution head (the \(\psi\) in Appendix A and Appendix B).

For our method, TNP- D, and TNP- A, the head consists of 2 layers of MLP with dimension \(128 \rightarrow 256 \rightarrow 3 * D_{y} * N_{\text{components}}\) , where \(D_{y}\) is the output dimension of the problem and \(N_{\text{components}}\) is the number of Gaussian components. The MLP output is then chunked into weights, means, and standard deviations (of the same shape) which parameterize the GMM, and the outputs are sampled in parallel for \(D_{y} > 1\) . We set \(N_{\text{components}} = 20\) for all tasks except for EEG where \(N_{\text{components}} = 8\) .

For TNP- ND, we use the setting from Nguyen & Grover (2022), where the targets are mapped to a mean and a Cholesky matrix, which parameterize the multivariate Gaussian. The mean of each target is mapped by an MLP with dimension \(128 \rightarrow 256 \rightarrow D_{y}\) . The Cholesky matrix requires two steps: (i) the target tokens (conditioned on context via the above transformer backbone) are first decoded into \(H \in \mathbb{R}^{M \times 20}\) by another 3- layer transformer (no positional encoding, 4 heads, each layer with dimension 128 and MLP \(128 \rightarrow 256 \rightarrow 128\) , no mask) and then an MLP projector \((128 \rightarrow 256 \rightarrow 20)\) ; (ii) the Cholesky matrix is taken as \(L = \mathrm{lower}(HH^{T})\) .

Trained model selection. We track the loss value in each epoch as we train the models. The parameters with the best loss value on the validation set are selected for the evaluations on a separate test set.

## D.2 DATASETS

Gaussian Process (GP) Functions. As a first toy case, we test on GP functions (see Rasmussen & Williams 2006 for details of GPs). In this example, a batch contains 128 functions of one dimensional inputs \((D = 1)\) and one dimensional observations \((D_{y} = 1)\) . The inputs are sampled from interval \([- 2, 2]\) using the Sobol sequence. For each batch, we first sample a kernel class from squared- exponential (RBF), Matern- \(3 / 2\) , Matern- \(5 / 2\) with probabilities 0.4, 0.3, and 0.3, respectively. Conditional on the chosen class, each function receives its own kernel hyperparameters: the variance \(\sigma_{f}^{2} \sim \mathrm{Uniform}[0.5, 1.5]\) and the lengthscale \(\ell \sim \mathrm{Uniform}[0.1, 1]\) , broadly covering diverse classes of functions of amplitude around 1. We then sample functions from \(\mathcal{GP}(0, k)\) , where \(k\) represents the sampled kernels, and add i.i.d. Gaussian observation noise with variance \(10^{- 5}\) . The resulting values are randomly partitioned into context, buffer, and target sets. Note that within a batch the kernel class is fixed, whereas the hyperparameters are sampled independently for each function.

During training, we sample the context set size \(N\) between 4 and 192 with a maximum buffer size of 16.

Sawtooth Functions. The second example is the non- Gaussian sawtooth functions (Bruinsma et al., 2023). In this example, a batch contains 128 functions of one dimensional inputs \((D = 1)\) and one dimensional observations \((D_{y} = 1)\) . The inputs are sampled from interval \([- 2, 2]\) using the Sobol sequence. An input \(\mathbf{x}\) and output \(y\) follows:

\[y(\mathbf{x}) = y_{\mathrm{nonise}}(\mathbf{x}) + \epsilon ,\] \[y_{\mathrm{nonise}}(\mathbf{x}) = (\omega (\langle u,\mathbf{x}\rangle -\phi))\mathrm{~mod~}1,\]

where \(u \in \mathbb{R}^{D}\) is a direction sampled uniformly from the unit sphere via \(u = g / \| g\|_{2}\) with \(g \sim \mathcal{N}(0, I_{D})\) ; \(\omega\) , \(\phi\) , and \(\epsilon\) denote the frequency, phase offset, and additive noise, respectively; and the parameters are drawn independently as \(\omega \sim \mathrm{Uniform}[3, 5]\) , \(\phi \sim \mathrm{Uniform}[0, 1]\) , and \(\epsilon \sim \mathcal{N}(0, \sigma^{2})\) with noise scale \(\sigma \sim \mathrm{Uniform}[0.05, 0.1]\) .

During the training, we sample \(N\) between 8 and 128 and the maximum number of buffer is 16.

Electroencephalogram (EEG). The dataset contains 11, 520 trials of 122 subjects from 7 correlated channels with 256 time points each. The output channels are individually standardized to zero mean and unit variance. We randomly select 10 for the test set, reserve 10 for cross- validation, and the remaining for the train set. This leaves 7802 trials for the training and 896 for testing.

During the training, the trials are replicated for 200 times and shuffled. Each batch contains 32 trials sampled from the shuffled set. We select between 4 and 192 of the 256 time points to be context points, 32 buffer points, with the remaining being target points. Each batch has a fixed size of context set.

We evaluate on both interpolation (random masking) and forecasting (temporal masking) tasks using the test subjects. The test set splits the 256 time points into context and target. For interpolation, we sample the specified number of context and target points from the full time sequence (Appendix E). For forecasting, we take the first \(N\) points as context set and the consecutive \(M\) points as target set. Forecasting with \(N = 192\) context and \(M = 64\) target sets involves the full sequence.

Multisensory causal inference model dataset. In the last example, we adopt one of the multisensory causal inference models described in Liu et al. (2025) to build a simulator, which we then use to generate training data (full setup and generation procedure, as well as a description of the experiment, are provided in Appendix D.3). The inputs \(\mathbf{x}\) correspond to the experimentally manipulated variables of the study, namely \(r_{\mathrm{type}}, s_{A}, s_{V}\) and \(V_{\mathrm{level}}\) , where \(r_{\mathrm{type}}\) denotes the task type (auditory vs visual localization), \(s_{A}\) and \(s_{V}\) are the true locations of auditory and visual cues presented to human participants, and \(V_{\mathrm{level}}\) the level of noise applied to visual cues. We first generate sets of input points for the simulator to obtain the outputs \(y\) , which represent the predicted responses.

For training, we construct two datasets from the simulator with different values of \(\rho\) , a variable of the model regulating the level of recalibration of the auditory perceptual range (with \(\rho = 1\) representing no recalibration and \(\rho = 4 / 3\) representing a full recalibration to the visual range, see Appendix D.3 for more details), and train two separate models for each setting. We sample \(N\) between 0 and 400, fix \(N_{t} = 256\) , and set the buffer size to a maximum of 16. For the zero- context case, we introduce one "dummy point", to indicate the absence of context to the model. During evaluation, we use the publicly available dataset obtained from the experiment described in Liu et al. (2025). \(^{8}\) For each of the 15 participants in the study, we extract two non- overlapping subsets of experimental data of 400 trials each. We do so by stratifying on the joint levels of \(V_{\mathrm{level}} \in \{0, 1, 2\}\) and \(r_{\mathrm{type}} \in \{0, 1\}\) (more details on these variables below), and extracting the two sets such that (i) within each split the six \((3 \times 2)\) strata are represented as evenly as possible, and (ii) the per- stratum counts are matched between splits. This yields 30 datasets overall (2 per participant).

For details of the real experiments and the complete data generation setup in the simulator, see Appendix D.3.

Tabular foundation model. We pretrain a task- agnostic tabular model on synthetic data and evaluate it on three UCI datasets. This larger model uses a dedicated training procedure; architectural and training details are in Appendix D.4, and the evaluation protocol is in Appendix D.5.

## D.3 MULTISENSORY CAUSAL INFERENCE MODEL AND EXPERIMENT DETAILS

To probe our method's suitability for Bayesian model comparison, we consider a computational neuroscience study investigating multisensory causal inference, described in Liu et al. (2025).

## D.3.1 ORIGINAL NEUROSCIENCE EXPERIMENT

Stimuli and procedure. In this work, we take into account a subset of the experimental data obtained from 15 human participants who, at each experimental trial, were asked to perform one of two localization tasks, which the authors refer to as bisensory visual (BV) and bisensory auditory (BA) localization. In both cases they were presented with an auditory cue, located at an angle uniformly sampled among \(\{-15^{\circ}, - 10^{\circ}, - 5^{\circ}, 0^{\circ}, 5^{\circ}, 10^{\circ}, 15^{\circ}\}\) from the participant, and a visual one, either at the same location as the auditory one \((\approx 1 / 2\) of trials) or at an angle uniformly sampled between \(- 20^{\circ}\) and \(20^{\circ}\) . They were either asked to report the location of the visual (BV) or the auditory (BA) stimulus on a screen. Here we call those locations \(s_{V}\) and \(s_{A}\) , respectively. The level of noise \(V_{\mathrm{level}}\) associated with the visual stimulus location was experimentally manipulated by modifying the size of the stimulus itself. In practice, this meant presenting a small \((V_{\mathrm{level}} = 0; \approx 1 / 3\) of trials), medium \((V_{\mathrm{level}} = 1; \approx 1 / 3\) of trials) or large \((V_{\mathrm{level}} = 2; \approx 1 / 3\) of trials) visual stimulus.

Each participant completed a total of 1000 trials.

Cognitive models. Here we focus on two versions of the "vanilla" model described in the original paper (Liu et al., 2025). On each trial, the participant is assumed to believe the two stimuli could come from either a common \((C = 1)\) or different \((C = 2)\) source, assigning a fixed prior probability \(p(C = 1) = p_{\mathrm{same}}\) to the former case. Regardless of this, the participant has Gaussian priors over stimuli locations \(p(s_{A}) = \mathcal{N}(s_{A} \mid 0, \sigma_{S}^{2})\) and \(p(s_{V}) = \mathcal{N}(s_{V} \mid 0, \sigma_{S}^{2})\) .

A key assumption of the model is that participants do not have direct access to the true location of the stimuli, but only to noisy auditory and visual percepts, a common feature in Bayesian models of perception (Knill & Pouget, 2004). These percepts are modeled as \(x_{A} = \rho (s + \epsilon_{A})\) and \(x_{V} = s + \epsilon_{V}\)

respectively in case of a common source, and \(x_{A} = \rho (s_{A} + \epsilon_{A})\) and \(x_{V} = s_{V} + \epsilon_{V}\) in case of separate sources. Here \(s = s_{A} = s_{V}\) represents their common location when \(C = 1\) , while \(\epsilon_{A} \sim \mathcal{N}(0, \sigma_{A}^{2})\) and \(\epsilon_{V} \sim \mathcal{N}(0, \sigma_{V}^{2})\) represent the auditory and visual perceptual noise. While \(\sigma_{A}\) is assumed to be fixed, \(\sigma_{V}\) can assume three separate values \((\sigma_{V}^{(\mathrm{low})}, \sigma_{V}^{(\mathrm{med})}, \sigma_{V}^{(\mathrm{high})})\) based on the (experimentally manipulated) size of the visual stimulus \(V_{\mathrm{level}}\) . Finally, \(\rho\) represents a "recalibration" factor to account for the fact that the range of auditory stimuli \((30^{\circ})\) is different from that of visual ones \((40^{\circ})\) . In our experiment, this is the factor that differentiates the two models we set out to compare: in the first, we set \(\rho = 1\) ; in the second, we set \(\rho = \frac{4}{3}\) (thus re- mapping auditory percepts to the same scale as visual ones).

Here we describe a BA trial, but the following is easily generalizable to BV ones. When asked about the location of the auditory stimulus, participants are assumed to consider both scenarios (common vs different sources) by evaluating

\[p(s\mid C = 1) = p(s\mid x_{A},x_{V},\sigma_{A},\sigma_{V},\sigma_{S}),\] \[p(s_{A}\mid C = 2) = p(s_{A}\mid x_{A},\sigma_{A},\sigma_{S}),\]

as well as

\[p(C\mid x_{A},x_{V},\sigma_{A},\sigma_{V},\sigma_{S},p_{\mathrm{same}}).\]

The final estimate \(\hat{s}_{A}\) of the location is then inferred by weighting the two hypotheses (common vs separate sources) by their posterior probability, so

\[\begin{array}{r l} & {\hat{s}_{A} = p(C = 1\mid x_{A},x_{V},\sigma_{A},\sigma_{V},\sigma_{S},p_{\mathrm{same}})\int_{-\infty}^{\infty}s\cdot p(s\mid C = 1)d s + }\\ & {\qquad p(C = 2\mid x_{A},x_{V},\sigma_{A},\sigma_{V},\sigma_{S},p_{\mathrm{same}})\int_{-\infty}^{\infty}s_{A}\cdot p(s_{A}\mid C = 2)d s_{A}.} \end{array} \quad (7)\]

Finally, the response of the participant is modeled as \(y \sim \mathcal{N}(\hat{s}_{A}, \sigma_{M}^{2})\) with a probability of \(1 - \lambda\) , and \(y \sim \mathrm{Uniform}[- 45, 45]\) with a probability of \(\lambda\) . Here \(\lambda\) represents the "lapse rate", or the probability of a participant being distracted/disengaged and giving a random answer (which we fix at 0.02), while \(\sigma_{M}\) represents motor noise.

Both models thus have 7 free parameters, which we re- parametrize as \(\log \sigma_{V}^{(\mathrm{low})}, \log \sigma_{V}^{(\mathrm{med})}, \log \sigma_{V}^{(\mathrm{high})}, \log \sigma_{A}, \log \sigma_{S}, \log \sigma_{M}\) and \(\log \mathrm{it}_{\mathrm{same}}\) for the purposes of simulation and model- fitting.

## D.3.2 SIMULATION

For training all models, we produce \(\sim 1.5\) millions synthetic datasets. In what follows we go through the simulation of a single trial. As trials are independent from one another, generating more of them simply involves repeating this process.

Stimuli. Following the setup used in Liu et al. (2025), we sample \(s_{A} \sim \mathrm{Uniform}\{- 15, - 10, - 5, 0, 5, 10, 15\}\) and \(C \sim \mathrm{Uniform}\{1, 2\}\) . Then we either sample \(s_{V} \sim \mathrm{Uniform}[- 20, 20]\) as a continuous variable (if \(C = 2\) ) or we set \(s_{V} = s_{A}\) (if \(C = 1\) ). We then sample \(V_{\mathrm{level}} \sim \mathrm{Uniform}\{0, 1, 2\}\) , representing the perceptual noise associated with \(s_{V}\) . This regulates whether \(\sigma_{V} = \sigma_{V}^{(\mathrm{low})}, \sigma_{V} = \sigma_{V}^{(\mathrm{med})}\) or \(\sigma_{V} = \sigma_{V}^{(\mathrm{high})}\) .

Finally, we sample \(r_{\mathrm{type}} \sim \mathrm{Uniform}\{0, 1\}\) , representing the task (BV if \(r_{\mathrm{type}} = 0\) , BA if \(r_{\mathrm{type}} = 1\) ).

Parameters. For each synthetic dataset, the prior generative distributions for the 7 free parameters are Gaussians truncated at two standard deviations above and below the mean. We use \(\mathcal{N}_{\mathrm{truncated}}(\mu , \sigma^{2})\) to denote such distributions, with \(\mu\) being the mean and \(\sigma\) the standard deviation. Similarly to empirical Bayes approaches (Murphy, 2023), we use maximum- likelihood estimates of the individual participants' parameters from Liu et al. (2025) as a guide for setting these priors, so

as to generate realistic parameter ranges. The parameter distributions we use in this work are:

\[\log \sigma_{V}^{(\mathrm{low})}\sim \mathcal{N}_{\mathrm{truncated}}(0,1.5^{2});\] \[\log \sigma_{V}^{(\mathrm{med})}\sim \mathcal{N}_{\mathrm{truncated}}(\log \sigma_{V}^{(\mathrm{low})} + 1,1^{2});\] \[\log \sigma_{V}^{(\mathrm{high})}\sim \mathcal{N}_{\mathrm{truncated}}(\log \sigma_{V}^{(\mathrm{med})} + 0.75,0.5^{2});\] \[\log \sigma_{A}\sim \mathcal{N}_{\mathrm{truncated}}(1.75,0.5^{2});\] \[\log \sigma_{S}\sim \mathcal{N}_{\mathrm{truncated}}(2.5,1^{2});\] \[\log \sigma_{M}\sim \mathcal{N}_{\mathrm{truncated}}(0,0.5^{2});\] \[\log \mathrm{itp}_{\mathrm{same}}\sim \mathcal{N}_{\mathrm{truncated}}(1.5,1.5^{2}).\]

Note that \(\log \sigma_{V}^{(\mathrm{low})}\) , \(\log \sigma_{V}^{(\mathrm{med})}\) , and \(\log \sigma_{V}^{(\mathrm{high})}\) are not independent from each other, but carry the assumption that in most cases \(\log \sigma_{V}^{(\mathrm{low})} \lesssim \log \sigma_{V}^{(\mathrm{med})} \lesssim \log \sigma_{V}^{(\mathrm{high})}\) , which reflects the intent of the experimental manipulation of \(V_{\mathrm{level}}\) .

Responses. Here we describe a scenario in which \(r_{\mathrm{type}} = 1\) (BA trial), but the process is the same for \(r_{\mathrm{type}} = 0\) . In simulating the responses, we follow the hierarchical structure specified by the model. First we computed the sensory percepts \(x_{A} = \rho (s_{A} + \epsilon_{A})\) and \(x_{V} = s_{V} + \epsilon_{V}\) by sampling \(\epsilon_{A} \sim \mathcal{N}(0, \sigma_{A}^{2})\) and \(\epsilon_{V} \sim \mathcal{N}(0, \sigma_{V}^{2})\) . We then evaluate \(\hat{s}_{A}\) (recall we are considering a BA trial) as in Eq. (7), and sample the final response as either \(y \sim \mathcal{N}(\hat{s}_{A}, \sigma_{M}^{2})\) or \(y \sim \mathrm{Uniform}[- 45, 45]\) , with a probability regulated by the lapse rate \(\lambda\) (which we set to 0.02, see above).

## D.3.3 GROUND-TRUTH ACQUISITION

Here we describe how we obtained our log marginal likelihood (LML) estimates (in the form of lower bounds, see below), which we then use as ground- truth to compare our approach to baselines.

Problem setting. Fitting the cognitive model to a dataset involves finding the posterior over model parameters given empirical data and model

\[p(\pmb {\theta}\mid \mathbf{y},\mathbf{X},\pmb {\rho}) = \frac{p(\mathbf{y}\mid\pmb{\theta},\mathbf{X},\pmb{\rho})p(\pmb{\theta})}{p(\mathbf{y}\mid\mathbf{X},\pmb{\rho})}, \quad (8)\]

where

\[\pmb {\theta} = \{\log \sigma_{V}^{(\mathrm{low})},\log \sigma_{V}^{(\mathrm{med})},\log \sigma_{V}^{(\mathrm{high})},\log \sigma_{A},\log \sigma_{S},\log \sigma_{M},\log \mathrm{itp}_{\mathrm{same}}\} ,\] \[\qquad \mathbf{X} = \{s_{A}^{(t)},s_{V}^{(t)},V_{\mathrm{level}}^{(t)},r_{\mathrm{type}}^{(t)}\}_{t = 1}^{400},\]

and

\[\mathbf{y} = \{y^{(t)}\}_{t = 1}^{400}.\]

Here \(t\) represents the trial number within the dataset (recall we are using data splits of 400 trials each, see Appendix D.2), and we set \(p(\pmb {\theta})\) to the truncated Gaussians we use for sampling the parameters in our simulation (see Appendix D.3.2), with probability density of values beyond the truncation boundaries set to a "floor value" of \(\mathcal{N}(5 \mid 0, 1)\) .

While the posterior over parameters is often instrumental in answering scientific questions, the crucial quantity we are interested in estimating is the model evidence (also called marginal likelihood) \(p(\mathbf{y} \mid \mathbf{X}, \rho)\) (i.e., the denominator in Eq. (8)), as it represents a straightforward metric for model selection. In fact, assuming a flat prior over models \(p(\rho = 1) = p(\rho = 4 / 3) = 0.5\) , the model evidence as a function of \(\rho\) represents the unnormalized posterior over models.

Stacking Variational Bayesian Monte Carlo. To compute a reliable estimate of the marginal likelihood to use as our ground- truth, we use Stacking Variational Bayesian Monte Carlo (S- VBMC, Silvestrin et al., 2025). This is a principled approach to merge ("stack") approximate posteriors generated by a set of independent runs of its parent algorithm, Variational Bayesian Monte Carlo (VBMC, Acerbi, 2018; 2020). This is done in a simple post- processing step, which has been shown to greatly improve the approximate posterior quality in a variety of challenging settings. In addition to a posterior distribution, S- VBMC outputs an estimate of the evidence lower bound (ELBO),

which, as the name suggests, is a lower bound on the (log) model evidence (Blei et al., 2017), the quantity we are interested in for model comparison. As the approximation of the posterior approaches the true one, this quantity gets closer to the true model evidence, with equality when the approximation is perfect. As S- VBMC proved very effective in computational neuroscience problems (Silvestrin et al., 2025), including one very similar to the one considered here (Acerbi et al., 2018), we deem it a suitable method for estimating a lower bound on model evidence to use as a ground- truth.

While an in- depth description of S- VBMC and VBMC is beyond the scope of this work (an interested reader should refer to the original papers cited above), in the following paragraphs we briefly report details of our implementation of both.

VBMC implementation details. To obtain an approximate posterior, the Python implementation of VBMC (Huggins et al., 2023) requires absolute and plausible upper and lower bounds for each parameter. We use the sampling bounds defined in Appendix D.3.2 as absolute bounds, and replicate the process considering 1.5 standard deviations (as opposed to 2) from the mean to establish the plausible ones.

Another required input is a target density function (i.e., the unnormalized posterior), for which we use the numerator of Eq. (8), \(p(\mathbf{y} \mid \theta , \mathbf{X}, \rho)p(\theta)\) . We do this both with \(\rho = 1\) and \(\rho = 4 / 3\) , representing the two models we set out to compare.

Finally, VBMC requires a starting point in the parameter space, which we uniformly sample between plausible bounds independently for each inference run.

S- VBMC implementation details. After obtaining 20 converging VBMC runs for each of our 30 datasets (2 for each of the 15 participants, see Appendix D.2) for both models, we stack the resulting posteriors with S- VBMC. We maintain the default settings, therefore the only inputs required are the VBMC runs themselves. With this, we obtain a total of 60 "stacked" ELBOs (two per each dataset, corresponding to our two competing models) to use as ground- truth.

## D.4 TABULAR MODEL DETAILS

This section describes the TabICL model and explains how the training dataset was generated. Notably, the base architecture used for this tabular data example is different from the one used in the other experiments, highlighting the broad applicability of our method.

## D.4.1 ARCHITECTURE

Set encoder. We reuse the first two stages of TabICL (Qu et al., 2025) without modification: the distribution- aware column processor \((\mathrm{TF}_{\mathrm{col}}\) , implemented with induced self- attention blocks) followed by the context- aware row- wise transformer \((\mathrm{TF}_{\mathrm{row}})\) with RoPE. Scalars are mapped by a \(1 \rightarrow 128\) linear layer; each column is then processed across rows by an ISAB stack (Lee et al., 2019) with three blocks, four heads, 128 inducing points, feed- forward hidden dimension of 256, The row- wise encoder has three layers with four heads, feed- forward hidden dimension of 256, and RoPE base 100,000. We prepend two [CLS] tokens per row and concatenate their outputs, yielding a 256- dimensional row embedding \((2 \times 128)\) . We use at most ten features per table.

Tokenization and additive target encoding. The set encoder produces one row token per sample for context, buffer, and target rows (dimension 128; only selects the subset of the vector corresponding to the [CLS] token dimensions). Context and buffer tokens receive the target value additively via a small target encoder (linear \(1 \rightarrow 128\) . Buffer tokens also receive a learned positional embedding indicating their autoregressive index (up to 32 positions). This keeps labels additive, lets us compute the set encoder once, and makes the buffer explicit at the token level.

Dataset- wise ICL with a buffered mask. On top of these tokens we run a dataset- wise transformer with twelve layers and four heads, model width 128, and feed- forward size 256. The attention mask is the only architectural change relative to TabICL: context attends bidirectionally and is read- only at inference; the buffer uses strictly causal self- attention; target queries attend to the cached context and to the causal prefix of the buffer; there are no edges into context from buffer or targets. The maximum buffer size is 32 tokens and we query 512 targets per task.

Prediction head. Predictions use a GMM head with 20 components and a minimum standard deviation of \(10^{- 3}\) .

Caching. The column and row set encoder is computed once for all rows. During autoregressive decoding we cache keys/values for the context once and update only the buffer cache, so the same context cache is reused across parallel generations.

## D.4.2 DATA GENERATION AND PREPROCESSING

SCM prior and task family. We generate datasets with the MLP- based structured causal model (SCM) prior in the style of Hollmann et al. (2023), following the dataset- wise, set- encoded regime of TabICL (Qu et al., 2025). Concretely, we first sample a DAG with layered (MLP- style) connectivity and then define each variable \(c\) as \(c = f(\mathrm{Pa}(c)) + \epsilon\) , where \(\mathrm{Pa}(c)\) are its parents, \(f\) is a small MLP with nonlinearity, and \(\epsilon\) is independent noise. Unless stated otherwise, we sample the feature dimension \(d \in [1, 10]\) , and per- task context sizes \(N \in [8, 1024]\) ; targets are continuous responses with dataset- specific noise levels. The cause sampler follows the TabPFN prior (including mixed marginals); the SCM therefore yields columns that may be non- Gaussian or discrete at source, which we handle with the TabICL preprocessing described below.

Sampling of task partitions. For each generated dataset we draw a random partition \((C, B, \mathcal{T})\) with \(N \sim \mathrm{Uniform}\{8, \ldots , 1024\}\) , buffer capacity fixed at \(K = 32\) , and target count \(M = 512\) . Per batch, we fix \((d, N, K, M)\) across tasks to avoid padding and stack samples directly.

Preprocessing. We adopt the TabICL PreprocessingPipeline and fit it on context features only. The fitted transform is then applied to context, buffer, and target features. Regression targets are standardized using context statistics, i.e., \(\tilde{y} = (y - \mu_{y,c}) / \sigma_{y,c}\) , and the same \((\mu , \sigma)\) are used for buffer and targets. No missing values are synthesized by the SCM generator.

Summary of preprocessing pipeline. We use a three- stage, per- column pipeline following Qu et al. (2025): (i) standard scaling; (ii) normalization (power, i.e., Yeo- Johnson); and (iii) outlier handling via a \(z\) - score threshold \(\tau = 4.0\) . At transform time, values outside the fitted range are clipped to the training (context) min/max before normalization, mirroring TabICL's behavior.

## D.4.3 TRAINING PROCEDURE

We train with AdamW (learning rate \(1 \times 10^{- 4}\) , \(\beta = (0.9, 0.95)\) , weight decay 0.0), batch size 64 datasets per step, gradient clipping at 0.5, and dropout 0.0 throughout the backbone. Mixed- precision training uses AMP with bfloat16. All runs use float32 tensors at the data interface. A cosine schedule with warmup is used (cosine_with_warmup); warmup_steps \(= 2000\) takes precedence over the nominal warmup_ratio \(= 0.20\) ; num_cycles \(= 1\) . Automatic mixed precision is enabled with amp_dtype=bfloat16. Each training step draws a batch of 64 independent tasks (datasets) with feature dimension \(d\) sampled from \(\{1, \ldots , 10\}\) and context size \(N\) from \(\{8, \ldots , 1024\}\) ; buffer size and target count are fixed at \(K = 32\) and \(M = 512\) . Training is capped at max_steps \(= 160,000\) , i.e., one epoch effective duration. This corresponds to approximately \(64 \times 160,000 = 10.24\) million synthetic tasks seen during pretraining. The global data seed is 123. We trained the model on a single NVIDIA A100 80 GB GPU for approximately 3 days.

## D.5 EVALUATION DETAILS

In this paper log- likelihood values are always averaged (LL divided by the number of target points \(M\) ).

GP & Sawtooth functions. We evaluate likelihood values over 1024 functions, each repeated 4 times with models trained on different seeds and context sizes \(N = 8, 16, 32, 64, 128\) (statistics of \(1024 \times 4 \times 5\) evaluations). Each likelihood evaluation is an average of 128 permutations (log averaged likelihood). In other words, we have \(1024 \times 4 \times 5\) averaged likelihoods, and each averaged value merges 128 orders of the target set.

Table A1: Head comparison on synthetic function. We compare average log-likelihood \((\uparrow)\) results on our main GMM head and on standard Gaussian distribution head.

| Model | TNP-D (AR) | TNP-D (Ind) | TNP w/ buffer (K=16) | TNP w/ buffer (K=4) | TNP w/ buffer (K=1) | TNP-D-Gaussian (AR) | TNP-D-Gaussian (Ind) | TNP Gaussian w/ buffer (K=16) | TNP Gaussian w/ buffer (K=4) | TNP Gaussian w/ buffer (K=1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GP (M = 16) | 2.57 (0.020) | 2.22 (0.022) | 2.51 (0.019) | 2.55 (0.019) | 2.56 (0.019) | 2.50 (0.019) | 2.13 (0.023) | 2.48 (0.019) | 2.53 (0.019) | 2.53 (0.019) |
| GP (M = 128) | 3.29 (0.013) | 2.15 (0.022) | 3.27 (0.013) | 3.28 (0.013) | 3.29 (0.013) | 3.23 (0.013) | 2.06 (0.023) | 3.25 (0.013) | 3.27 (0.013) | 3.27 (0.013) |
| Sawtooth (M = 16) | 1.05 (0.004) | 0.94 (0.005) | 1.00 (0.005) | 1.08 (0.004) | 1.09 (0.004) | 0.96 (0.004) | 0.82 (0.006) | 0.85 (0.006) | 0.98 (0.004) | 0.99 (0.004) |
| Sawtooth (M = 128) | 1.15 (0.003) | 1.16 (0.003) | 1.15 (0.003) | 1.16 (0.003) | 1.16 (0.003) | 1.10 (0.003) | 0.82 (0.005) | 1.10 (0.003) | 1.11 (0.003) | 1.11 (0.003) |

EEG data. We train each model once with a fixed seed; the evaluations are over 896 trials from 20 subjects held out during training, each repeated with \(N = 8, 16, 32, 64, 128, 192\) . For the EEG forecasting, the target set consists of time points immediately after context points, and, in the main results (Table 1), the target set permutations are applied, as done in Bruinsma et al. (2023). We additionally demonstrate in appendix Table A3 that forecasting with permuted target set outperforms fixed sorted target. The number of permutations we apply is 128.

Multisensory causal inference model. We train one model for each setting of \(\rho\) ( \(\rho = 1\) and \(\rho = 4 / 3\) ). In the model selection scenario, the full 400- point dataset from each of the 30 batches is used as the target, and we evaluate the LML across all cases. This procedure is repeated 5 times, with 128 different sequence permutations per run. In the data prediction scenario, we first select the winning model from the model selection stage, and then compute log- likelihoods on the same 30 batches, each repeated with \(N = 8, 16, 32, 64, 128, 256\) . The results of both experiments are summarized in Table 2. Here we also use 128 permutation for all batches.

Tabular foundation model. We pretrain a task- agnostic tabular model on synthetic data (Appendix D.4) and evaluate it on three UCI datasets: Individual Household Electric Power Consumption \(^{9}\) , Gas Turbine CO and NOx Emission DataSet \(^{10}\) , Bike Sharing \(^{11}\) , Jena climate dataset \(^{12}\) , Power Consumption of Tetouan City \(^{13}\) , and California Housing Price \(^{14}\) .

For each dataset, we evaluate likelihood values over 16 randomly sampled subsets. The context and target sets are set to \(N = 128\) , \(M = 32\) . Each likelihood evaluation is an average of 128 permutations.

Table A2: Average Log-likelihood \((\uparrow)\) results on synthetic functions and EEG example. Supplementary results of Table 1 on larger target set and various deployed \(K\) . When \(M > K\) , we evaluate every \(K\) targets once and perform AR for \(M / K\) steps.   

| | TNP-D (AR) | TNP-D (Ind) | TNP-ND | TNP-A |
| :--- | :--- | :--- | :--- | :--- |
| GP (M = 16) | 2.57 (0.020) | 2.22 (0.022) | 0.80 (0.082) | 2.24 (0.018) |
| GP (M = 128) | 3.29 (0.013) | 2.15 (0.022) | 2.27 (0.023) | 3.10 (0.012) |
| Sawtooth (M = 16) | 1.05 (0.004) | 0.94 (0.005) | -0.43 (0.008) | 0.98 (0.004) |
| Sawtooth (M = 128) | 1.14 (0.003) | 0.94 (0.005) | 0.39 (0.005) | 1.12 (0.003) |
| EEG-Int (M = 16) | 0.51 (0.013) | 0.36 (0.014) | 0.46 (0.011) | 0.58 (0.014) |
| EEG-Int (M = 64) | 0.88 (0.011) | 0.35 (0.014) | 0.50 (0.010) | 0.95 (0.012) |
| EEG-For (M = 16) | 1.07 (0.004) | -0.74 (0.008) | -0.04 (0.005) | 1.23 (0.003) |
| EEG-For (M = 64) | 1.12 (0.003) | -1.08 (0.007) | -0.23 (0.004) | 1.20 (0.003) |

| | TNP w/ buffer (K=16) | TNP w/ buffer (K=4) | TNP w/ buffer (K=1) |
| :--- | :--- | :--- | :--- |
| GP (M = 16) | 2.51 (0.019) | 2.55 (0.019) | 2.56 (0.019) |
| GP (M = 128) | 3.27 (0.013) | 3.28 (0.013) | 3.29 (0.013) |
| Sawtooth (M = 16) | 1.00 (0.005) | 1.08 (0.004) | 1.09 (0.004) |
| Sawtooth (M = 128) | 1.15 (0.003) | 1.16 (0.003) | 1.16 (0.003) |
| EEG-Int (M = 16) | 0.52 (0.013) | 0.54 (0.014) | 0.54 (0.014) |
| EEG-Int (M = 64) | 0.90 (0.011) | 0.91 (0.011) | 0.91 (0.011) |
| EEG-For (M = 16) | 0.85 (0.004) | 1.17 (0.003) | 1.21 (0.003) |
| EEG-For (M = 64) | 1.12 (0.003) | 1.18 (0.003) | 1.19 (0.003) |

## E ADDITIONAL LOG-PREDICTIVE DENSITY RESULTS ON SYNTHETIC AND EEG TASKS

## E.1 PREDICTIVE POWER OF DIFFERENT HEADS

In this paper, we use GMM as our prediction head. We compare the predictive performance of GMM to standard Gaussian distribution head. In Table A1, GMM is able to achieve better predictive performance, particularly on the non- Gaussian Sawtooth functions.

## E.2 RESULTS OF LARGER \(M\)

As a supplementary results of Table 1, we evaluate log- likelihood values on a larger target set. For TNP w/ buffer, we evaluate \(K\) points per Algorithm 2 and proceed to the next target subsets by conditioning on the context and evaluated points. This requires \(M / K\) steps of evaluations. The results are reported in Table A2. As we decrease the number of buffer targets \(K^{15}\) , the performance of our TNP w/ buffer becomes stronger, while more iterations (and thus computational time) are required.

Table A3: EEG forecasting w/ and w/o target set permutation. The target set of EEG forecasting are the points immediately after the context set. Our main paper applies permutation to the target set while this table compares against forecasting of fixed temporal order (sorted).

| | TNP-D (AR) | TNP-D (Ind) | TNP-ND | TNP-A | TNP w/ buffer (K=16) | TNP w/ buffer (K=4) | TNP w/ buffer (K=1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EEG-For (M = 16)** | 1.07 (0.004) | -0.74 (0.008) | -0.04 (0.005) | 1.23 (0.003) | 0.85 (0.004) | 1.17 (0.003) | 1.21 (0.003) |
| **EEG-For (M = 16, sorted)** | 0.85 (0.005) | -0.74 (0.008) | -0.004 (0.005) | 1.14 (0.004) | 0.76 (0.006) | 0.87 (0.005) | 1.09 (0.004) |
| **EEG-For (M = 64)** | 1.12 (0.003) | -1.08 (0.007) | -0.23 (0.004) | 1.20 (0.003) | 1.12 (0.003) | 1.18 (0.003) | 1.19 (0.003) |
| **EEG-For (M = 64, sorted)** | 0.89 (0.005) | -1.08 (0.007) | -0.23 (0.004) | 1.16 (0.003) | 0.78 (0.005) | 0.89 (0.004) | 1.11 (0.004) |

Table A4: Multisensory causal inference model selection extra results. Supplement for Table 2 on model comparison case with extra evaluation on \(K = 4\) and \(R^{2}\) metrics for LML and \(\Delta\) LML.

| Metric | TNP-D (AR) | TNP-D (Ind) | TNP-ND | TNP-A | TNP w/ buffer (K=16) | TNP w/ buffer (K=4) | TNP w/ buffer (K=1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LML RMSE (↓) | 3.10 (0.005) | 86.96 (0.000) | 208.51 (0.041) | 4.75 (0.012) | 3.56 (0.004) | 3.48 (0.002) | 3.47 (0.004) |
| $\Delta$LML RMSE (↓) | 2.44 (0.008) | 36.18 (0.000) | 25.60 (0.023) | 3.29 (0.019) | 2.60 (0.010) | 2.59 (0.009) | 2.59 (0.011) |
| LML $R^2$ (↑) | 1.00 (0.000) | -0.43 (0.000) | -7.22 (0.003) | 1.00 (0.000) | 1.00 (0.000) | 1.00 (0.000) | 1.00 (0.000) |
| $\Delta$LML $R^2$ (↑) | 0.93 (0.001) | -14.47 (0.000) | -6.74 (0.014) | 0.87 (0.002) | 0.92 (0.001) | 0.92 (0.001) | 0.92 (0.001) |

## E.3 EEG FORECASTING W/ AND W/O TARGET PERMUTATION

In our main paper, the EEG forecasting task is evaluated with the permuted target set, following the procedure of Bruinsma et al. (2023). We repeat the experiment by forecasting the target of a fixed temporal order. In Table A3, we show that averaging over random target order as done in the paper, provides better overall performance.

## F ADDITIONAL MULTISENSORY CAUSAL INFERENCE MODEL RESULTS

As supplementary results to Table 2, we include additional metrics and evaluation settings. Specifically, for the model comparison task, we report the coefficient of determination ( \(R^{2}\) ) for both the LML and \(\Delta\) LML with respect to the ground- truth (see Table A4). For the data prediction task, we present results with a larger target size of \(M = 128\) (see Table A5). In addition, for completeness, we evaluate both the model comparison and data prediction tasks with \(K = 4\) . With varying \(K\) , we observe little to almost no performance degradation compared to TNP- D AR, especially for the data prediction case.

## G ADDITIONAL TABULAR FOUNDATION MODEL RESULTS

We report results for an intermediate context size ( \(N = 256\) ) in Table A6. Consistent with our main findings, AR w/ buffer matches standard AR within standard errors across all tasks, while both autoregressive methods outperform independent predictions on forecasting tasks.

Table A5: Multisensory causal inference model data prediction task normalized log-likelihood \((\uparrow)\) results. Supplementary results of Table 2, with extra evaluation on \(K = 4\) and on larger target set \(M = 128\)

| | TNP-D | TNP-ND | TNP-A | TNP w/ buffer (K=16) | TNP w/ buffer (K=4) | TNP w/ buffer (K=1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pred LL (M=16) | -2.76 (0.021) | -2.77 (0.025) | -3.12 (0.019) | -2.76 (0.024) | -2.76 (0.024) | -2.76 (0.024) |
| Pred LL (M=128) | -2.71 (0.015) | -2.74 (0.016) | -3.17 (0.012) | -2.71 (0.015) | -2.71 (0.015) | -2.71 (0.015) |

Table A6: Average log-predictive density \((\uparrow)\) results on UCI datasets with TabICL ( \(N = 256\) ). Results are reported as mean and standard error over 16 randomly sampled mini-datasets ( \(M = 32\) ) for interpolation (Int) and forecasting (For) tasks.

| | Electric Cons. Int | Electric Cons. For | Gas Turbine Int | Gas Turbine For | Bike Sharing Int | Bike Sharing For | Tetouan Int | Tetouan For | Jena Int | Jena For | Cali. Int |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INTERMEDIATE CONTEXT REGIME (N = 256)** | | | | | | | | | | | |
| Independent | 1.65 (0.15) | 1.21 (0.30) | -0.44 (0.16) | -1.06 (0.32) | 2.38 (0.05) | 1.98 (0.11) | 0.56 (0.08) | -1.45 (0.44) | 2.03 (0.06) | 0.59 (0.18) | -0.55 (0.12) |
| Standard AR | 1.67 (0.14) | 1.57 (0.22) | -0.44 (0.16) | -0.73 (0.23) | 2.39 (0.05) | 2.24 (0.09) | 0.57 (0.08) | 0.39 (0.18) | 2.03 (0.06) | 1.45 (0.15) | -0.54 (0.12) |
| AR w/buffer | 1.67 (0.14) | 1.56 (0.21) | -0.44 (0.16) | -0.76 (0.23) | 2.38 (0.05) | 2.23 (0.08) | 0.57 (0.08) | 0.28 (0.20) | 2.03 (0.06) | 1.30 (0.17) | -0.54 (0.12) |

## H ABLATIONS AND EXTRA EXPERIMENTS

## H.1 COMPARISON TO NON-PERMUTATION-INVARIANT TRANSFORMERS

To isolate the effect of permutation invariance in the context set, we replace our model with a plain autoregressive decoder Transformer that treats the context as a fixed input sequence. This sequential baseline performs substantially worse than our method in the GP task and across context sizes (see Fig. A14), indicating that explicitly maintaining permutation invariance over the context set – or at least part of it – is critical for performance.

## H.2 POSITIONAL EMBEDDINGS ABLATION

We also trained our method without positional embeddings in the buffer and performed evaluations with \(M = K = 16\) , as shown in Table A7, and observed no statistically significant difference in predictive performance. This aligns with findings in causal- transformer work showing that models can infer positional structure without explicit encodings (Haviv et al., 2022; Irie, 2024; Zuo et al., 2025). While not strictly required, positional embeddings may still support future extensions, such as scaling to larger buffer sizes via ALiBi (Press et al., 2022) or RoPE (Su et al., 2024).

Table A7: Average joint predictive log- density \((\uparrow)\) for positional embedding ablation on the GP task; reported as mean (SEM).

| TNP-D-AR | TNP-D-Ind | TNP-ND | TNP-A | Ours w/ pos. emb | Ours w/o pos. emb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2.57 (0.02) | 2.22 (0.02) | 0.80 (0.08) | 2.24 (0.02) | 2.51 (0.02) | 2.51 (0.02) |

## H.3 NUMBER OF SAMPLES ORDER AVERAGING ABLATION

We study the effect of the number of sequence samples (permutations) used for order averaging. We report results on the multisensory causal inference task, where our method (with buffer size \(K = 16\) ) is used to compute the LML of a dataset by averaging over multiple permutations. As shown in Fig. A15, increasing the number of permutations reduces the RMSE of the estimated joint

> **Image description.** A multi-panel line graph consisting of three separate plots arranged horizontally, illustrating the comparison of different model performances based on their "Joint log-density" across varying numbers of context points.
>
> The figure is divided into three panels, each representing a different number of targets, labeled $M=1$ (left), $M=4$ (middle), and $M=16$ (right).
>
> **Common Elements (Axes and Legend):**
> *   **Y-axis:** All three panels share a vertical axis labeled "Joint log-density," ranging from 0.0 to 3.2, with major tick marks every 0.2 units.
> *   **X-axis:** All three panels share a horizontal axis labeled "Number of context points $N$," with discrete data points marked at 8, 16, 32, 64, and 128.
> *   **Data Series (Legend):** Four distinct lines represent different models:
>     *   Blue solid line: "TNP w/ buffer, K=M"
>     *   Green solid line: "TNP-D-AR"
>     *   Orange solid line: "Decoder Transformer"
>     *   Red dashed line: "TNP w/ buffer, K=1"
>
> **Visual Trends and Performance:**
> In all three panels ($M=1, M=4, M=16$), a consistent positive trend is observed: as the "Number of context points $N$" increases, the "Joint log-density" for all models increases.
>
> The relative performance of the models remains consistent across all values of $M$:
> 1.  The blue line ("TNP w/ buffer, K=M") consistently achieves the highest joint log-density.
> 2.  The green line ("TNP-D-AR") generally performs second.
> 3.  The orange line ("Decoder Transformer") follows in third place.
> 4.  The red dashed line ("TNP w/ buffer, K=1") consistently shows the lowest joint log-density across all tested values of $N$.
>
> The visual gap between the top-performing models (blue and green) and the lowest-performing model (red) appears to widen as the number of context points $N$ increases, suggesting that the performance advantage of the top models scales with the amount of input data.

<center>Figure A14: Average joint predictive log-density \((\uparrow)\) on the GP regression task comparing the Decoder Transformer models with ours and gold standard TNP-D-AR on varying number of context points \(N\) and number of targets \(M = 1,4,16\) . </center>

> **Image description.** A composite image consisting of two side-by-side line graphs, labeled "LML" (left) and "$\Delta$ LML" (right), which compare the Root Mean Square Error (RMSE) of three different models across varying numbers of sequence samples.
>
> **General Structure and Axes:**
> Both graphs share a common vertical axis (Y-axis) labeled "RMSE," scaled from 0 to 5. The horizontal axis (X-axis) is labeled "N. of sequence samples" and is scaled logarithmically from 2 to 128, with major tick marks at 2, 4, 16, 32, 64, and 128.
>
> **Legend and Models:**
> A legend at the bottom identifies the three models being compared:
> *   "Ours" (represented by a blue line with diamond markers).
> *   "TNP-A" (represented by a pink/magenta line with diamond markers).
> *   "TNP-D-AR" (represented by a green line with diamond markers).
>
> **Panel Analysis (LML - Left Graph):**
> The left panel, titled "LML," displays the RMSE for the LML estimation.
> *   **Trends:** All three models exhibit a slight downward trend in RMSE as the number of sequence samples increases, indicating improved performance with more data.
> *   **Performance:** The "TNP-A" model consistently maintains the highest RMSE across all sample sizes. The "Ours" model (blue) and the "TNP-D-AR" model (green) perform closely, with the "Ours" model generally showing a slightly lower RMSE than "TNP-D-AR," particularly at smaller sample sizes (e.g., at N=2, "Ours" is approximately 3.5, while "TNP-D-AR" is approximately 3.2).
>
> **Panel Analysis ($\Delta$ LML - Right Graph):**
> The right panel, titled "$\Delta$ LML," displays the RMSE for the $\Delta$ LML estimation.
> *   **Trends:** Similar to the LML panel, all three models show a slight decrease in RMSE as the number of sequence samples increases.
> *   **Performance:** The "TNP-A" model again maintains the highest RMSE across the entire range of sequence samples. The "Ours" model (blue) and the "TNP-D-AR" model (green) remain competitive, with "Ours" generally achieving a lower RMSE than "TNP-D-AR." At N=2, "Ours" is approximately 3.0, and "TNP-D-AR" is approximately 2.8. By N=128, all three models cluster between an RMSE of 2.0 and 3.0.
>
> In summary, both graphs demonstrate that while increasing the number of sequence samples generally reduces RMSE for all models, the "TNP-A" model consistently performs the worst, while the proposed "Ours" model and "TNP-D-AR" model show comparable and strong performance.

<center>Figure A15: Average RMSE \((\downarrow)\) on the LML (left) and \(\Delta\) LML (right) estimation in the multisensory causal inference task for different numbers of sample permutations. </center>

density relative to the true joint at a rate comparable to existing autoregressive baselines, while our method remains significantly faster. This indicates that for our proposed method, order averaging does not introduce additional performance degradation relative to the gold standard baselines for a given number of permutations.

## H.4 EXTENSION TO LATENT BOTTLENECKED ATTENTIVE NEURAL PROCESSES MODEL

To assess the generality of the proposed autoregressive buffer, we integrate it into a perceiver- style Latent Bottlenecked Attentive Neural Processes (LBANP) architecture (Jaegle et al., 2021; Feng et al., 2023). The context set is first encoded into a fixed- size latent array, and the autoregressive buffer operates over targets on top of this latent array bottleneck. We evaluate this BNP with buffer model on the GP regression task with 4 and 16 targets. As shown in Fig. A16, the LBANP equipped with our autoregressive buffer matches or slightly outperforms a standard autoregressive deployment of the Perceiver architecture (when \(K = 1\) ). This result is likely due to the fact that the buffer allows the model to explicitly represent the recent history of targets, bypassing the compressed representation of the context for immediate short- term dependencies, thus slightly enhancing predictive performance. These results demonstrate that our method extends naturally to bottlenecked / perceiver- style architectures, supporting its generality beyond full- attention models.

> **Image description.** A technical line graph consisting of two side-by-side panels, comparing the "Average joint predictive log-density" across different configurations of the Latent Bottlenecked Attentive Neural Processes (LBANP) model.
>
> The overall figure is divided into two plots, labeled "M=4" on the left and "M=16" on the right, representing different numbers of targets (M). Both panels share identical axes. The Y-axis, labeled "Joint log-density," ranges from -0.8 to 2.4. The X-axis, labeled "Number of context points N," ranges from 8 to 128, with major tick marks at 8, 16, 32, 64, and 128.
>
> A legend located below the left panel identifies three distinct methods, represented by colored lines:
> *   Blue line: LBANP w/ buffer, K=M
> *   Green line: LBANP, K=1
> *   Yellow line: LBANP Ind
>
> In the left panel (M=4), all three lines exhibit a positive trend, showing that the joint log-density increases as the number of context points (N) increases. The blue line (LBANP w/ buffer, K=M) consistently maintains the highest log-density across all values of N, followed by the green line (LBANP, K=1), with the yellow line (LBANP Ind) performing the lowest.
>
> In the right panel (M=16), which represents a higher number of targets, the same three lines are plotted. Again, all lines show a positive correlation between N and log-density. The blue line (LBANP w/ buffer, K=M) maintains a clear lead, showing the highest joint log-density across the entire range of N. The green line remains in the middle, and the yellow line remains the lowest, indicating that the buffered approach (blue line) is most effective even when the number of targets (M) is increased from 4 to 16. The lines in both panels are smooth, suggesting a continuous relationship between the number of context points and the predictive log-density.

<center>Figure A16: Autoregressive buffer extension on Latent Bottlenecked Attentive Neural Processes (LBANP) model. Average joint predictive log-density \((\uparrow)\) on GP with varying number of context \(N\) and number of targets \(M = 4, 16\) . </center>

> **Image description.** A multi-panel line graph, arranged in a 2x3 grid, illustrating the "Average joint predictive log-density" across different configurations of a Gaussian Process (GP) task. The figure compares the performance of three different models: "TNP w/ buffer, K=Targets" (blue dashed line), "TNP w/ buffer, K=1" (orange dashed line), and "TNP-AR" (green dashed line).
>
> The common axes for all six panels are:
> *   **Y-axis:** Labeled "Joint log-density," ranging from 0 to 3.
> *   **X-axis:** Labeled "Number of targets ($M$)," ranging from 8 to 64, with major tick marks at 8, 16, 32, and 64.
>
> The panels are labeled by the number of context points ($N$) in the top row and bottom row:
> *   **Top Row:** $N=4$, $N=8$, $N=16$
> *   **Bottom Row:** $N=32$, $N=64$, $N=128$
>
> **Visual Trends and Observations:**
>
> 1.  **Performance with Targets ($M$):** In all panels, the joint log-density generally increases as the number of targets ($M$) increases, indicating improved predictive performance as more targets are included.
> 2.  **Performance with Context ($N$):**
>     *   When the number of context points ($N$) is small (e.g., $N=4$ and $N=8$), the blue and orange lines (TNP variants) consistently show a higher joint log-density than the green line (TNP-AR).
>     *   As $N$ increases (moving from $N=4$ to $N=128$), the performance gap between the three models narrows significantly.
>     *   In the panels with large context sizes ($N=64$ and $N=128$), the three lines are tightly clustered and nearly overlap, suggesting that the choice of buffer size ($K$) or the specific model architecture has less impact on performance when a large amount of context is available.
> 3.  **Model Comparison:** The blue line ("TNP w/ buffer, K=Targets") and the orange line ("TNP w/ buffer, K=1") generally track very closely together, often slightly above the green line ("TNP-AR"), particularly in the panels with smaller $N$.
>
> The legend at the bottom clearly identifies the three colored dashed lines corresponding to the different model configurations being tested. The overall visual pattern demonstrates that while the TNP variants show a slight advantage over the autoregressive baseline (TNP-AR) when context is limited, all models converge to similar performance levels as the context size increases.

<center>Figure A17: Average joint predictive log-density \((\uparrow)\) on the GP task across number of targets \(M\) up to 64, with buffer size \(K\) set equal to \(M\) , for varying numbers of context points \(N\) from 4 to 128. </center>

## H.5 BUFFER SIZE ABLATION

We evaluate the effect of the buffer size \(K\) on the GP regression task, training a model with a maximum buffer size \(K = 64\) . As shown in Fig. A17, the performance of our method remains stable across this range and does not degrade relative to the autoregressive baseline, indicating that increasing \(K\) up to 64 does not harm predictive quality.

## I USE OF LARGE LANGUAGE MODELS

Idea generation and exploration. We used Large Language Models (LLMs) in the early stages of this work to support idea generation, brainstorming, and the exploration of possible methodological

directions. LLMs were also employed for tasks such as identifying related work through web search and summarization, which helped us gain an initial overview of relevant literature.

Coding assistant. LLMs provided assistance with coding, primarily by generating boilerplate components of the codebase, visualization scripts, and test codes. They were also used for drafting parts of the implementation in PyTorch. All code produced or suggested by LLMs was carefully reviewed, verified, and modified where necessary to ensure correctness and reliability.

Writing assistant. Finally, LLMs were used in preparing the manuscript, particularly for refining clarity, conciseness, and grammatical correctness. They supported rephrasing and restructuring of text, helping us to communicate ideas more effectively while maintaining the accuracy and integrity of the content.

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
