# ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition - Appendix

---

#### Page 16

# Appendix 

The appendix is organized as follows:

- In Section A, we provide detailed derivations and proofs for the theoretical claims made regarding information gain and variational bounds.
- In Section B, we present the complete training algorithm and the specifics of the Aline model.
- In Section C, we provide comprehensive details for each experimental setup, including task descriptions and baseline implementations.
- In Section D, we present additional experimental results, including further visualizations, performance on more benchmarks, and analyses of inference times.
- In Section E, we provide an overview of the computational resources and software dependencies for this work.

## A Proofs of theoretical results

## A. 1 Derivation of total EIG for $\theta_{S}$

Following Eq. 3, we can write the expression for the total expected information gain $\operatorname{sEIG}_{\theta_{S}}$ about a parameter subset $\theta_{S} \subseteq \theta$ given data $\mathcal{D}_{T}$ generated under policy $\pi_{\psi}$ as:

$$
\operatorname{sEIG}_{\theta_{S}}(\psi)=H\left[p\left(\theta_{S}\right)\right]+\underbrace{\mathbb{E}_{p\left(\theta_{S}, \mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\log p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right]}_{E_{1}}
$$

where $p\left(\theta_{S}\right)$ is the marginal prior for $\theta_{S}$, and $p\left(\theta_{S}, \mathcal{D}_{T} \mid \pi_{\psi}\right)$ is the joint distribution of $\theta_{S}$ and $\mathcal{D}_{T}$ under $\pi_{\psi}$. Now, let $\theta_{R}=\theta \backslash \theta_{S}$ be the remaining component of $\theta$ not included in $\theta_{S}$. Then, we can express $E_{1}$ from Eq. A1 as

$$
\begin{aligned}
E_{1} & =\int \log p\left(\theta_{S} \mid \mathcal{D}_{T}\right) p\left(\theta_{S}, \mathcal{D}_{T} \mid \pi_{\psi}\right) \mathrm{d} \theta_{S} \\
& =\int \log p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\left[\int p\left(\theta_{S}, \theta_{R}, \mathcal{D}_{T} \mid \pi_{\psi}\right) \mathrm{d} \theta_{R}\right] \mathrm{d} \theta_{S} \\
& =\int \log p\left(\theta_{S} \mid \mathcal{D}_{T}\right) \int p\left(\theta, \mathcal{D}_{T} \mid \pi_{\psi}\right) \mathrm{d} \theta \\
& =\mathbb{E}_{p\left(\theta, \mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\log p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right]
\end{aligned}
$$

Plugging the above expression in Eq. A1 and noting that $p\left(\theta, \mathcal{D}_{T} \mid \pi_{\psi}\right)=p(\theta) p\left(\mathcal{D}_{T} \mid \theta, \pi_{\psi}\right)$, we arrive at the expression for $\operatorname{sEIG}_{\theta_{S}}$ in Eq. 7.

## A. 2 Proof of Proposition 1

Proposition (Proposition 1). The total expected predictive information gain for a design policy $\pi_{\psi}$ over a data trajectory of length $T$ is:

$$
\begin{aligned}
\operatorname{sEPIG}(\psi) & :=\mathbb{E}_{p_{*}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]-H\left[p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]\right] \\
& =\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p_{*}\left(x^{\star}\right) p\left(y^{\star} \mid x^{\star}, \theta\right)}\left[\log p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]+\mathbb{E}_{p_{*}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right]
\end{aligned}
$$

Proof. Let $p_{*}\left(x^{\star}\right)$ be the target distribution over inputs $x^{\star}$ for which we want to improve predictive performance. Let $y^{\star}$ be the corresponding target output. The single-step EPIG for acquiring data $(x, y)$ measures the expected reduction in uncertainty (entropy) about $y^{\star}$ for a random target $x^{\star} \sim p_{*}\left(x^{\star}\right)$ :

$$
\operatorname{EPIG}(x)=\mathbb{E}_{p_{*}\left(x^{\star}\right) p(y \mid x)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]-H\left[p\left(y^{\star} \mid x^{\star}, x, y\right)\right]\right]
$$

Following Theorem 1 in [23], the total EPIG, is the total expected reduction in predictive entropy from the initial prediction $p\left(y^{\star} \mid x^{\star}\right)$ to the final prediction based on the full history $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ :

$$
\begin{aligned}
\operatorname{sEPIG}(\psi) & =\mathbb{E}_{p_{*}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]-H\left[p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]\right] \\
& =\mathbb{E}_{p_{*}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\mathbb{E}_{p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}\left[\log p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]\right]+\mathbb{E}_{p_{*}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right] \\
& =\mathbb{E}_{p_{*}\left(x^{\star}\right) p\left(\mathcal{D}_{T}, y^{\star} \mid \pi_{\psi}, x^{\star}\right)}\left[\log p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]+\mathbb{E}_{p_{*}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right]
\end{aligned}
$$

---

#### Page 17

Here, Eq. A3 follows from conditioning EPIG on the entire trajectory $\mathcal{D}_{T}$ instead of a single data point $y$, Eq. A4 follows from the definition of entropy $H[\cdot]$, and Eq. A5 follows from noting that $p\left(\mathcal{D}_{T}, y^{\star} \mid \pi_{\psi}, x^{\star}\right)=p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right) p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$. Next, we combine the expectations and express the joint distribution $p\left(\mathcal{D}_{T}, y^{\star} \mid \pi_{\psi}, x^{\star}\right)=\int p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p\left(y^{\star} \mid x^{\star}, \theta\right) d \theta$, where, following [67], we assume conditional independence between $\mathcal{D}_{T}$ and $y^{\star}$ given $\theta$. This yields:

$$
\operatorname{sEPIG}(\psi)=\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p_{\star}\left(x^{\star}\right) p\left(y^{\star} \mid x^{\star}, \theta\right)}\left[\log p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]+\mathbb{E}_{p_{\star}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right]
$$

which completes our proof.

# A. 3 Proof of Proposition 2 

Proposition (Proposition 2). Let the policy $\pi_{\psi}$ generate the trajectory $\mathcal{D}_{T}$. With $q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)$ approximating $p\left(\theta_{S} \mid \mathcal{D}_{T}\right)$, and $q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ approximating $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$, we have $\mathcal{J}_{S}^{\theta}(\psi) \leq$ $s E I G_{\theta_{S}}(\psi)$ and $\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi) \leq s E P I G(\psi)$. Moreover,

$$
\begin{aligned}
& s E I G_{\theta_{S}}(\psi)-\mathcal{J}_{S}^{\theta}(\psi)=\mathbb{E}_{p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[K L\left(p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right)\left\|q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right)\right], \quad \text { and } \\
& s E P I G(\psi)-\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi)=\mathbb{E}_{p_{\star}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[K L\left(p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\left\|q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right)\right]\right.
\end{aligned}
$$

Proof. Using the expressions for $\operatorname{sEIG}_{\theta_{S}}$ and $\mathcal{J}_{S}^{\theta}$ from Eq. 7 and Eq. 8, respectively, and noting that $\log q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)=\sum_{l \in S} \log q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{T}\right)$, we can write the expression for $\operatorname{sEIG}_{\theta_{S}}(\psi)-\mathcal{J}_{S}^{\theta}(\psi)$ as:

$$
\begin{aligned}
\operatorname{sEIG}_{\theta_{S}}(\psi)-\mathcal{J}_{S}^{\theta}(\psi) & =\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right)}\left[\log p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right]-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right)}\left[\log q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right] \\
& =\mathbb{E}_{p\left(\mathcal{D}_{T}, \theta \mid \pi_{\psi}\right)}\left[\log \frac{p\left(\theta_{S} \mid \mathcal{D}_{T}\right)}{q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)}\right] \\
& =\mathbb{E}_{p\left(\mathcal{D}_{T}, \theta_{S} \mid \pi_{\psi}\right)}\left[\log \frac{p\left(\theta_{S} \mid \mathcal{D}_{T}\right)}{q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)}\right] \\
& =\mathbb{E}_{p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\mathbb{E}_{p\left(\theta_{S} \mid \mathcal{D}_{T}\right)}\left[\log \frac{p\left(\theta_{S} \mid \mathcal{D}_{T}\right)}{q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)}\right]\right] \\
& =\mathbb{E}_{p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\operatorname{KL}\left(p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\left\|q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right)\right]\right.
\end{aligned}
$$

Here, Eq. A8 follows from the fact $p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right)=p\left(\mathcal{D}_{T}, \theta \mid \pi_{\psi}\right)$, Eq. A9 follows from Eq. A2, Eq. A10 follows from the fact that $p\left(\mathcal{D}_{T}, \theta_{S} \mid \pi_{\psi}\right)=p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right) p\left(\theta_{S} \mid \mathcal{D}_{T}\right)$, and Eq. A11 follows from the definition of KL divergence.
Since the KL divergence is always non-negative ( $\operatorname{KL}(P \| Q) \geq 0$ ), its expectation over trajectories $p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)$ must also be non-negative. Therefore:

$$
\mathcal{J}_{S}^{\theta}(\psi) \leq \operatorname{sEIG}_{\theta_{S}}(\psi)
$$

Now, we consider the difference between $\operatorname{sEPIG}(\psi)$ and $\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi)$ :

$$
\begin{aligned}
\operatorname{sEPIG}(\psi)-\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi) & =\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p_{\star}\left(x^{\star}\right) p\left(y^{\star} \mid x^{\star}, \theta\right)}\left[\log \frac{p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}{q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}\right] \\
& =\mathbb{E}_{p_{\star}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\mathbb{E}_{p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}\left[\log \frac{p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}{q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)}\right]\right]
\end{aligned}
$$

Similar to the previous case, the inner expectation is the definition of the KL divergence between the true posterior predictive $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ and the variational approximation $q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ :

$$
\operatorname{sEPIG}(\psi)-\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi)=\mathbb{E}_{p_{\star}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[\operatorname{KL}\left(p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\left\|q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right)\right]\right.
$$

Since the KL divergence is always non-negative, therefore:

$$
\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi) \leq \operatorname{sEPIG}(\psi)
$$

which completes the proof.

---

#### Page 18

# B Further details on Aline 

## B. 1 Training algorithm

```
Algorithm 1 Aline Training Procedure
    Input: Prior \(p(\theta)\), likelihood \(p(y \mid x, \theta)\), target distribution \(p(\xi)\), query horizon \(T\), total training
    episodes \(E_{\text {max }}\), warm-up episodes \(E_{\text {warm }}\).
    Output: Trained Aline model \(\left(q_{\phi}, \pi_{\psi}\right)\).
    for epoch \(=1\) to \(E_{\max }\) do
        Sample parameters \(\theta \sim p(\theta)\).
        Sample target specifier set \(\xi \sim p(\xi)\) and corresponding targets \(\theta_{S}\) or \(\left\{x_{m}^{\star}, y_{m}^{\star}\right\}_{m=1}^{M}\).
        Initialize candidate query set \(\mathcal{Q}\).
        for \(t=1\) to \(T\) do
            if epoch \(\leq E_{\text {warm }}\) then
                Select next query \(x_{t}\) uniformly at random from \(\mathcal{Q}\).
            else
                Select next query \(x_{t} \sim \pi_{\psi}\left(\cdot \mid \mathcal{D}_{t-1}, \xi\right)\) from \(\mathcal{Q}\).
            end if
            Sample outcome \(y_{t} \sim p\left(y \mid x_{t}, \theta\right)\).
            Update history \(\mathcal{D}_{t} \leftarrow \mathcal{D}_{t-1} \cup\left\{\left(x_{t}, y_{t}\right)\right\}\).
            Update query set \(\mathcal{Q} \leftarrow \mathcal{Q} \backslash\left\{x_{t}\right\}\).
            if epoch \(\leq E_{\text {warm }}\) then
                \(\mathcal{L}=\mathcal{L}_{\mathrm{NLL}}\) (Eq. 12)
            else
                Calculate reward \(R_{t}\) (Eq. 10)
                \(\mathcal{L}=\mathcal{L}_{\mathrm{NLL}}\) (Eq. 12) \(+\mathcal{L}_{\mathrm{PG}}\) (Eq. 11)
            end if
            Update Aline using \(\mathcal{L}\).
        end for
    end for
```

## B. 2 Architecture and training details

In Aline, the data is first processed by different embedding layers. Inputs (context $x_{i}$, query candidates $x_{o}^{\mathrm{q}}$, target locations $x_{k}^{\star}$ ) are passed through a shared nonlinear embedder $f_{x}$. Observed outcomes $y_{i}$ are embedded using a separate embedder $f_{y}$. For discrete parameters, we assign a unique indicator $\ell_{l}$ to each parameter $\theta_{l}$, which is then associated with a unique, learnable embedding vector, denoted as $f_{\theta}\left(\ell_{l}\right)$. We compute the final context embedding by summing the outputs of the respective embedders: $E^{\mathcal{D}_{t}}=\left\{\left(f_{x}\left(x_{i}\right)+f_{y}\left(y_{i}\right)\right)\right\}_{i=1}^{t}$. Query and target sets are embedded as $E^{\mathcal{Q}}=\left\{\left(f_{x}\left(x_{o}^{\mathrm{q}}\right)\right)\right\}_{n=1}^{N}$ and $E^{T}$ (either $\left\{\left(f_{x}\left(x_{m}^{\star}\right)\right)\right\}_{m=1}^{M}$ or $\left\{f_{\theta}\left(\ell_{l}\right)\right\}_{t \in S}$ ). Both $f_{x}$ and $f_{y}$ are MLPs consisting of an initial linear layer, followed by a ReLU activation function, and a final linear layer. For all our experiments, the embedders use a feedforward dimension of 128 and project inputs to an embedding dimension of 32 .
The core of our architecture is a transformer network. We employ a configuration with 3 transformer layers, each equipped with 4 attention heads. The feedforward networks within each transformer layer have a dimension of 128. The model's internal embedding dimension, consistent across the transformer layers and the output of the initial embedding layers, is 32 . These transformer layers process the embedded representations of the context, query, and target sets. The interactions between these sets are governed by specific attention masks, visually detailed in Figure A1, where a shaded element indicates that the token corresponding to its row is permitted to attend to the token corresponding to its column.
Aline has two specialized output heads. The inference head, responsible for approximating posteriors and posterior predictive distributions, parameterizes a Gaussian Mixture Model (GMM) with 10 components. The embeddings corresponding to the inference targets are processed by 10 separate MLPs, one for each GMM component. Each MLP outputs parameters for its component: a mixture weight, a mean, and a standard deviation. The standard deviations are passed through a Softplus

---

#### Page 19

> **Image description.** This image displays two distinct panels, labeled (a) and (b), side-by-side. Both panels illustrate an attention mask using a grid of small, rounded-corner squares. Shaded squares are filled with a light purple color, while unshaded squares are white, both outlined in thin black lines. The labels for rows are positioned to the left of the grid, and labels for columns are positioned below the grid, rotated counter-clockwise.
> 
> **Panel (a):**
> This panel presents a 9x9 grid of squares, conceptually representing a larger matrix where '...' indicates omitted intermediate elements.
> The row labels, from top to bottom, are:
> `(x_1, y_1)`
> `:`
> `(x_t, y_t)`
> `x_1^q`
> `:`
> `x_N^q`
> `x_1^*`
> `:`
> `x_M^*`
> 
> The column labels, from left to right, are:
> `(x_1, y_1)`
> `...`
> `(x_t, y_t)`
> `x_1^q`
> `...`
> `x_N^q`
> `x_1^*`
> `...`
> `x_M^*`
> 
> The shading pattern indicates allowed attention:
> *   **Rows `(x_1, y_1)` through `(x_t, y_t)` (first 3 visible rows):** All squares in these rows are shaded purple, indicating full attention to all columns.
> *   **Rows `x_1^q` through `x_N^q` (middle 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^q` through `x_N^q`. The squares corresponding to `x_1^*` through `x_M^*` columns are unshaded white.
> *   **Rows `x_1^*` through `x_M^*` (last 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^*` through `x_M^*`. The squares corresponding to `x_1^q` through `x_N^q` columns are unshaded white.
> 
> **Panel (b):**
> This panel presents a 7x7 grid of squares, similar to panel (a) with '...' indicating omitted elements.
> The row labels, from top to bottom, are:
> `(x_1, y_1)`
> `:`
> `(x_t, y_t)`
> `x_1^q`
> `:`
> `x_N^q`
> `l_1`
> 
> The column labels, from left to right, are:
> `(x_1, y_1)`
> `...`
> `(x_t, y_t)`
> `x_1^q`
> `...`
> `x_N^q`
> `l_1`
> 
> The shading pattern indicates allowed attention:
> *   **Rows `(x_1, y_1)` through `(x_t, y_t)` (first 3 visible rows):** All squares in these rows are shaded purple, indicating full attention to all columns.
> *   **Rows `x_1^q` through `x_N^q` (middle 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^q` through `x_N^q`. The square corresponding to the `l_1` column is unshaded white.
> *   **Row `l_1` (last visible row):** Squares in this row are shaded purple only for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)`. The squares corresponding to `x_1^q` through `x_N^q` and `l_1` columns are unshaded white.

Figure A1: Example attention masks in Aline's transformer architecture. (a) Mask for a predictive target $\xi=\xi_{p_{*}}^{y^{*}}$ (b) Mask for a parameter target $\xi=\xi_{(1)}^{\theta}$. Shaded squares indicate allowed attention.

activation function to ensure positivity, and the mixture weights are normalized using a Softmax function. The policy head, which generates a probability distribution over the candidate query points, is a 2-layer MLP with a feedforward dimension of 128. Its output is passed through a Softmax function to ensure that the probabilities of all actions sum to unity. The architecture of Aline is shown in Figure 2.

ALINE is trained using the AdamW optimizer with a weight decay of 0.01 . The initial learning rate is set to 0.001 and decays according to a cosine annealing schedule.

# C Experimental details 

This section provides details for the experimental setups. Section C. 1 outlines the specifics for the active learning experiments in Section 4.1, including the synthetic function sampling procedures (Section C.1.1), implementation details for baseline methods (Section C.1.2), and training and evaluation details for these tasks (Section C.1.3). Next, in Section C. 2 we describe the details of BED tasks, including the task descriptions (Section C.2.1), implementation of the baselines (Section C.2.2), and the training and evaluation details (Section C.2.3). Lastly, Section C. 3 contains the specifics of the psychometric modeling experiments, detailing the psychometric function we use (Section C.3.1) and the setup for the experimental comparisons (Section C.3.2).

## C. 1 Active learning for regression and hyperparameter inference

## C.1.1 Synthetic functions sampling procedure

For active learning tasks, Aline is trained exclusively on synthetically generated Gaussian Process (GP) functions. The procedure for generating these functions is as follows. First, the hyperparameters of the GP kernels, namely the output scale and lengthscale(s), are sampled from their respective prior distributions. For multi-dimensional input spaces ( $d_{x}>1$ ), there is a $p_{\text {iso }}=0.5$ probability that an isotropic kernel is used, meaning that all input dimensions share a common lengthscale. Otherwise, an anisotropic kernel is employed, with a distinct lengthscale sampled for each input dimension. Subsequently, a kernel function is chosen randomly from a pre-defined set, with each kernel having a uniform probability of selection. In our experiments, we utilize the Radial Basis Function (RBF), Matérn 3/2, and Matérn 5/2 kernels.
The kernel's output scale is sampled uniformly from the interval $U(0.1,1)$. The lengthscale(s) are sampled from $U(0.1,2) \times \sqrt{d_{x}}$. Input data points $x$ are sampled uniformly within the range $[-5,5]$ for each dimension. Finally, Gaussian noise with a fixed standard deviation of 0.01 is added to

---

#### Page 20

> **Image description.** The image displays a grid of 20 individual line graphs, arranged in 5 rows and 4 columns. Each graph presents a distinct blue curve plotted against a white background within its own coordinate system. These graphs are examples of 1D synthetic functions.
> 
> Each of the 20 subplots shares a consistent visual structure for its axes:
> *   **X-axis:** The horizontal axis ranges from -5.0 to 5.0, with major tick marks and numerical labels at -5.0, -2.5, 0.0, 2.5, and 5.0.
> *   **Y-axis:** The vertical axis ranges from -2 to 2, with major tick marks and numerical labels at -2, 0, and 2.
> *   **Plot Line:** A single, solid blue line represents the function in each subplot.
> 
> Above each subplot, a unique title is provided, following the format "Sample X: ls = Y, scale = Z". The values for 'ls' and 'scale' vary across the samples, visually correlating with the characteristics of the plotted functions. Generally, lower 'ls' values correspond to more erratic and jagged functions with frequent oscillations, while higher 'ls' values indicate smoother functions with broader curves or gentler slopes. Similarly, higher 'scale' values tend to correspond to functions with larger amplitudes, reaching closer to the y-axis limits, whereas lower 'scale' values indicate functions with smaller amplitudes, staying closer to the x-axis.
> 
> The specific titles and visual characteristics of each sample are as follows:
> 
> **Row 1:**
> *   **Sample 1:** "Sample 1: ls = 1.20, scale = 0.33". A relatively smooth curve with gentle undulations, mostly staying between -1 and 1.
> *   **Sample 2:** "Sample 2: ls = 1.51, scale = 0.29". A very smooth, gently sloping curve, starting slightly above 0 and gradually decreasing, then slightly increasing, remaining close to the x-axis.
> *   **Sample 3:** "Sample 3: ls = 1.92, scale = 0.69". A smooth, wave-like curve with one prominent peak and trough, oscillating between approximately -1.5 and 1.5.
> *   **Sample 4:** "Sample 4: ls = 0.38, scale = 0.81". A highly erratic and jagged function with numerous rapid fluctuations, covering nearly the full y-range from -2 to 2.
> 
> **Row 2:**
> *   **Sample 5:** "Sample 5: ls = 0.34, scale = 0.18". An erratic function with small amplitude oscillations, mostly staying between -0.5 and 0.5.
> *   **Sample 6:** "Sample 6: ls = 1.01, scale = 0.68". A smooth, S-shaped curve with a significant dip below -1 and a rise above 1.
> *   **Sample 7:** "Sample 7: ls = 0.86, scale = 0.99". A complex, moderately erratic function with several distinct peaks and troughs, reaching close to the y-axis limits.
> *   **Sample 8:** "Sample 8: ls = 0.65, scale = 0.44". A moderately smooth, wavy function with several oscillations, mostly between -1 and 1.
> 
> **Row 3:**
> *   **Sample 9:** "Sample 9: ls = 0.78, scale = 0.74". A smooth, multi-peak and trough curve, resembling a sine wave with varying amplitudes, mostly between -1.5 and 1.5.
> *   **Sample 10:** "Sample 10: ls = 1.69, scale = 0.70". A relatively smooth function with gentle, broad undulations, mostly between -1 and 1.
> *   **Sample 11:** "Sample 11: ls = 1.95, scale = 0.53". A very smooth, almost flat line that gently curves downwards from left to right, staying close to the x-axis.
> *   **Sample 12:** "Sample 12: ls = 0.51, scale = 0.89". A highly periodic and smooth wave-like pattern with multiple prominent peaks and troughs, resembling a sine wave with large amplitude, reaching close to the y-axis limits.
> 
> **Row 4:**
> *   **Sample 13:** "Sample 13: ls = 0.61, scale = 0.34". A moderately erratic function with several small oscillations, mostly between -1 and 1.
> *   **Sample 14:** "Sample 14: ls = 0.54, scale = 0.51". A moderately erratic function with more pronounced peaks and troughs than Sample 13, covering a wider y-range.
> *   **Sample 15:** "Sample 15: ls = 0.88, scale = 0.41". A relatively smooth, gently undulating function, mostly staying between -1 and 1.
> *   **Sample 16:** "Sample 16: ls = 0.24, scale = 0.46". A highly erratic and jagged function with frequent, rapid fluctuations, similar to Sample 4 but with slightly smaller amplitude.
> 
> **Row 5:**
> *   **Sample 17:** "Sample 17: ls = 1.02, scale = 0.17". A very smooth, almost flat line with minimal undulations, staying very close to the x-axis, mostly between -0.5 and 0.5.
> *   **Sample 18:** "Sample 18: ls = 1.19, scale = 0.54". A smooth, wave-like curve with a prominent trough and peak, resembling a single cycle of a wave, mostly between -1.5 and 1.
> *   **Sample 19:** "Sample 19: ls = 0.16, scale = 0.72". A highly erratic and jagged function with very frequent and large fluctuations, covering the full y-range from -2 to 2, appearing to be the most erratic among all samples.
> *   **Sample 20:** "Sample 20: ls = 1.62, scale = 0.66". A relatively smooth curve, starting slightly above 0, dipping below 0, and then rising again, resembling a gentle 'U' shape.

Figure A2: Examples of randomly sampled 1D synthetic GP functions used to train Aline.

the true function output $y$ for each sampled data point. Figure A2 illustrates some examples of the synthetic GP functions generated using this procedure.

# C.1.2 Details of acquisition functions 

We compare Aline with four commonly used AL acquisition functions. For Random Sampling (RS), we randomly select one point from the candidate pool as the next query point.
Uncertainty Sampling (US) is a simple and widely used AL acquisition strategy that prioritizes points where the model is most uncertain about its prediction:

$$
\operatorname{US}(x)=\sqrt{\mathbb{V}[y \mid x, \mathcal{D}]}
$$

where $\mathbb{V}[y \mid x, \mathcal{D}]$ is the predictive variance at $x$ given the current training data $\mathcal{D}$.
Variance Reduction (VR) [74] aims to select a candidate point that is expected to maximally reduce the predictive variance over a pre-defined test set $\left\{x_{m}^{*}\right\}_{m=1}^{M}$, which is defined as:

$$
\operatorname{VR}(x)=\frac{\sum_{m=1}^{M}\left(\operatorname{Cov}_{\text {post }}\left(x^{\star}, x\right)\right)^{2}}{\mathbb{V}[y \mid x, \mathcal{D}]}
$$

$\operatorname{Cov}_{\text {post }}\left(x^{\star}, x\right)$ is the posterior covariance between the latent function values at $x^{\star}$ and $x$, given the history $\mathcal{D}=\left\{\left(X_{\text {train }}, y_{\text {train }}\right)\right\}$, where $X_{\text {train }}$ comprises all currently observed inputs with $y_{\text {train }}$ being their corresponding outputs. It is computed as:

$$
\operatorname{Cov}_{\text {post }}\left(x^{\star}, x\right)=k\left(x^{\star}, x\right)-k\left(x^{\star}, X_{\text {train }}\right)\left(K_{\text {train }}+\alpha I\right)^{-1} k\left(X_{\text {train }}, x\right)
$$

Here, $k(\cdot, \cdot)$ is the GP kernel function, $K_{\text {train }}=k\left(X_{\text {train }}, X_{\text {train }}\right)$, and $\alpha$ is the noise variance.
Expected Predictive Information Gain (EPIG) [67] measures the expected reduction in predictive uncertainty on a target input distribution $p_{\star}\left(x^{\star}\right)$. Following Smith et al. [67], for a Gaussian predictive distribution, the EPIG for a candidate point can be expressed as:

$$
\operatorname{EPIG}(x)=\mathbb{E}_{p_{\star}\left(x^{\star}\right)}\left[\frac{1}{2} \log \frac{\mathbb{V}[y \mid x, \mathcal{D}] \mathbb{V}\left[y^{\star} \mid x^{\star}, \mathcal{D}\right]}{\mathbb{V}[y \mid x, \mathcal{D}] \mathbb{V}\left[y^{\star} \mid x^{\star}, \mathcal{D}\right]-\operatorname{Cov}_{\text {post }}\left(x^{\star}, x\right)^{2}}\right]
$$

---

#### Page 21

In practice, we approximate it by averaging over $m$ sampled test points:

$$
\operatorname{EPIG}(x) \approx \frac{1}{2 M} \sum_{m=1}^{M} \log \frac{\mathbb{V}[y \mid x, \mathcal{D}] \mathbb{V}\left[y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}\right]}{\mathbb{V}[y \mid x, \mathcal{D}] \mathbb{V}\left[y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}\right]-\operatorname{Cov}_{\text {post }}\left(x_{m}^{\star}, x\right)^{2}}
$$

# C.1.3 Training and evaluation details 

For both 1D and 2D input scenarios, Aline is trained for $2 \cdot 10^{5}$ epochs using a batch size of 200. The discount factor $\gamma$ for the policy gradient loss is set to 1 . For the GP-based baselines, we utilized Gaussian Process Regressors implemented via the scikit-learn library [56]. The hyperparameters of the GP models are optimized at each step. For the ACE baseline [15], we use a transformer architecture and an inference head design consistent with our Aline model.
All active learning experiments are evaluated with a candidate query pool consisting of 500 points. Each experimental run commenced with an initial context set consisting of a single data point. The target set size for predictive tasks is set to 100 .

## C. 2 Benchmarking on Bayesian experimental design tasks

## C.2.1 Task descriptions

Location Finding [66] is a benchmark problem commonly used in BED literature [24, 40, 9, 41]. The objective is to infer the unknown positions of $K$ hidden sources, $\theta=\left\{\theta_{k} \in \mathbb{R}^{d}\right\}_{k=1}^{K}$, by strategically selecting a sequence of observation locations, $x \in \mathbb{R}^{d}$. Each source emits a signal whose intensity attenuates with distance following an inverse-square law. The total signal intensity at an observation location $x$ is given by the superposition of signals from all sources:

$$
\mu(\theta, x)=b+\sum_{k=1}^{K} \frac{\alpha_{k}}{m+\left\|\theta_{k}-x\right\|^{2}}
$$

where $\alpha_{k}$ are known source strength constants, and $b, m>0$ are constants controlling the background level and maximum signal intensity, respectively. In this experiment, we use $K=1, d=2, \alpha_{k}=1$, $b=0.1$ and $m=10^{-4}$, and the prior distribution over each component of a source's location $\theta_{k}=\left(\theta_{k, 1}, \ldots, \theta_{k, d}\right)$ is uniform over the interval $[0,1]$.
The observation is modeled as the log-transformed total intensity corrupted by Gaussian noise:

$$
\log y \mid \theta, x \sim \mathcal{N}\left(\log \mu(\theta, x), \sigma^{2}\right)
$$

where we use $\sigma=0.5$ in our experiments.
Constant Elasticity of Substitution (CES) [3] considers a behavioral economics problem in which a participant compares two baskets of goods and rates the subjective difference in utility between the baskets on a sliding scale from 0 to 1 . The utility of a basket $z$, consisting of $K$ goods with different values, is characterized by latent parameters $\theta=(\rho, \boldsymbol{\alpha}, u)$. The design problem is to select pairs of baskets, $x=\left(z, z^{\prime}\right) \in[0,100]^{2 K}$, to infer the participant's latent utility parameters.
The utility of a basket $z$ is defined using the constant elasticity of substitution function, as:

$$
U(z)=\left(\sum_{i=1}^{K} z_{i}^{\rho} \alpha_{i}\right)^{\frac{1}{\rho}}
$$

The prior of the latent parameters is specified as:

$$
\begin{aligned}
\rho & \sim \operatorname{Beta}(1,1) \\
\boldsymbol{\alpha} & \sim \operatorname{Dirichlet}\left(\mathbf{1}_{K}\right) \\
\log u & \sim \mathcal{N}\left(1,3^{2}\right)
\end{aligned}
$$

The subjective utility difference between two baskets is modeled as follows:

$$
\begin{aligned}
& \eta \sim \mathcal{N}\left(u \cdot\left(U(z)-U\left(z^{\prime}\right)\right), u^{2} \cdot \tau^{2} \cdot\left(1+\left\|z-z^{\prime}\right\|\right)^{2}\right) \\
& y=\operatorname{clip}(\operatorname{sigmoid}(\eta), \epsilon, 1-\epsilon)
\end{aligned}
$$

In this experiment, we choose $K=3, \tau=0.005$ and $\epsilon=2^{-22}$.

---

#### Page 22

# C.2.2 Implementation details of baselines 

We compare Aline with four baseline methods. For Random Design policy, we randomly sample a design from the design space using a uniform distribution.
VPCE [25] iteratively infers the posterior through variational inference and maximizes the myopic Prior Contrastive Estimation (PCE) lower bound by gradient descent with respect to the experimental design. The hyperparameters used in the experiments are given in Table A1.

Table A1: Additional hyperparameters used in VPCE [25].

| Parameter | Location Finding | CES |
| :-- | :--: | --: |
| VI gradient steps | 1000 | 1000 |
| VI learning rate | $10^{-3}$ | $10^{-3}$ |
| Design gradient steps | 2500 | 2500 |
| Design learning rate | $10^{-3}$ | $10^{-3}$ |
| Contrastive samples $L$ | 500 | 10 |
| Expectation samples | 500 | 10 |

Deep Adaptive Design (DAD) [23] learns an amortized design policy guided by sPCE lower bound. For a design policy $\pi$, and $L \geq 0$ contrastive samples, sPCE over a sequence of $T$ experiments is defined as:

$$
\mathcal{L}_{T}(\pi, L)=\mathbb{E}_{p\left(\theta_{0}, \mathcal{D}_{T} \mid \pi\right) p\left(\theta_{1: L}\right)}\left[\log \frac{p\left(\mathcal{D}_{T} \mid \theta_{0}, \pi\right)}{\frac{1}{L+1} \sum_{\ell=0}^{L} p\left(\mathcal{D}_{T} \mid \theta_{\ell}, \pi\right)}\right]
$$

where the contrastive samples $\theta_{1: L}$ are drawn independently from the prior $p(\theta)$. The bound becomes tight as $L \rightarrow \infty$, with a convergence rate of $\mathcal{O}\left(L^{-1}\right)$.
The design network comprises an MLP encoder that encodes historical data into a fixed-dimensional representation, and an MLP emitter that proposes the next design point. The encoder processes the concatenated design-observation pairs from history and aggregates their representations through a pooling operation.
Following the work of Foster et al. [23], the encoder network consists of two fully connected layers with 128 and 16 units with ReLU activation applied to the hidden layer. The emitter is implemented as a fully connected layer that maps the pooled representation to the design space. The policy is trained using the Adam optimizer with an initial learning rate of $5 \cdot 10^{-5}, \beta=(0.8,0.998)$, and an exponentially decaying learning rate, reduced by a factor of $\gamma=0.98$ every 1000 epochs. In the Location Finding task, the model is trained for $10^{5}$ epochs, and $10^{4}$ contrastive samples are utilized in each training step for the estimation of the sPCE lower bound. Note that, for the CES task, we applied several adjustments, including normalizing the input, applying the Sigmoid and Softplus transformations to the output before mapping it to the design space, increasing the depth of the network, and initializing weights using Xavier initialization [32]. However, DAD failed to converge during training in our experiments. Therefore, we report the results provided by Blau et al. [9].
vsOED [65] is an amortized BED method that employs an actor-critic reinforcement learning framework. It utilizes separate networks for the actor (policy), the critic (value function), and the variational posterior approximation. The reward signal for training the policy is the incremental improvement in the log-probability of the ground-truth parameters under the variational posterior, which is estimated at each step of the experiment. Following the original implementation, a distinct posterior network is trained for each design stage, while the actor and critic share a common backbone. For our implementation, the hidden layers of all networks are 3-layer MLPs with 256 units and ReLU activations. The posterior network outputs the parameters for an 8-component Gaussian Mixture Model (GMM). The input to the actor and critic networks is the zero-padded history of design-observation pairs, concatenated with a one-hot encoding of the current time step. We train for $10^{4}$ epochs with a batch size of $10^{4}$ and a $10^{6}$-sized replay buffer. The learning rate starts at $10^{-3}$ with a 0.9999 exponential decay per epoch, and the discount factor is 0.9 . To encourage exploration for the deterministic policy, Gaussian noise is added during training; the initial noise scale is 0.5

---

#### Page 23

for the Location Finding task and 5.0 for the CES task, with decay rates of 0.9999 and 0.9998 , respectively.
RL-BOED [9] frames the design policy optimization as a Markov Decision Process (MDP) and employs reinforcement learning to learn the design policy. It utilizes a stepwise reward function to estimate the marginal contribution of the $t$-th experiment to the sEIG.
The design network shares a similar backbone architecture to that of DAD, with the exception that the deterministic output of the emitter is replaced by a Tanh-Gaussian distribution. The encoder comprises two fully connected layers with 128 units and ReLU activation, followed by an output layer with 64 units and no activation. Training is conducted using Randomized Ensembled Double Q-learning (REDQ) [16], the full configurations are reported in Table A2.

Table A2: Additional hyperparameters used in RL-BOED [9].

| Parameter | Location Finding | CES |
| :-- | --: | --: |
| Critics | 2 | 2 |
| Random subsets | 2 | 2 |
| Contrastive samples $L$ | $10^{5}$ | $10^{5}$ |
| Training epochs | $10^{5}$ | $5 \cdot 10^{4}$ |
| Discount factor $\gamma$ | 0.9 | 0.9 |
| Target update rate | $10^{-3}$ | $5 \cdot 10^{-3}$ |
| Policy learning rate | $10^{-4}$ | $3 \cdot 10^{-4}$ |
| Critic learning rate | $3 \cdot 10^{-4}$ | $3 \cdot 10^{-4}$ |
| Buffer size | $10^{7}$ | $10^{6}$ |

# C.2.3 Training and evaluation details 

In the Location Finding task, the number of sequential design steps, $T$, is set to 30 . For all evaluated methods, the sPCE lower bound is estimated using $L=10^{6}$ contrastive samples. The Aline is trained over $10^{5}$ epochs with a batch size of 200. The discount factor $\gamma$ for the policy gradient loss is set to 1 . During the evaluation phase, the query set consists of 2000 points, which are drawn uniformly from the defined design space. For the CES task, each experimental run consists of $T=10$ design steps. The sPCE lower bound is estimated using $L=10^{7}$ contrastive samples. The Aline is trained for $2 \cdot 10^{5}$ epochs with a batch size of 200, and we use 2000 for the query set size.

## C. 3 Psychometric model

## C.3.1 Model description

In this experiment, we use a four-parameter psychometric function with the following parameterization:

$$
\pi(x)=\theta_{3} \cdot \theta_{4}+\left(1-\theta_{4}\right) F\left(\frac{x-\theta_{1}}{\theta_{2}}\right)
$$

where:

- $\theta_{1}$ (threshold): The stimulus intensity at which the probability of a positive response reaches a specific criterion. It represents the location of the psychometric curve. We use a uniform prior $U[-3,3]$ for $\theta_{1}$.
- $\theta_{2}$ (slope): Describes the steepness of the psychometric function. Smaller values of $\theta_{2}$ indicate a sharper transition, reflecting higher sensitivity around the threshold. We use a uniform prior $U[0.1,2]$ for $\theta_{2}$.
- $\theta_{3}$ (guess rate): The baseline response probability for stimuli far below the threshold, reflecting responses made by guessing. We use a uniform prior $U[0.1,0.9]$ for $\theta_{3}$.
- $\theta_{4}$ (lapse rate): The rate at which the observer makes errors independent of stimulus intensity, representing an upper asymptote on performance below 1. We use a uniform prior $U[0,0.5]$ for $\theta_{4}$.

---

#### Page 24

We employ a Gumbel-type internal link function $F=1-e^{-10^{z}}$ where $z=\frac{x-\theta_{t}}{\theta_{d}}$. Lastly, a binary response $y$ is simulated from the psychometric function $\pi(x)$ using a Bernoulli distribution with probability of success $p=\pi(x)$.

# C.3.2 Experimental details 

We compare Aline against two established Bayesian adaptive methods:

- QUEST+ [70]: QUEST+ is an adaptive psychometric procedure that aims to find the stimulus that maximizes the expected information gain about the parameters of the psychometric function, or equivalently, minimizes the expected entropy of the posterior distribution over the parameters. It typically operates on a discrete grid of possible parameter values and selects stimuli to reduce uncertainty over this entire joint parameter space. In our experiments, QUEST+ is configured to infer all four parameters simultaneously.
- Psi-marginal [58]: The Psi-marginal method is an extension of the psi method [43] that allows for efficient inference by marginalizing over nuisance parameters. When specific parameters are designated as targets of interest, Psi-marginal optimizes stimulus selection to maximize information gain specifically for these target parameters, effectively treating the others as nuisance variables. This makes it highly efficient when only a subset of parameters is critical.

For each simulated experiment, true underlying parameters are sampled from their prior distributions. Stimulus values $x$ are selected from a discrete set of size 200 drawn uniformly from the range $[-5,5]$.

## D Additional experimental results

## D. 1 Active Exploration of High-Dimensional Hyperparameter Landscapes

To demonstrate Aline's utility on complex, high-dimensional tasks, we conduct a new set of experiments on actively exploring hyperparameter performance landscapes. This experiment aims to efficiently characterize a machine learning model's overall behavior on a new task, allowing practitioners to quickly assess a model family's viability or understand its sensitivities. The task is to actively query a small number of hyperparameter configurations to build a surrogate model that accurately predicts performance for a larger, held-out set of target configurations. We use high-dimensional, real-world tasks from the HPO-B benchmark [2], evaluating on rpart (6D), svm (8D), ranger (9D), and xgboost (16D) datasets. Aline is trained on their multiple pre-defined training sets. We then evaluate its performance, alongside non-amortized GP-based baselines and an amortized surrogate baseline (ACE-US), on the benchmark's held-out and entirely unseen test sets.
Table A3 shows the RMSE results after 30 steps, averaged across all test tasks for each dataset. First, both amortized methods, Aline and ACE-US, significantly outperform all non-amortized GP-based baselines across all tasks. This highlights the power of meta-learning in this domain. GP-based methods must learn each new performance landscape from scratch, which is highly inefficient in high dimensions. In contrast, both Aline and ACE-US are pre-trained on hundreds of related tasks, and their Transformer architectures meta-learn the structural patterns common to these landscapes. This shared prior knowledge allows them to make far more accurate predictions from sparse data. Second, while ACE-US performs strongly due to its amortized nature, Aline consistently achieves the best or joint-best performance. This demonstrates the additional, crucial benefit of our core contribution: the learned acquisition policy. ACE-US relies on a standard heuristic, whereas Aline's policy is trained end-to-end to learn how to optimally explore the landscape, leading to more informative queries and ultimately a more accurate final surrogate model.

## D. 2 Active learning for regression and hyperparameter inference

AL results on more benchmark functions. To further assess Aline, we present performance evaluations on an additional set of active learning benchmark functions, see Figure A3. The results on Gramacy and Branin show that we are on par with the GP baselines. For the Three Hump Camel, we see both Aline and ACE-US showing reduced accuracy. This is because the function's output value range extends beyond that of the GP functions used during pre-training. This highlights a potential

---

#### Page 25

Table A3: RMSE ( $\downarrow$ ) on the HPO-B benchmark after 30 active queries. Results show the mean and $95 \%$ CI over all test tasks for each dataset.

|  | GP-RS | GP-US | GP-VR | GP-EPIG | ACE-US | Aline (ours) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| rpart (6D) | $0.07 \pm 0.03$ | $0.04 \pm 0.02$ | $0.04 \pm 0.02$ | $0.05 \pm 0.02$ | $\mathbf{0 . 0 1} \pm 0.00$ | $\mathbf{0 . 0 1} \pm 0.00$ |
| svm (8D) | $0.22 \pm 0.11$ | $0.11 \pm 0.05$ | $0.12 \pm 0.07$ | $0.15 \pm 0.08$ | $0.04 \pm 0.01$ | $\mathbf{0 . 0 3} \pm 0.01$ |
| ranger (9D) | $0.10 \pm 0.02$ | $0.07 \pm 0.01$ | $0.08 \pm 0.02$ | $0.08 \pm 0.02$ | $\mathbf{0 . 0 2} \pm 0.01$ | $\mathbf{0 . 0 2} \pm 0.01$ |
| xgboost (16D) | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ | $0.04 \pm 0.01$ | $\mathbf{0 . 0 3} \pm 0.01$ |

> **Image description.** The image presents a multi-panel line graph consisting of three subplots arranged horizontally, each illustrating predictive performance in terms of RMSE (Root Mean Squared Error) across different active learning benchmark functions. Each subplot shows multiple colored lines, representing different methods, with translucent shaded areas indicating confidence intervals. A common legend is placed below the three panels.
> 
> Each of the three panels shares a common y-axis labeled "RMSE ↓", with the downward arrow suggesting that lower values are better. The x-axis in each panel represents an increasing numerical value, likely corresponding to the number of queries or iterations, though not explicitly labeled.
> 
> The panels are titled as follows:
> 1.  **Gramacy 1D (Left Panel):**
>     *   The y-axis ranges from 0.0 to 0.5.
>     *   The x-axis ranges from 0 to 30, with major ticks at 5-unit intervals.
>     *   Six distinct lines are plotted:
>         *   **GP-RS (grey circles):** Starts at approximately 0.5 and gradually decreases, leveling off around 0.3 with a relatively wide shaded confidence interval.
>         *   **GP-US (brown squares):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.15 with a narrow shaded confidence interval.
>         *   **GP-VR (blue upward triangles):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.1 with a narrow shaded confidence interval.
>         *   **GP-EPIG (purple diamonds):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.1 with a narrow shaded confidence interval.
>         *   **ACE-US (green crosses):** Starts at approximately 0.5 and decreases, leveling off around 0.1 with a narrow shaded confidence interval.
>         *   **ALINE (ours) (orange stars):** Starts at approximately 0.5 and shows the steepest initial decrease, reaching the lowest RMSE value of approximately 0.05 by the end of the x-axis range, with a narrow shaded confidence interval.
> 
> 2.  **Branin 2D (Middle Panel):**
>     *   The y-axis ranges from 0.0 to 0.9.
>     *   The x-axis ranges from 0 to 50, with major ticks at 10-unit intervals.
>     *   Similar to the first panel, six lines are plotted, all showing a decreasing trend:
>         *   **GP-RS (grey circles):** Starts at approximately 0.9 and decreases, leveling off around 0.2 with a relatively wide shaded confidence interval.
>         *   **GP-US (brown squares):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>         *   **GP-VR (blue upward triangles):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>         *   **GP-EPIG (purple diamonds):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>         *   **ACE-US (green crosses):** Starts at approximately 0.9 and decreases, leveling off around 0.1 with a narrow shaded confidence interval.
>         *   **ALINE (ours) (orange stars):** Starts at approximately 0.9 and shows the steepest initial decrease, reaching the lowest RMSE value of approximately 0.03 by the end of the x-axis range, with a narrow shaded confidence interval.
> 
> 3.  **Three Hump Camel 2D (Right Panel):**
>     *   The y-axis ranges from 0.0 to 2.5.
>     *   The x-axis ranges from 0 to 50, with major ticks at 10-unit intervals.
>     *   The six lines are plotted, generally showing a decreasing trend, but with higher RMSE values compared to the other panels:
>         *   **GP-RS (grey circles):** Starts at approximately 2.5 and decreases, leveling off around 0.5 with a relatively wide shaded confidence interval.
>         *   **GP-US (brown squares):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.2 with a narrow shaded confidence interval.
>         *   **GP-VR (blue upward triangles):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.3 with a narrow shaded confidence interval.
>         *   **GP-EPIG (purple diamonds):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.3 with a narrow shaded confidence interval.
>         *   **ACE-US (green crosses):** Starts at approximately 2.5 and decreases, leveling off around 0.6 with a narrow shaded confidence interval.
>         *   **ALINE (ours) (orange stars):** Starts at approximately 2.5 and decreases, leveling off around 0.7 with a narrow shaded confidence interval.
> 
> A horizontal legend below the plots identifies each method by its color and marker:
> *   `GP-RS` (grey line with circles)
> *   `GP-US` (brown line with squares)
> *   `GP-VR` (blue line with upward triangles)
> *   `GP-EPIG` (purple line with diamonds)
> *   `ACE-US` (green line with crosses)
> *   `ALINE (ours)` (orange line with stars)

Figure A3: Predictive performance in terms of RMSE on three other active learning benchmark functions. Results show the mean and $95 \%$ confidence interval (CI) across 100 runs. Notably, on the Three Hump Camel function, the performance of amortized methods like Aline and ACE-US is limited, as its output scale significantly differs from the pre-training distribution, highlighting a scenario of distribution shift.

area for future work, such as training Aline on a broader prior distribution of functions, potentially leading to more universally capable models.

Acquisition visualization for AL. To qualitatively understand the behavior of our model, we visualize the query strategy employed by Aline for AL on a randomly sampled synthetic function Figure A4. This visualization illustrates how Aline iteratively selects query points to reduce uncertainty and refine its predictive posterior.

Hyperparameter inference visualization. We now visualize the evolution of Aline's estimated posterior distributions for the underlying GP hyperparameters for a randomly drawn 2D synthetic GP function (see Figure A5). The posteriors are shown after 1, 15, and 30 active data acquisition steps. As Aline strategically queries more informative data points, its posterior beliefs about these generative parameters become increasingly concentrated and accurate.

Inference time. To assess the computational efficiency of Aline, we report the inference times for the AL tasks in Table A4. The times represent the total duration to complete a sequence of 30 steps for 1D functions and 50 steps for 2D functions, averaged over 10 independent runs. As both Aline and ACE-US perform inference via a single forward pass per step once trained, they are significantly faster compared to traditional GP-based methods.

Table A4: Comparison of inference times (seconds) for different AL methods on 1D (30 steps) and 2D (50 steps) tasks. Values are averaged over 10 runs (mean $\pm$ standard deviation).

| Methods | Inference time (s) |  |
| :-- | :--: | :--: |
|  | 1D \& 30 steps | 2D \& 50 steps |
| GP-US | $0.62 \pm 0.09$ | $1.72 \pm 0.23$ |
| GP-VR | $1.41 \pm 0.14$ | $4.03 \pm 0.18$ |
| GP-EPIG | $1.34 \pm 0.11$ | $3.43 \pm 0.24$ |
| ACE-US | $0.08 \pm 0.00$ | $0.19 \pm 0.02$ |
| ALINE | $0.08 \pm 0.00$ | $0.19 \pm 0.02$ |

---

#### Page 26

> **Image description.** This image presents a grid of 20 individual line graphs, arranged in 4 rows and 5 columns, each labeled sequentially from "Step 1" to "Step 20". These graphs illustrate a sequential query strategy, likely for a Gaussian Process (GP) model, over 20 iterative steps.
> 
> Each individual graph shares common visual elements:
> *   **Axes**: A horizontal x-axis labeled "x" ranging from -4 to 4, and a vertical y-axis labeled "y" with varying ranges (e.g., -1.5 to 1.5 in early steps, narrowing to -0.5 to 1.0 in later steps).
> *   **Legend (visible in Step 1 only)**:
>     *   "Prediction": A solid blue line representing the model's current prediction.
>     *   "Ground Truth": A dashed red line representing the true underlying function.
>     *   "Targets": Small, dark red circular markers, indicating specific points of interest or the true values at certain locations.
>     *   "Context": Small, green circular markers, representing data points that have already been queried and observed by the model.
>     *   "Next Query": A dashed red vertical line, indicating the x-coordinate where the model proposes to query next.
> *   **Uncertainty Region**: A light blue shaded area around the "Prediction" line, representing the model's uncertainty or confidence interval.
> 
> **Progression across the 20 steps:**
> *   **Steps 1-5**: In the initial steps, the "Prediction" (blue line) deviates significantly from the "Ground Truth" (red dashed line), and the "Uncertainty" (blue shaded region) is broad across the entire x-range. The "Next Query" (red dashed vertical line) is strategically placed in regions of high uncertainty or where the prediction is poor. For example, in Step 1, the query is around x=-2.5. By Step 5, the prediction starts to align better with the ground truth, and the uncertainty begins to reduce around the initial "Context" points.
> *   **Steps 6-10**: As more "Context" points (green dots) accumulate from previous queries, the "Prediction" line increasingly conforms to the "Ground Truth" curve. The "Uncertainty" region visibly narrows in areas where context points are dense. The "Next Query" continues to target regions where uncertainty remains high or where the prediction still needs refinement.
> *   **Steps 11-15**: The model's "Prediction" closely matches the "Ground Truth" across most of the x-range. The "Uncertainty" region is significantly reduced, becoming very thin in many areas, indicating high confidence. The y-axis range in these plots has also narrowed, reflecting a more precise prediction. The "Next Query" points are still visible, suggesting continuous refinement.
> *   **Steps 16-20**: In the final steps, the "Prediction" line is almost perfectly aligned with the "Ground Truth" line, and the "Uncertainty" region is minimal across the entire range of x. The model has learned the underlying function very well. The "Next Query" continues to be placed, likely for minor adjustments or to confirm the model's high confidence. The density of "Context" points (green dots) has increased substantially, covering the ground truth curve densely.
> 
> Overall, the sequence of graphs demonstrates how an active learning or sequential query strategy progressively reduces uncertainty and improves the model's prediction by strategically selecting new data points to query over time. The model's confidence and accuracy visibly increase with each step.

Figure A4: Sequential query strategy of ALINE on a 1D synthetic GP function over 20 steps. As more points are queried, the model's prediction increasingly aligns with the ground truth, and the uncertainty is strategically reduced.

# D. 3 Benchmarking on Bayesian experimental design tasks 

In this section, we provide additional qualitative results for Aline's performance on the BED benchmark tasks. Specifically, for the Location Finding task, we visualize the sequence of designs chosen by Aline and the resulting posterior distribution over the hidden source's location (Figure A6). For the CES task, we present the estimated marginal posterior distributions for the model parameters, comparing them against their true underlying values (Figure A7). We see that Aline offers accurate parameter inference.

## D. 4 Psychometric model

Demonstrations of flexibility. We conduct two ablations to explicitly validate Aline's flexible targeting capabilities.
First, we test the ability to switch targets mid-rollout. We configure a single experiment where for the first 15 steps, the target is threshold \& slope parameters, and at step 16, the target is switched to the guess rate \& lapse rate. As shown in Figure A8(a), Aline's acquisition strategy adapts immediately and correctly, shifting its queries from the decision threshold region to the extremes of the stimulus range to gain maximal information about the new targets.

Second, we test generalization to novel target combinations. A single Aline model is trained to handle two distinct targets separately: (1) threshold \& slope and (2) guess \& lapse rate. At deployment, we task this model with a novel, unseen combination: targeting all four parameters simultaneously. As shown in Figure A8(b), the resulting policy is a sensible mixture of the two underlying strategies it has learned, strategically alternating queries between points near the decision threshold and points at the extremes. This confirms that Aline can successfully compose its learned strategies to generalize to new inference goals at runtime.

Inference time. We additionally assess the computational efficiency of each method in proposing the next design point. The average per-step design proposal time, measured over the 30-step psychometric experiments across 20 runs, is $0.002 \pm 0.00$ s for Aline, $0.07 \pm 0.00$ s for QUEST+, and $0.02 \pm 0.00$ s for Psi-marginal. Methods like QUEST+ and Psi-marginal, which often rely on

---

#### Page 27

> **Image description.** The image displays a 3x3 grid of nine density plots, illustrating the evolution of posterior distributions for three different parameters over three time steps. The grid is organized into three columns, labeled "t = 1", "t = 15", and "t = 30" at the top, representing increasing active query steps. Each column contains three plots, corresponding to "Lengthscale 1 Posterior", "Lengthscale 2 Posterior", and "Scale Posterior" from top to bottom. Horizontal black arrows connect the plots in each row, indicating a progression from left to right (e.g., from t=1 to t=15, and t=15 to t=30).
> 
> Each individual plot shares common visual elements:
> - A white background with a light gray grid.
> - A y-axis labeled "Density", with numerical values ranging from 0.0 up to varying maximums depending on the plot.
> - A blue curve outlining a shaded purple area, representing the "ALINE Posterior" distribution.
> - A vertical dashed red line, representing the "True Value".
> - The x-axis label varies depending on the parameter being estimated.
> 
> **Column 1: t = 1**
> - **Top plot (Lengthscale 1 Posterior):** The x-axis is "Lengthscale 1 Value" (0 to 3). The blue curve is broad and relatively flat, peaking around 1.5-2.0. The red dashed line is at x = 1.0. A legend in the top-left plot clarifies "ALINE Posterior" (blue curve) and "True Value" (red dashed line).
> - **Middle plot (Lengthscale 2 Posterior):** The x-axis is "Lengthscale 2 Value" (0 to 3). The blue curve is broad and somewhat bimodal, with peaks around 1.0 and 2.0-2.5. The red dashed line is at x = 2.6.
> - **Bottom plot (Scale Posterior):** The x-axis is "Scale Value" (0.00 to 1.00). The blue curve is broad, with a sharp peak around 0.15-0.20, then gradually decreasing. The red dashed line is at x = 0.15.
> 
> **Column 2: t = 15**
> - **Top plot (Lengthscale 1 Posterior):** The x-axis is "Lengthscale 1 Value" (0 to 3). The blue curve is noticeably narrower and taller than at t=1, peaking sharply around 1.2-1.3. The red dashed line remains at x = 1.0.
> - **Middle plot (Lengthscale 2 Posterior):** The x-axis is "Lengthscale 2 Value" (0.5 to 3.0). The blue curve is narrower and taller, still somewhat bimodal but more concentrated, with peaks around 1.5 and 2.2. The red dashed line is at x = 2.6.
> - **Bottom plot (Scale Posterior):** The x-axis is "Scale Value" (0.00 to 1.00). The blue curve is significantly narrower and taller, peaking sharply around 0.15-0.20. The red dashed line is at x = 0.15.
> 
> **Column 3: t = 30**
> - **Top plot (Lengthscale 1 Posterior):** The x-axis is "Lengthscale 1 Value" (0.5 to 3.0). The blue curve is very narrow and tall, peaking sharply around 1.1-1.2. The red dashed line is at x = 1.0. The peak is now very close to the true value.
> - **Middle plot (Lengthscale 2 Posterior):** The x-axis is "Lengthscale 2 Value" (1.5 to 3.0). The blue curve is very narrow and tall, now appearing unimodal and peaking sharply around 2.6-2.7. The red dashed line is at x = 2.6. The peak is very close to the true value.
> - **Bottom plot (Scale Posterior):** The x-axis is "Scale Value" (0.0 to 0.6). The blue curve is extremely narrow and tall, peaking sharply around 0.15-0.16. The red dashed line is at x = 0.15. The peak is almost perfectly aligned with the true value.
> 
> In summary, across all three rows, as 't' increases from 1 to 30, the posterior distributions become progressively more concentrated (narrower and taller) and their peaks shift closer to the respective "True Value" lines, indicating an increase in precision and accuracy of the parameter estimation over time.

Figure A5: Estimated posteriors for the two lengthscales and the output scale obtained from Aline after $t=1, t=15$, and $t=30$ active query steps. The posteriors progressively concentrate around the true parameter values as more data is acquired.

grid-based posterior estimation, face rapidly increasing computational costs as the parameter space dimensionality or required grid resolution grows. Aline, however, estimates the posterior via the transformer in a single forward pass, making its inference time largely insensitive to these factors. Thus, this computational efficiency gap is anticipated to become even more pronounced for more complex psychometric models.

# E Computational resources and software 

All experiments presented in this work, encompassing model development, hyperparameter optimization, baseline evaluations, and preliminary analyses, are performed on a GPU cluster equipped with AMD MI250X GPUs. The total computational resources consumed for this research, including all development stages and experimental runs, are estimated to be approximately 5000 GPU hours. For each experiment, it takes around 20 hours to train an Aline model for $10^{5}$ epochs. The core code base is built using Pytorch (https://pytorch.org/, License: modified BSD license). For the Gaussian Process (GP) based baselines, we utilize Scikit-learn [56] (https://scikit-learn.org/, License: modified BSD license). The DAD baseline is adapted from the original authors' publicly available code [23] (https://github.com/ae-foster/dad; MIT License). Our implementations of the RL-BOED and vsOED baselines are adapted from the official repositories provided by [6] (https://github.com/yasirbarlas/RL-BOED; MIT License) and [65] (https://github.com/wgshen/vsOED; MIT License), respectively. We use questplus package (https://github.com/hoechenberger/questplus, License: GPL-3.0) to implement QUEST+, and use Psi-staircase (https://github.com/NNiehof/Psi-staircase, License: GPL-3.0) to implement the Psi-marginal method.

---

#### Page 28

> **Image description.** A contour plot titled "Location Finding" visualizes a two-dimensional probability distribution, overlaid with data points and a marker. The plot area is a square grid ranging from 0.0 to 1.0 on both the horizontal (x) and vertical (y) axes.
> 
> The background of the plot is filled with concentric, roughly elliptical contour lines, representing a probability density. The color gradient for these contours ranges from dark blue-purple at the outer edges (lowest probability density) to light blue-teal and nearly white in the central region (highest probability density). The highest density area is centered approximately at coordinates (0.5, 0.6).
> 
> Superimposed on this contour map are several small, circular data points and a single star symbol:
> *   **Data Points ($\text{x}_t$)**: Numerous small, circular points are scattered across the plot. A dense cluster of these points, ranging in color from light orange to dark reddish-brown, is concentrated around the central high-density region of the contour plot, specifically around (0.5, 0.6). A few additional, lighter orange points are sparsely distributed further out, for example, near (0.0, 0.0), (0.9, 0.1), and (0.9, 0.9).
> *   **True Location ($\theta$)**: A single, prominent blue star symbol is positioned within the dense cluster of data points, also centered around (0.5, 0.6), indicating a specific target location.
> 
> Two vertical colorbars are positioned to the right of the main plot:
> *   **Left Colorbar (Posterior log probability density)**: This colorbar displays a gradient from black at the bottom to white at the top, with intermediate shades of gray and light blue-teal, corresponding to the contour colors. It is labeled vertically as "Posterior $\text{log } q(\theta|\mathcal{D}_T)$". The scale ranges from -1800 at the bottom to 0 at the top, with tick marks at -1800, -1500, -1200, -900, -600, -300, and 0.
> *   **Right Colorbar (Time step)**: This colorbar shows a gradient from light orange at the bottom to dark reddish-brown at the top. It is labeled vertically as "Time step $\text{t}$". The scale ranges from 0 at the bottom to 30 at the top, with tick marks at 0, 5, 10, 15, 20, 25, and 30. The color of the circular data points on the main plot corresponds to this time step color scale, with earlier steps being lighter orange and later steps being darker brown.
> 
> A small legend box is located in the top-right corner of the main plot, explaining the symbols:
> *   An orange dot next to "$\text{x}_t$"
> *   A blue star next to "$\theta$"
> 
> The visual arrangement suggests that the contour plot represents a probability distribution, the blue star marks a true value, and the colored dots represent sampled or queried locations over time, with their concentration around the star indicating an inference process converging towards the true location.

Figure A6: Visualization of Aline's design policy and resulting posterior for the Location Finding task. The contour plot shows the log posterior probability density of the source location $\theta$ (true location marked by blue star) after $T=30$ steps. Queried locations, with color indicating the time step of acquisition, demonstrating a concentration of queries around the true source.

> **Image description.** A series of three line graphs, arranged horizontally, displays marginal posterior distributions for different parameters. Each graph features a white background, black axes, and labels, and contains one or more colored lines representing probability distributions, along with a dashed red vertical line indicating a "true parameter value" as per the context.
> 
> 1.  **Left Graph (Parameter $\rho$)**:
>     *   **Title**: $\rho$
>     *   **Y-axis**: Labeled "$p(\theta)$", ranging from 0 to 10, with major ticks at 0, 5, and 10.
>     *   **X-axis**: Labeled "$\rho$", ranging from 0.0 to 1.0, with major ticks at 0.0, 0.5, and 1.0.
>     *   **Content**: A single blue line forms a distinct, bell-shaped curve, peaking sharply at approximately x=0.7 and reaching a maximum y-value just above 10. The curve is narrow, indicating a concentrated distribution. A dashed red vertical line is positioned precisely at the peak of the blue curve, around x=0.7.
> 
> 2.  **Middle Graph (Parameter $\alpha$)**:
>     *   **Title**: $\alpha$
>     *   **Y-axis**: Unlabeled but implicitly representing $p(\theta)$, ranging from 0 to 40, with major ticks at 0, 20, and 40.
>     *   **X-axis**: Labeled "$\alpha$", ranging from 0.0 to 1.0, with major ticks at 0.0, 0.5, and 1.0.
>     *   **Content**: This graph shows three very narrow, sharp peaks, each representing a distribution.
>         *   A blue line peaks around x=0.3, reaching a y-value slightly above 40.
>         *   A green line peaks slightly to the right of the blue line, around x=0.35, also reaching a y-value slightly above 40.
>         *   An orange line peaks further to the right, around x=0.4, also reaching a y-value slightly above 40.
>         All three peaks are extremely narrow and tall, indicating highly concentrated distributions. Three dashed red vertical lines are present, each aligned with the peak of one of the colored curves, indicating their respective "true parameter values."
> 
> 3.  **Right Graph (Parameter $u$)**:
>     *   **Title**: $u$
>     *   **Y-axis**: Unlabeled but implicitly representing $p(\theta)$, ranging from 0 to 2, with major ticks at 0, 1, and 2.
>     *   **X-axis**: Labeled "$\log(u)$", ranging from -5.0 to 5.0, with major ticks at -5.0, 0.0, and 5.0.
>     *   **Content**: A single blue line forms an extremely narrow, almost vertical spike, peaking around x=-4.0 and reaching a y-value just above 2. This signifies a very highly concentrated distribution. A dashed red vertical line is positioned precisely at the peak of the blue curve, around x=-4.0.

Figure A7: Aline's estimated marginal posterior distributions for the parameters of the CES task after $T=10$ query steps. The dashed red lines indicate the true parameter values. The posteriors are well-concentrated around the true values, demonstrating accurate parameter inference.

> **Image description.** The image displays two side-by-side scatter plots, labeled (a) and (b), illustrating data points over a series of steps. Both plots share a common y-axis label and scale, and similar x-axis labels and scales. The background of both plots is white, with gray lines for axes and tick marks.
> 
> **Panel (a):**
> This scatter plot shows "Stimuli Values" on the y-axis, ranging from -5 to 5, against "Number of Steps $t$" on the x-axis, ranging from 0 to 30. A horizontal dashed gray line is present at y=0. The data points are represented by circular markers, which exhibit a color gradient from light orange for earlier steps to dark brown for later steps.
> Initially, from approximately step 0 to step 15, the points are clustered around the dashed line at y=0. These points start as light orange and gradually become a darker orange. Around step 15, the pattern changes distinctly: the points diverge into two separate trajectories. One trajectory moves upwards, with points increasing in y-value from approximately 2 to 5, becoming progressively darker orange to brown. The second trajectory moves downwards, with points decreasing in y-value from approximately -2 to -4.5, also becoming darker orange to brown. The panel is labeled "(a)" centered below the x-axis.
> 
> **Panel (b):**
> This scatter plot also shows "Stimuli Values" on the y-axis (implied by the shared label with panel a) against "Number of Steps $t$" on the x-axis, with the same ranges as panel (a). A horizontal dashed gray line is present at approximately y=2.5. The data points are again circular markers with a color gradient from light orange to dark brown, indicating progression over steps.
> From approximately step 0 to step 5, the points are somewhat scattered, with some near y=0 and others near y=5, all in light orange hues. From step 5 onwards, the majority of the points cluster closely around the dashed line at y=2.5. As the number of steps increases, these clustered points transition from orange to dark brown. There is one notable outlier point around step 18, located at approximately y=-3, depicted in a dark orange color. The panel is labeled "(b)" centered below the x-axis.

Figure A8: Demonstration of Aline's runtime flexibility on the psychometric task. (a) The acquisition strategy adapts after the inference target is switched mid-rollout from (threshold \& slope) to (guess \& lapse rate). (b) When tasked with a novel combined target (all four parameters), the policy generalizes by mixing the two distinct strategies it learned during training.