```
@inproceedings{huang2025aline,
  title={ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition},
  author={Daolang Huang and Xinyi Wen and Ayush Bharti and Samuel Kaski and Luigi Acerbi},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025},
}
```

---

#### Page 1

# ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition

Daolang Huang ${ }^{1,2}$ Xinyi Wen ${ }^{1,2}$ Ayush Bharti ${ }^{2}$ Samuel Kaski ${ }^{* 1,2,4}$ Luigi Acerbi ${ }^{* 3}$<br>${ }^{1}$ ELLIS Institute Finland<br>${ }^{2}$ Department of Computer Science, Aalto University, Finland<br>${ }^{3}$ Department of Computer Science, University of Helsinki, Finland<br>${ }^{4}$ Department of Computer Science, University of Manchester, UK<br>\{daolang.huang, xinyi.wen, ayush.bharti, samuel.kaski\}@aalto.fi<br>luigi.acerbi@helsinki.fi

#### Abstract

Many critical applications, from autonomous scientific discovery to personalized medicine, demand systems that can both strategically acquire the most informative data and instantaneously perform inference based upon it. While amortized methods for Bayesian inference and experimental design offer part of the solution, neither approach is optimal in the most general and challenging task, where new data needs to be collected for instant inference. To tackle this issue, we introduce the Amortized Active Learning and Inference Engine (ALINE), a unified framework for amortized Bayesian inference and active data acquisition. ALINE leverages a transformer architecture trained via reinforcement learning with a reward based on self-estimated information gain provided by its own integrated inference component. This allows it to strategically query informative data points while simultaneously refining its predictions. Moreover, Aline can selectively direct its querying strategy towards specific subsets of model parameters or designated predictive tasks, optimizing for posterior estimation, data prediction, or a mixture thereof. Empirical results on regression-based active learning, classical Bayesian experimental design benchmarks, and a psychometric model with selectively targeted parameters demonstrate that Aline delivers both instant and accurate inference along with efficient selection of informative points.

## 1 Introduction

Bayesian inference [28] and Bayesian experimental design [61] offer principled mathematical means for reasoning under uncertainty and for strategically gathering data, respectively. While both are foundational, they introduce notorious computational challenges. For example, in scenarios with continuous data streams, repeatedly applying gold-standard inference methods such as Markov Chain Monte Carlo (MCMC) [13] to update posterior distributions can be computationally demanding, leading to various approximate sequential inference techniques [10, 18], yet challenges in achieving both speed and accuracy persist. Similarly, in Bayesian experimental design (BED) or Bayesian active learning (BAL), the iterative estimation and optimization of design objectives can become costly, especially in sequential learning tasks requiring rapid design decisions [21, 30, 55], such as in psychophysical experiments where the goal is to quickly infer the subject's perceptual or cognitive parameters [70, 57]. Moreover, a common approach in BED involves greedily optimizing for singlestep objectives, such as the Expected Information Gain (EIG), which measures the anticipated

[^0]
[^0]: \* Equal contribution.

---

#### Page 2

Table 1: Comparison of different methods for amortized Bayesian inference and data acquisition.

| Method                                       | Amortized Inference |              | Amortized Data Acquisition |              |              |
| -------------------------------------------- | ------------------- | ------------ | -------------------------- | ------------ | ------------ |
|                                              | Posterior           | Predictive   | Posterior                  | Predictive   | Flexible     |
| Neural Processes [27, 26, 52, 53]            | $\chi$              | $\checkmark$ | $\chi$                     | $\chi$       | $\chi$       |
| Neural Posterior Estimation [49, 54, 34, 60] | $\checkmark$        | $\chi$       | $\chi$                     | $\chi$       | $\chi$       |
| DAD [23], RL-BOED [9]                        | $\chi$              | $\chi$       | $\checkmark$               | $\chi$       | $\chi$       |
| RL-sCEE [8], vsOED [65]                      | $\checkmark$        | $\chi$       | $\checkmark$               | $\chi$       | $\chi$       |
| AAL [46]                                     | $\chi$              | $\chi$       | $\chi$                     | $\checkmark$ | $\chi$       |
| JANA [59], Simformer [31], ACE [15]          | $\checkmark$        | $\checkmark$ | $\chi$                     | $\chi$       | $\chi$       |
| Aline (this work)                            | $\checkmark$        | $\checkmark$ | $\checkmark$               | $\checkmark$ | $\checkmark$ |

reduction in uncertainty. However, this leads to myopic designs that may be suboptimal, which do not fully consider how current choices influence future learning opportunities [23].

To address these issues, a promising avenue has been the development of amortized methods, leveraging recent advances in deep learning [75]. The core principle of amortization involves pretraining a neural network on a wide range of simulated or existing problem instances, allowing the network to "meta-learn" a direct mapping from a problem's features (like observed data) to its solution (such as a posterior distribution or an optimal experimental design). Consequently, at deployment, this trained network can provide solutions for new, related tasks with remarkable efficiency, often in a single forward pass. Amortized Bayesian inference (ABI) methods—such as neural posterior estimation [49, 54, 34, 60] that target the posterior, or neural processes [27, 26, 52, 53] that target the posterior predictive distribution—yield almost instantaneous results for new data, bypassing MCMC or other approximate inference methods. Similarly, methods for amortized data acquisition, including amortized BED [23, 41, 40, 9] and BAL [46], instantaneously propose the next design that targets the learning of the posterior or the posterior predictive distribution, respectively, using a deep policy network—bypassing the iterative optimization of complex information-theoretic objectives.

While amortized approaches have significantly advanced Bayesian inference and data acquisition, progress has largely occurred in parallel. ABI offers rapid inference but typically assumes passive data collection, not addressing strategic data acquisition under limited budgets and data constraints common in fields such as clinical trials [7] and material sciences [48, 44]. Conversely, amortized data acquisition excels at selecting informative data points, but often the subsequent inference update based on new data is not part of the amortization, potentially requiring separate, costly procedures like MCMC. This separation means the cycle of efficiently deciding what data to gather and then instantaneously updating beliefs has not yet been seamlessly integrated. Furthermore, existing amortized data acquisition methods often optimize for information gain across all model parameters or a fixed predictive target, lacking the flexibility to selectively target specific subsets of parameters or adapt to varying inference goals. This is a significant drawback in scenarios with nuisance parameters [58] or when the primary interest lies in particular aspects of the model or predictions—which might not be fully known in advance. A unified framework that jointly amortizes both active data acquisition and inference, while also offering flexible acquisition goals, would therefore be highly beneficial.

In this paper, we introduce Amortized Active Learning and INference Engine (Aline), a novel framework designed to overcome these limitations by unifying amortized Bayesian inference and active data acquisition within a single, cohesive system (Table 1). Aline utilizes a transformer-based architecture [69] that, in a single forward pass, concurrently performs posterior estimation, generates posterior predictive distributions, and decides which data point to query next. Critically, and in contrast to existing methods, Aline offers flexible, targeted acquisition: it can dynamically adjust at runtime its data-gathering strategy to focus on any specified combination of model parameters or predictive tasks. This is enabled by an attention mechanism allowing the policy to condition on specific inference goals, making it particularly effective in the presence of nuisance variables [58] or for focused investigations. Aline is trained using a self-guided reinforcement learning objective; the reward is the improvement in the log-probability of its own approximate posterior over the selected targets, a principle derived from variational bounds on the expected information gain [24]. Extensive experiments on diverse tasks demonstrate Aline’s ability to simultaneously deliver fast, accurate inference and rapidly propose informative data points.

---

#### Page 3

2 Background

Consider a parametric conditional model defined on some space $\mathcal{Y}\subseteq\mathbb{R}^{d_{\mathcal{Y}}}$ of output variables $y$ given inputs (or covariates) $x \in \mathcal{X} \subseteq \mathbb{R}^{d_{\mathcal{X}}}$, and parameterized by $\theta \in \Theta \subseteq \mathbb{R}^{L}$. Let $\mathcal{D}_{T}=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{T}$ be a collection of $T$ data points (or context) and $p\left(\mathcal{D}_{T} \mid \theta\right) \equiv p\left(y_{1: T} \mid x_{1: T}, \theta\right)$ denote the likelihood function associated with the model, which we assume to be well-specified in this paper (i.e., it matches the true data generation process). Given a prior distribution $p(\theta)$, the classical Bayesian inference or prediction problem involves estimating either the posterior distribution $p\left(\theta \mid \mathcal{D}_{T}\right) \propto p\left(\mathcal{D}_{T} \mid \theta\right) p(\theta)$, or the posterior predictive distribution $p\left(y_{1: M}^{*} \mid x_{1: M}^{*}, \mathcal{D}_{T}\right)=\mathbb{E}_{p\left(\theta \mid \mathcal{D}_{T}\right)}\left[p\left(y_{1: M}^{*} \mid x_{1: M}^{*}, \theta, \mathcal{D}_{T}\right)\right]$ over target outputs $y_{1: M}^{*}:=\left(y_{1}^{*}, \ldots, y_{M}^{*}\right)$ corresponding to a given set of target inputs $x_{1: M}^{*}:=\left(x_{1}^{*}, \ldots, x_{M}^{*}\right)$. Estimating these quantities repeatedly via approximate inference methods such as MCMC can be computationally costly [28], motivating the need for amortized inference methods.

Amortized Bayesian inference (ABI). ABI methods involve training a conditional density network $q_{\phi}$, parameterized by learnable weights $\phi$, to approximate either the posterior predictive distribution $q_{\phi}\left(y_{1: M}^{*} \mid x_{1: M}^{*}, \mathcal{D}_{T}\right) \approx p\left(y_{1: M}^{*} \mid x_{1: M}^{*}, \mathcal{D}_{T}\right)$ [26, 42, 33, 11, 38, 12, 53, 52], the joint posterior $q_{\phi}\left(\theta \mid \mathcal{D}_{T}\right) \approx p\left(\theta \mid \mathcal{D}_{T}\right)[49,54,34,60]$, or both [59, 31, 15]. These networks are usually trained by minimizing the negative log-likelihood (NLL) objective with respect to $\phi$ :

$$
\mathcal{L}(\phi)= \begin{cases}-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right) p\left(x_{1: M}^{*}, y_{1: M}^{*} \mid \theta\right)}\left[\log q_{\phi}\left(y_{1: M}^{*} \mid x_{1: M}^{*}, \mathcal{D}_{T}\right)\right], & \text { (predictive tasks) } \\ -\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right)}\left[\log q_{\phi}\left(\theta \mid \mathcal{D}_{T}\right)\right], & \text { (posterior estimation) }\end{cases}
$$

where the expectation is over datasets simulated from the generative process $p\left(\mathcal{D}_{T} \mid \theta\right) p(\theta)$. Once trained, $q_{\phi}$ can then perform instantaneous approximate inference on new contexts and unseen data points with a single forward pass. However, these ABI methods do not have the ability to strategically collect the most informative data points to be included in $\mathcal{D}_{T}$ in order to improve inference outcomes.

Amortized data acquisition. BED [47, 14, 63, 61] methods aim to sequentially select the next input (or design parameter) $x$ to query in order to maximize the Expected Information Gain (EIG), that is, the information gained about parameters $\theta$ upon observing $y$ :

$$
\operatorname{EIG}(x):=\mathbb{E}_{p(y \mid x)}[H[p(\theta)]-H[p(\theta \mid x, y)]]
$$

where $H$ is the Shannon entropy $H[p(\cdot)]=-\mathbb{E}_{p(\cdot)}[\log p(\cdot)]$. Directly computing and optimizing EIG sequentially at each step of an experiment is computationally expensive due to the nested expectations, and leads to myopic designs. Amortized BED methods address these limitations by offline learning a design policy network $\pi_{\psi}: \mathcal{X} \times \mathcal{Y} \rightarrow \mathcal{X}$, parameterized by $\psi[23,40,9]$, such that at any step $t$ the policy $\pi_{\psi}$ proposes a query $x_{t} \sim \pi_{\psi}\left(\cdot \mid \mathcal{D}_{t-1}, \theta\right)$ to acquire a data point $y_{t} \sim p\left(y \mid x_{t}, \theta\right)$, forming $\mathcal{D}_{t}=\mathcal{D}_{t-1} \cup\left\{\left(x_{t}, y_{t}\right)\right\}$. To propose non-myopic designs, $\pi_{\psi}$ is trained by maximizing tractable lower bounds of the total EIG over $T$-step sequential trajectories generated by the policy $\pi_{\psi}$ :

$$
\operatorname{sEIG}(\psi)=\mathbb{E}_{p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}[H[p(\theta)]-H\left[p\left(\theta \mid \mathcal{D}_{T}\right)\right]]
$$

By pre-compiling the design strategy into the policy network, amortized BED methods allow for near-instantaneous design proposals during the deployment phase via a fast forward pass. Typically, these amortized BED methods are designed to maximize information gain about the full set of model parameters $\theta$. Separately, for applications where the primary interest lies in reducing predictive uncertainty rather than parameter uncertainty, objectives like the Expected Predictive Information Gain (EPIG) [67] have been proposed, so far in non-amortized settings:

$$
\operatorname{EPIG}(x)=\mathbb{E}_{p_{\star}\left(x^{\star}\right) p(y \mid x)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]-H\left[p\left(y^{\star} \mid x^{\star}, x, y\right)\right]\right]
$$

This measures the EIG about predictions $y^{\star}$ at target inputs $x^{\star}$ drawn from a target input distribution $p_{\star}\left(x^{\star}\right)$. Notably, current amortized data acquisition methods are inflexible: they are generally trained to learn about all parameters $\theta$ (via objectives like sEIG) and lack the capability to dynamically target specific subsets of parameters or adapt their acquisition strategy to varying inference goals at runtime.

Related work. A major family of ABI methods is Neural Processes (NPs) [27, 26], that learn a mapping from the observed context data points to a predictive distribution for new target points. Early NPs often employed MLP-based encoders [27, 26, 38, 11], while recent works utilize more advanced attention and transformer architectures [42, 53, 52, 19, 20, 5, 4]. Complementary to

---

#### Page 4

> **Image description.** A conceptual workflow diagram illustrates the ALINE system, demonstrating its sequential data acquisition and inference capabilities. The diagram is composed of three main, rounded rectangular panels with dashed borders, arranged horizontally, and interconnected by dashed arrows.
>
> The leftmost panel, titled "Experimentation" and set against a light blue background, depicts the input and output of an experimental process. At the top, two chemical structures are shown: a benzene ring labeled "x₁" and a fused polycyclic aromatic hydrocarbon labeled "xₜ". An ellipsis "..." separates these two structures, indicating a sequence of inputs. Arrows point downwards from these structures towards a central flask icon, which contains a small amount of liquid. From the flask, arrows point downwards and outwards to "y₁" and "yₜ", representing the experimental outputs corresponding to the inputs "x₁" and "xₜ".
>
> The central panel, representing the "ALINE" system, has a purple background. It is a stack of four vertically aligned, rounded rectangular blocks, with two smaller blocks at the very top. From bottom to top, these blocks are:
>
> - "Data Embedder"
> - "Transformer"
> - "Selective Mask"
> - "Policy Head" (left, top) and "Inference Head" (right, top).
>   Upward arrows indicate the flow of information from the "Data Embedder" through the "Transformer" and "Selective Mask" to both "Policy Head" and "Inference Head". The acronym "ALINE" is written in orange text at the bottom center of this panel.
>
> The rightmost panel, associated with "Approximate inference" (text above the panel), also has a light blue background and is divided horizontally by a dashed line into two sections.
>
> - The top section is labeled "Posterior". It displays two bell-shaped curves (representing probability distributions) in pink/red, labeled "Step 1" and "Step t", separated by an ellipsis "...". The curve for "Step t" appears slightly narrower and taller than "Step 1", suggesting increased certainty or precision over time.
> - The bottom section is labeled "Posterior Predictive". It shows two line graphs with shaded uncertainty regions, also in pink/red, labeled "Step 1" and "Step t", separated by an ellipsis "...". The graph for "Step 1" shows a wider shaded region, while "Step t" shows a narrower shaded region around the central line, indicating a reduction in predictive uncertainty.
>
> Dashed arrows connect these panels, illustrating the workflow:
>
> - A dashed arrow labeled "Next query" (with "instant" in green below it) points from the "Policy Head" of the ALINE system to the "Experimentation" panel, specifically towards the chemical structures.
> - A dashed arrow labeled "Updated history" points from the outputs (y₁, yₜ) of the "Experimentation" panel to the "Data Embedder" of the ALINE system.
> - A dashed arrow labeled "Approximate inference" (with "instant" in green below it) points from the "Inference Head" of the ALINE system to the "Posterior" section of the rightmost panel.
> - A dashed arrow labeled "Posterior improvement" points from the "Posterior Predictive" section of the rightmost panel back to the "Data Embedder" of the ALINE system.

Figure 1: Conceptual workflow of ALINE, demonstrating its capability to sequentially query informative data points and perform rapid posterior or predictive inference based on the gathered data.

NPs, methods within simulation-based inference [17] focus on amortizing the posterior distribution [54, 49, 34, 60, 51, 75]. More recently, methods for amortizing both the posterior and posterior predictive distributions have been proposed [31, 59, 15]. Specifically, ACE [15] shows how to flexibly condition on diverse user-specified inference targets, a method ALINE incorporates for its own flexible inference capabilities. Building on this principle of goal-directed adaptability, ALINE advances it by integrating a learned policy that dynamically tailors the data acquisition strategy to the specified objectives. Existing amortized BED or BAL methods [23, 40, 9, 46] that learn an offline design policy do not provide real-time estimates of the posterior, unlike ALINE. Recent exceptions include methods like RL-sCEE [8] and vsOED [65], that use a variational posterior bound of EIG to provide amortized posterior inference via a separate proposal network. Compared to these methods, ALINE uses a single, unified architecture where the same model performs amortized inference for both posterior and posterior predictive distributions, and learns the flexible acquisition policy.

## 3 Amortized active learning and inference engine

**Problem setup.** We aim to develop a system that intelligently acquires a sequence of $T$ informative data points, $\mathcal{D}_{T} = \{(x_i, y_i)\}_{i=1}^T$, to enable accurate and rapid Bayesian inference. This system must be flexible: capable of targeting different quantities of interest, such as subsets of model parameters or future predictions. To formalize this flexibility, we introduce a _target specifier_, denoted by $\xi \in \Xi$, which defines the specific inference goal. We consider two primary types of targets: (1) **Parameter targets** ($\xi_\beta^{\theta}$) with the goal to infer a specific subset of model parameters $\theta_S$, where $S \subseteq \{1, \ldots, L\}$ is an index set of parameters of interest. For example, $\xi_\{1,2\}^{\theta}$ would target the joint posterior of $\theta_1$ and $\theta_2$, while $\xi_\{1, \ldots, L\}$ targets all parameters, aligning with standard BED. We define $\mathcal{S} = \{S_1, \ldots, S_{|\mathcal{S}|}\}$ as the collection of all predefined parameter index subsets the system can target. (2) **Predictive targets** ($\xi_\hat{p}_*^{\theta}$), where the objective is to improve the posterior predictive distribution $p(y^\star | x^\star, \mathcal{D}_T)$ for inputs $x^\star$ drawn from a specified target input distribution $p_\star(x^\star)$. For simplicity, and following Smith et al. [67], we consider a single target distribution $p_\star(x^\star)$ in this work. The set of all target specifiers that ALINE is trained to handle is thus $\Xi = \{\xi_\beta^{\theta}\}_{S \in \mathcal{S}} \cup \{\xi_\hat{p}_*^{\theta}\}$. We assume a discrete distribution $p(\xi)$ over these possible targets, reflecting the likelihood or importance of each specific goal.

To achieve both instant, informative querying and accurate inference, we propose to jointly learn an _amortized inference model_ $q_\phi$ and an _acquisition policy_ $\pi_\psi$ within a single, integrated architecture. Given the accumulated data history $\mathcal{D}_{t-1}$ and a specific target $\xi \in \Xi$, the policy $\pi_\psi$ selects the next query $x_t$ designed to be most informative for that target. Subsequently, the new data point $(x_t, y_t)$ is observed, and the inference model $q_\phi$ updates its estimate of the corresponding posterior or posterior predictive distribution. A conceptual workflow of ALINE is illustrated in Figure 1. In the remainder of this section, we detail the objectives for training the inference network $q_\phi$ (Section 3.1) and the acquisition policy $\pi_\psi$ (Section 3.2), discuss their practical implementation (Section 3.3), and describe the unified model architecture (Section 3.4).

---

#### Page 5

# 3.1 Amortized inference

We use the inference network $q_{\phi}$ to provide accurate approximations of the true Bayesian posterior $p\left(\theta \mid \mathcal{D}_{T}\right)$ or posterior predictive distribution $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$, given the acquired data $\mathcal{D}_{T}$. We train $q_{\phi}$ via maximum-likelihood (Eq. 1). Specifically, for parameter targets $\xi=\xi_{S}^{\theta}$, our objective is:

$$
\mathcal{L}_{S}^{\theta}(\phi)=-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right)}\left[\log q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right] \approx-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right)}\left[\sum_{l \in S} \log q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{T}\right)\right]
$$

where we adopt a diagonal or mean field approximation, where the joint distribution is obtained as a product of marginals $q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right) \approx \prod_{l \in S} q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{T}\right)$. Analogously, for predictive targets $\xi=\xi_{p_{\star}}^{\eta^{*}}$, we assume a factorized likelihood over targets sampled from the target input distribution $p_{\star}\left(x^{\star}\right)$ :

$$
\begin{aligned}
\mathcal{L}_{p_{\star}}^{\eta^{*}}(\phi) & =-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right) p_{\star}\left(x_{1: M}^{\star}\right) p\left(y_{1: M}^{\star} \mid x_{1: M}^{\star}, \theta\right)}\left[\log q_{\phi}\left(y_{1: M}^{\star} \mid x_{1: M}^{\star}, \mathcal{D}_{T}\right)\right] \\
& \approx-\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \theta\right) p_{\star}\left(x_{1: M}^{\star}\right) p\left(y_{1: M}^{\star} \mid x_{1: M}^{\star}, \theta\right)}\left[\sum_{m=1}^{M} \log q_{\phi}\left(y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}_{T}\right)\right]
\end{aligned}
$$

The factorized form of these training objectives is a common scalable choice in the neural process literature [26, 52, 53, 15] and is more flexible than it might seem, as conditional marginal distributions can be extended to represent full joints autoregressively [53, 11, 15]. However, a full autoregressive model would require multiple forward passes to compute the reward signal for our policy at each training step, making the learning process computationally intractable. Therefore, for simplicity and tractability, within the scope of this paper we focus on the marginals, leaving the autoregressive extension to future work. Eqs. 5 and 6 form the basis for training the inference component $q_{\phi}$. Optimizing them minimizes the Kullback-Leibler (KL) divergence between the true target distributions (posterior or predictive) defined by the underlying generative process and the model's approximations $q_{\phi}$ [52]. Learning an accurate $q_{\phi}$ is crucial as it not only determines the quality of the final inference output but also serves as the basis for guiding the data acquisition policy, as we see next.

### 3.2 Amortized data acquisition

The quality of inference from $q_{\phi}$ depends critically on the informativeness of the acquired dataset $\mathcal{D}_{T}$. The acquisition policy $\pi_{\psi}$ is thus responsible for actively selecting a sequence of query-data pairs $\left(x_{t}, y_{t}\right)$ to maximize the information gained about a specific target $\xi$.
When targeting parameters $\theta_{S}$ (i.e., $\xi=\xi_{S}^{\theta}$ ), the objective is the total Expected Information Gain $\left(\mathrm{sEIG}_{\theta_{S}}\right)$ about $\theta_{S}$ over the $T$-step trajectory generated by $\pi_{\psi}$ (see Eq. 3):

$$
\mathrm{sEIG}_{\theta_{S}}(\psi)=\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right)}\left[\log p\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right]+H\left[p\left(\theta_{S}\right)\right]
$$

For completeness, we include a derivation of $\mathrm{sEIG}_{\theta_{S}}$ in Section A. 1 which is analogous to that of sEIG in [23]. Directly optimizing sEIG is generally intractable due to its reliance on the unknown true posterior $p\left(\theta_{S} \mid \mathcal{D}_{T}\right)$. We circumvent this by substituting $p\left(\theta_{S} \mid \mathcal{D}_{T}\right)$ with its approximation $q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)$ (from Eq. 5), yielding the tractable objective $\mathcal{J}_{S}^{\theta}$ for training $\pi_{\psi}$ :

$$
\mathcal{J}_{S}^{\theta}(\psi):=\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right)}\left[\sum_{l \in S} \log q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{T}\right)\right]+H\left[p\left(\theta_{S}\right)\right]
$$

The inference objective (Eq. 5) and this policy objective are thus coupled: $\mathcal{L}_{S}^{\theta}(\phi)$ depends on data $\mathcal{D}_{T}$ acquired through $\pi_{\psi}$, and $\mathcal{J}_{S}^{\theta}(\psi)$ depends on the inference network $q_{\phi}$.
Similarly, when targeting predictions for $\xi=\xi_{p_{\star}}^{\eta^{*}}$, we aim to maximize information about $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ for $x^{\star} \sim p_{\star}\left(x^{\star}\right)$. We extend the Expected Predictive Information Gain (EPIG) framework [67] to the amortized sequential setting, defining the total sEPIG:
Proposition 1. The total expected predictive information gain for a design policy $\pi_{\psi}$ over a data trajectory of length $T$ is:

$$
\begin{aligned}
s E P I G(\psi) & :=\mathbb{E}_{p_{\star}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]-H\left[p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]\right] \\
& =\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p_{\star}\left(x^{\star}\right) p\left(y^{\star} \mid x^{\star}, \theta\right)}\left[\log p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]+\mathbb{E}_{p_{\star}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right]
\end{aligned}
$$

This result adapts Theorem 1 in [23] for the predictive case (see Section A. 2 for proof) and, unlike single-step EPIG (Eq. 4), considers the entire trajectory $\mathcal{D}_{T}$ given the policy $\pi_{\psi}$.

---

#### Page 6

Now, similar to Eq. 8, we use the inference network $q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ to replace the true posterior predictive distribution $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ in Proposition 1 to obtain our active learning objective:

$$
\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi):=\mathbb{E}_{p(\theta) p\left(\mathcal{D}_{T} \mid \pi_{\psi}, \theta\right) p_{\star}\left(x^{\star}\right) p\left(y^{\star} \mid x^{\star}, \theta\right)}\left[\log q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right]+\mathbb{E}_{p_{\star}\left(x^{\star}\right)}\left[H\left[p\left(y^{\star} \mid x^{\star}\right)\right]\right]
$$

Finally, the following proposition proves that our acquisition objectives, $\mathcal{J}_{S}^{\theta}$ and $\mathcal{J}_{p_{\star}}^{y^{\star}}$, are variational lower bounds on the true total information gains ( $\mathrm{sEIG}_{\theta_{S}}$ and sEPIG, respectively), making them principled tractable objectives for our goal. The proof is given in Section A.3.
Proposition 2. Let the policy $\pi_{\psi}$ generate the trajectory $\mathcal{D}_{T}$. With $q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)$ approximating $p\left(\theta_{S} \mid \mathcal{D}_{T}\right)$, and $q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$ approximating $p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)$, we have $\mathcal{J}_{S}^{\theta}(\psi) \leq s E I G_{\theta_{S}}(\psi)$ and $\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi) \leq s E P I G(\psi)$. Moreover,

$$
\begin{aligned}
& s E I G_{\theta_{S}}(\psi)-\mathcal{J}_{S}^{\theta}(\psi)=\mathbb{E}_{p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[K L\left(p\left(\theta_{S} \mid \mathcal{D}_{T}\right) \mid \mid q_{\phi}\left(\theta_{S} \mid \mathcal{D}_{T}\right)\right)\right], \quad \text { and } \\
& s E P I G(\psi)-\mathcal{J}_{p_{\star}}^{y^{\star}}(\psi)=\mathbb{E}_{p_{\star}\left(x^{\star}\right) p\left(\mathcal{D}_{T} \mid \pi_{\psi}\right)}\left[K L\left(p\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right) \mid \mid q_{\phi}\left(y^{\star} \mid x^{\star}, \mathcal{D}_{T}\right)\right)\right]
\end{aligned}
$$

This principle of using approximate posterior (or predictive) distributions to bound information gain is foundational in Bayesian experimental design (e.g., [24]) and has been extended to sequential amortized settings [8, 65]. Maximizing these $\mathcal{J}$ objectives thus encourages policies that increase information about the targets. The tightness of these bounds is governed by the expected KL divergence between the true quantities and their approximation, with a more accurate $q_{\phi}$ leading to tighter bounds and a more effective training signal for the policy. Additionally, our objectives solely rely on Aline's variational posterior, which does not require an explicit likelihood, making it naturally applicable to problems with implicit likelihoods where only forward sampling is possible.

Data acquisition objective for Aline. To handle any target $\xi \in \Xi$ from a user-specified set, we unify the previously defined acquisition objectives. We define $\mathcal{J}(\psi, \xi)$ based on the type of target $\xi$ :

$$
\mathcal{J}(\psi, \xi)= \begin{cases}\mathcal{J}_{S}^{\theta}(\psi), & \text { if } \xi=\xi_{S}^{\theta} \\ \mathcal{J}_{p_{\star}}^{y^{\star}}(\psi), & \text { if } \xi=\xi_{p_{\star}}^{y^{\star}}\end{cases}
$$

The final objective for learning the policy network $\pi_{\psi}$, denoted as $\mathcal{J}^{\Xi}$, is the expectation of $\mathcal{J}(\psi, \xi)$ taken over the distribution of possible target specifiers $p(\xi): \mathcal{J}^{\Xi}(\psi)=\mathbb{E}_{\xi \sim p(\xi)}[\mathcal{J}(\psi, \xi)]$.

# 3.3 Training methodology for Aline

The policy and inference networks, $\pi_{\psi}$ and $q_{\phi}$, in Aline are trained jointly. Training $\pi_{\psi}$ for sequential data acquisition over a $T$-step horizon is naturally framed as a reinforcement learning (RL) problem.

Policy network training ( $\pi_{\psi}$ ). To guide the policy, we employ a dense, per-step reward signal $R_{t}$ rather than relying on a sparse reward at the end of the trajectory. This approach, common in amortized experimental design [9, 8, 37], helps stabilize and accelerate learning. The reward $R_{t}$ quantifies the immediate improvement in the inference quality provided by $q_{\phi}$ upon observing a new data point $\left(x_{t}, y_{t}\right)$, specifically concerning the current target $\xi$. It is defined based on the one-step change in the log-probabilities from our acquisition objectives (Eqs. 8 and 9):

$$
R_{t}(\xi)= \begin{cases}\frac{1}{|S|} \sum_{l \in S}\left(\log q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{t}\right)-\log q_{\phi}\left(\theta_{l} \mid \mathcal{D}_{t-1}\right)\right), & \text { if } \xi=\xi_{S}^{\theta} \\ \frac{1}{M} \sum_{m=1}^{M}\left(\log q_{\phi}\left(y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}_{t}\right)-\log q_{\phi}\left(y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}_{t-1}\right)\right), & \text { if } \xi=\xi_{p_{\star}}^{y^{\star}}\end{cases}
$$

As per common practice, for gradient stabilization we take averages (not sums) over predictions, which amounts to a constant relative rescaling of our objectives. The policy $\pi_{\psi}$ is then trained using a policy gradient (PG) algorithm with per-episode loss:

$$
\mathcal{L}_{\mathrm{PG}}(\psi)=-\sum_{t=1}^{T} \gamma^{t} R_{t}(\xi) \log \pi_{\psi}\left(x_{t} \mid \mathcal{D}_{t-1}, \xi\right)
$$

which maximizes the expected cumulative $\gamma$-discounted reward over trajectories [68]. Gradients from this policy loss only update the policy parameters $\psi$. They are not propagated back to the inference network $q_{\phi}$ to ensure each component has a clear and distinct objective.

---

#### Page 7

> **Image description.** A detailed architectural diagram illustrates the ALINE model, showing a flow of information from input sets at the bottom through embedding and transformer layers to two distinct output heads at the top. The diagram uses a combination of rounded rectangles, rectangular blocks, arrows, and mathematical notation to represent different components and data transformations.
>
> At the bottom, three distinct input sets are enclosed in dotted-line rounded rectangles:
>
> - **Context Set**: On the left, it contains a series of light teal rounded rectangles labeled `x₁`, `y₁`, followed by an ellipsis `...`, and then `x_t`, `y_t`. These pairs represent historical observations.
> - **Query Set**: In the middle, it contains light teal rounded rectangles labeled `x₁^q`, `...`, and `x_N^q`, representing candidate points.
> - **Target Set**: On the right, it presents two alternative inputs: either a single light teal rounded rectangle labeled `l_l` (where `l ∈ S` is indicated above) or a series of light teal rounded rectangles labeled `x₁^*`, `...`, `x_M^*`. This set represents the current inference goal.
>
> Above these input sets, a stacked series of three light teal rectangular blocks, slightly offset to create a layered effect, are labeled "Embedding Layers". Arrows point from each element in the input sets upwards into these embedding layers. Specific embedding functions are indicated on the left side of these layers: `f_x` for `x` inputs, `f_y` for `y` inputs, and `f_θ` for `l` inputs. The outputs of these embedding layers are shown as mathematical expressions:
>
> - From the Context Set: `f_x(x₁)` and `f_y(y₁)` are combined via a small grey circle with a `+` sign, and similarly `f_x(x_t)` and `f_y(y_t)` are combined. Arrows from these combined outputs point upwards.
> - From the Query Set: `f_x(x₁^q)` and `f_x(x_N^q)` are outputted directly, with arrows pointing upwards.
> - From the Target Set: `f_θ(l_l)`, `f_x(x₁^*)`, and `f_x(x_M^*)` are outputted directly, with arrows pointing upwards.
>
> All these outputs from the Embedding Layers converge into a large, single light purple rectangular block labeled "Transformer Layers". Multiple black arrows point from the outputs of the embedding stage into this transformer block.
>
> Finally, at the top of the diagram, two distinct output heads receive input from the Transformer Layers via black arrows:
>
> - **Acquisition Head**: A yellow rounded rectangle on the left, labeled "Acquisition Head". Above it, the output is represented by the mathematical expression `x_{t+1} ~ π_ψ(⋅|D_t)`.
> - **Inference Head**: A light blue rounded rectangle on the right, labeled "Inference Head". Above it, a light blue wavy line graphically represents a probability distribution.
>
> The overall flow is from the bottom (inputs) upwards through processing layers to the top (outputs), illustrating a neural network architecture with distinct stages for embedding, transformation, and task-specific output generation.

Figure 2: The ALINE architecture. The model takes historical observations (context set), candidate points (query set), and the current inference goal (target set) as inputs. These are transformed by embedding layers and subsequently by transformer layers. Finally, an acquisition head determines the next data point to query, while an inference head performs the approximate Bayesian inference.

Inference network training ( $q_{\phi}$ ). For the per-step rewards $R_{t}$ to be meaningful, the inference network $q_{\phi}$ must provide accurate estimates of posteriors or predictive distributions at each intermediate step $t$ of the acquisition sequence, not just at the final step $T$. Consequently, the practical training objective for $q_{\phi}$, denoted $\mathcal{L}_{\mathrm{NLL}}(\phi)$, encourages this step-wise accuracy. In practice, training proceeds in episodes. For each episode: (1) A ground truth parameter set $\theta$ is sampled from the prior $p(\theta)$. (2) A target specifier $\xi$ is sampled from $p(\xi)$. (3) If the target is predictive $\left(\xi=\xi_{p_{\star}}^{\theta^{\prime \prime}}\right), M$ target inputs $\left\{x_{m}^{\star}\right\}_{m=1}^{M}$ are sampled from $p_{\star}\left(x^{\star}\right)$, and their corresponding true outcomes $\left\{y_{m}^{\star}\right\}_{m=1}^{M}$ are simulated using $\theta$, similarly as [67]. The negative log-likelihood loss $\mathcal{L}_{\mathrm{NLL}}(\phi)$ for $q_{\phi}$ in an episode is then computed by averaging over the $T$ acquisition steps and predictions, using the Monte Carlo estimates of the objectives defined in Eqs. 5 and 6:

$$
\mathcal{L}_{\mathrm{NLL}}(\phi) \approx \begin{cases}-\frac{1}{T} \frac{1}{|S|} \sum_{t=1}^{T} \sum_{l \in S} \log q\left(\theta_{l} \mid \mathcal{D}_{t}\right), & \text { if } \xi=\xi_{S}^{\theta} \\ -\frac{1}{T} \frac{1}{M} \sum_{t=1}^{T} \sum_{m=1}^{M} \log q\left(y_{m}^{\star} \mid x_{m}^{\star}, \mathcal{D}_{t}\right), & \text { if } \xi=\xi_{p_{\star}}^{\theta^{\prime \prime}}\end{cases}
$$

Joint training. To ensure $q_{\phi}$ provides a reasonable reward signal early in training, we employ an initial warm-up phase where only $q_{\phi}$ is trained, with data acquisition $\left(x_{t}, y_{t}\right)$ guided by random actions instead of $\pi_{\psi}$. After the warm-up, $q_{\phi}$ and $\pi_{\psi}$ are trained jointly. A detailed step-by-step training algorithm is provided in Section B.1.

# 3.4 Architecture

AlINE employs a single, integrated neural architecture based on Transformer Neural Processes (TNPs) [53, 15], a network architecture that has been successfully applied to various amortized sequential decision-making settings [37, 1, 50, 39, 76]. Aline leverages TNPs to concurrently manage historical observations, propose future queries, and condition predictions on specific, potentially varying, inference objectives. An overview of Aline's architecture is provided in Figure 2.
The model inputs are structured into three sets. Following standard TNP-based architecture, the context set $\mathcal{D}_{t}=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{t}$ comprises the history of observations, and the target set $\mathcal{T}$ contains the specific target specifier $\xi$. To facilitate active data acquisition, we incorporate a query set $\mathcal{Q}=\left\{x_{a}^{q}\right\}_{n=1}^{N}$ of candidate points. In this paper, we focus on a discrete pool-based setting for consistency, though it can be straightforwardly extended to continuous design spaces (e.g., [64]). Details regarding the embeddings of these inputs are provided in Section B.2.

---

#### Page 8

> **Image description.** A grid of six line graphs, arranged in two rows and three columns, displays the predictive performance of different active learning methods. Each graph plots "RMSE ↓" (Root Mean Squared Error, indicating that lower values are better) on the vertical y-axis against "Number of Steps t" on the horizontal x-axis. All lines include a shaded area representing the 95% confidence interval. A common legend is positioned below the entire grid.
>
> The top row of graphs presents results for:
>
> - **Synthetic 1D (top-left):** The y-axis ranges from 0.0 to 0.6, and the x-axis from 0 to 30. All six methods show a decreasing trend in RMSE as the number of steps increases. The orange line with star markers ("ALINE (ours)") consistently achieves the lowest RMSE, particularly after approximately 10 steps. The gray line with circle markers ("GP-RS") shows the highest RMSE throughout.
> - **Synthetic 2D (top-middle):** The y-axis ranges from 0.0 to 0.6, and the x-axis from 0 to 50. Similar to Synthetic 1D, all methods show decreasing RMSE. "ALINE (ours)" (orange, stars) demonstrates the best performance, reaching the lowest RMSE by 50 steps. "GP-RS" (gray, circles) again performs the worst.
> - **Higdon 1D (top-right):** The y-axis ranges from 0.0 to 0.6, and the x-axis from 0 to 30. The general trend of decreasing RMSE is observed. "ALINE (ours)" (orange, stars) consistently achieves the lowest RMSE, especially from around 10 steps onwards. "GP-RS" (gray, circles) maintains the highest RMSE.
>
> The bottom row of graphs displays results for:
>
> - **Goldstein-Price 2D (bottom-left):** The y-axis ranges from 0.0 to 0.8, and the x-axis from 0 to 50. All methods show a steep initial drop in RMSE, followed by a slower decrease. "ALINE (ours)" (orange, stars) consistently achieves the lowest RMSE, while "GP-RS" (gray, circles) shows the highest.
> - **Ackley 2D (bottom-middle):** The y-axis ranges from 0.00 to 1.00, and the x-axis from 0 to 50. The RMSE for all methods decreases significantly in the initial steps. "ALINE (ours)" (orange, stars) achieves the lowest RMSE, closely followed by "ACE-US" (green, downward triangles) and "GP-EPIG" (purple, diamonds). "GP-RS" (gray, circles) again shows the highest RMSE.
> - **Gramacy 2D (bottom-right):** The y-axis ranges from 0.000 to 0.100, and the x-axis from 0 to 50. This graph shows a slightly different pattern where some methods initially increase slightly before decreasing. "ALINE (ours)" (orange, stars) maintains the lowest RMSE throughout the entire range of steps, starting lower and remaining consistently below all other methods. "GP-RS" (gray, circles) shows the highest RMSE.
>
> The common legend at the bottom identifies the six methods represented by the lines:
>
> - `GP-RS` (gray line with circular markers)
> - `GP-US` (brown line with square markers)
> - `GP-VR` (blue line with upward triangular markers)
> - `GP-EPIG` (purple line with diamond markers)
> - `ACE-US` (green line with downward triangular markers)
> - `ALINE (ours)` (orange line with star markers)
>
> Across all six benchmark functions, "ALINE (ours)" consistently demonstrates the lowest RMSE, indicating superior predictive performance compared to the other methods, particularly as the number of steps increases.

Figure 3: Predictive performance on active learning benchmark functions (RMSE ↓). Results show the mean and 95% confidence interval (CI) across 100 runs.

Standard transformer attention mechanisms process these embedded representations. Self-attention operates within the context set, capturing dependencies within $\mathcal{D}_{t}$. Both the query and target set then employ cross-attention to attend to the processed context set representations. To enable the policy $\pi_{\psi}$ to dynamically adapt its acquisition strategy based on the specific inference goal $\xi$, we introduce an additional query-target cross-attention mechanism to allow the query candidates to directly attend to the target set. This allows the evaluation of each candidate $x_{n}^{\mathrm{q}}$ to be informed by its relevance to the different potential targets $\xi$. Examples of the attention mask are shown in Figure A1.

Finally, two specialized output heads operate on these processed representations. The inference head ( $q_{\phi}$ ), following ACE [15], uses a Gaussian mixture to parameterize the approximate posteriors and posterior predictives. The acquisition head ( $\pi_{\psi}$ ) generates a policy over the query set $\pi_{\psi}\left(x_{t+1} \mid \mathcal{D}_{t}\right)$, drawing on principles from policy-based design methods [37, 50]. This unified design, which leverages a shared transformer backbone with specialized heads, significantly improves parameter efficiency by avoiding the need for separate encoders for the two tasks, leading to faster training and more efficient deployment.

# 4 Experiments

We now empirically evaluate Aline's performance in different active data acquisition and amortized inference tasks. We begin with the active learning task in Section 4.1, where we want to efficiently minimize the uncertainty over an unknown function by querying $T$ data points. Then, we test Aline's policy on standard BED benchmarks in Section 4.2. In Section 4.3, we demonstrate the benefit of Aline's flexible targeting feature in a psychometric modeling task [72]. Finally, to demonstrate the scalability of Aline, we test it on a high-dimensional task of actively exploring hyperparameter performance landscapes; the results are presented in Section D.1. The code to reproduce our experiments is available at: https://github.com/huangdaolang/aline.

### 4.1 Active learning for regression and hyperparameter inference

For the active learning task, Aline is trained on a diverse collection of fully synthetic functions drawn from Gaussian Process (GP) [62] priors (see Section C.1.1 for details). We evaluate Aline's performance under both in-distribution and out-of-distribution settings. For the in-distribution setting, Aline is evaluated on synthetic functions sampled from the same GP prior that is used during training. In the out-of-distribution setting, we evaluate Aline on benchmark functions (Higdon, Goldstein-Price, Ackley, and Gramacy) unseen during training, to assess generalization beyond the training regime. We compare Aline against non-amortized GP models equipped with standard acquisition functions such as Uncertainty Sampling (GP-US), Variance Reduction (GP-VR) [74], EPIG (GP-EPIG) [67], and Random Sampling (GP-RS). Additionally, we include an amortized neural

---

#### Page 9

Table 2: Results on BED benchmarks. For the EIG lower bound, we report the mean $\pm 95\%$ CI across 2000 runs (200 for VPCE). For deployment time, we use the mean $\pm 95\%$ CI from 20 runs.

|              | Location Finding                  |                        |                          | Constant Elasticity of Substitution |                        |                          |
| ------------ | --------------------------------- | ---------------------- | ------------------------ | ----------------------------------- | ---------------------- | ------------------------ |
|              | EIG lower <br> bound $(\uparrow)$ | Training <br> time (h) | Deployment <br> time (s) | EIG lower <br> bound $(\uparrow)$   | Training <br> time (h) | Deployment <br> time (s) |
| Random       | $5.17 \pm 0.05$                   | N/A                    | N/A                      | $9.05 \pm 0.26$                     | N/A                    | N/A                      |
| VPCE [25]    | $5.25 \pm 0.22$                   | N/A                    | $146.59 \pm 0.09$        | $9.40 \pm 0.27$                     | N/A                    | $788.90 \pm 1.03$        |
| DAD [23]     | $7.33 \pm 0.06$                   | 7.24                   | $0.0001 \pm 0.00$        | $10.77 \pm 0.15$                    | 13.70                  | $0.0001 \pm 0.00$        |
| vsOED [65]   | $7.30 \pm 0.06$                   | 4.31                   | $0.0002 \pm 0.00$        | $12.12 \pm 0.18$                    | 0.49                   | $0.0003 \pm 0.00$        |
| RL-BOED [9]  | $7.70 \pm 0.06$                   | 63.29                  | $0.0003 \pm 0.00$        | $14.60 \pm 0.10$                    | 67.28                  | $0.0004 \pm 0.00$        |
| Aline (ours) | $8.91 \pm 0.04$                   | 21.20                  | $0.03 \pm 0.00$          | $13.50 \pm 0.15$                    | 13.29                  | $0.04 \pm 0.00$          |

process baseline, the Amortized Conditioning Engine (ACE) [15], paired with Uncertainty Sampling (ACE-US), to specifically evaluate the advantage of Aline's learned acquisition policy over using a standard acquisition function with an amortized inference model. The performance metric is the root mean squared error (RMSE) of the predictions on a held-out test set.

Results for the active learning task in Figure 3 show that Aline performs comparably to the best-performing GP-based methods for the in-distribution setting. Importantly, for the out-of-distribution setting, Aline outperforms the baselines in 3 out of the 4 benchmark functions. These results highlight the advantage of Aline's end-to-end learning strategy, which obviates the need for kernel specification using GPs or explicit acquisition function selection. Further evaluations on additional benchmark functions (Gramacy 1D, Branin, Three Hump Camel), visualizations of Aline's sequential querying strategy with corresponding predictive updates, and a comparison of average inference times are provided in Section D.2.

> **Image description.** A 2D line graph titled "Log Probability $\uparrow$" on the y-axis and "Number of Steps $t$" on the x-axis, illustrating the performance of three different methods over a series of steps.
>
> The x-axis, labeled "Number of Steps $t$", ranges from 0 to 30, with major tick marks at intervals of 5. The y-axis, labeled "Log Probability $\uparrow$", ranges from -0.6 to 0.2, with major tick marks at intervals of 0.2. The upward arrow next to "Log Probability" suggests that higher values are better.
>
> Three distinct lines are plotted, each representing a different method, accompanied by a shaded region indicating a confidence interval:
>
> - **ACE-RS**: Represented by a gray line with upward-pointing triangle markers. The line starts around -0.6 at the first step and gradually increases to approximately -0.1 at 30 steps. A light gray shaded area surrounds this line.
> - **ACE-US**: Represented by a green line with 'X' markers. This line also starts around -0.6 and increases to about -0.1 at 30 steps. It generally runs slightly above the ACE-RS line for most of its length. A light green shaded area surrounds this line.
> - **ALINE (ours)**: Represented by an orange line with star markers. This line consistently shows the highest log probability among the three methods. It starts around -0.6 and rises more steeply, reaching approximately 0.1 at 30 steps. A light orange shaded area surrounds this line.
>
> All three lines show an increasing trend in Log Probability as the Number of Steps increases. ALINE (ours) demonstrates superior performance, maintaining a higher log probability compared to ACE-US and ACE-RS throughout the observed range of steps. The legend, located in the upper central part of the plot, clearly identifies each line with its corresponding method name and marker type.

Figure 4: Hyperparameter inference performance on synthetic GP functions.

Additionally, for the in-distribution setting, we test Aline's capability to infer the underlying GP's hyperparameters-without retraining, by leveraging Aline's flexible target specification at runtime. For baselines, we use ACE-US and ACE-RS, since ACE [15] is also capable of posterior estimation. Figure 4 shows that Aline yields higher log probabilities of the true parameter value under the estimated posterior at each step compared to the baselines. This is due to the ability to flexibly switch Aline's acquisition strategy to parameter inference, unlike other active learning methods. We also visualize the obtained posteriors in Section D.2.

# 4.2 Benchmarking on Bayesian experimental design tasks

We test Aline on two classical BED tasks: Location Finding [66] and Constant Elasticity of Substitution (CES) [3], with two- and six-dimensional design space, respectively. As baselines, we include a random design policy, a gradient-based method with variational Prior Contrastive Estimation (VPCE) [25], and three amortized BED methods: Deep Adaptive Design (DAD) [23], vsOED [65], and RL-BOED [9]. Details of the tasks and the baselines are provided in Section C.2.

To evaluate performance, we compute a lower bound of the total EIG, namely the sequential Prior Contrastive Estimation lower bound [23]. As shown in Table 2, Aline surpasses all the baselines in the Location Finding task and achieves competitive performance on the CES task, outperforming most other methods, except RL-BOED. Notably, Aline's training time is reduced as its reward is based on the internal posterior improvement and does not require a large number of contrastive samples to estimate sEIG. While Aline's deployment time is slightly higher than MLP-based amortized methods due to the computational cost of its transformer architecture, it remains orders of magnitude faster than non-amortized approaches like VPCE. Visualizations of Aline's inferred posterior distributions are provided in Section D.3.

### 4.3 Psychometric model

Our final experiment involves the psychometric modeling task [72]-a fundamental scenario in behavioral sciences, from neuroscience to clinical settings [29, 57, 73], where the goal is to infer parameters governing an observer's responses to varying stimulus intensities. The psychometric

---

#### Page 10

> **Image description.** The image displays a 2x2 grid of four scientific plots, labeled (a), (b), (c), and (d), presenting results related to psychometric models and query strategies. The top row of plots is titled "Threshold & Slope", while the bottom row is titled "Guess Rate & Lapse Rate".
>
> **Top Row: Threshold & Slope**
>
> - **Panel (a): RMSE for Threshold & Slope**
>   This is a line graph showing Root Mean Square Error (RMSE) as a function of the number of steps.
>
>   - The x-axis is labeled "Number of Steps t" and ranges from 0 to 30, with major ticks at 10, 20, and 30.
>   - The y-axis is labeled "RMSE ↓" (indicating lower values are better) and ranges from 0.5 to 1.5, with major ticks at 0.5, 1.0, and 1.5.
>   - A legend in the top right identifies three methods:
>     - "QUEST+" is represented by a blue dashed line with square markers.
>     - "Psi-marginal" is represented by a green dash-dot line with upward-pointing triangle markers.
>     - "ALINE (ours)" is represented by an orange solid line with star markers.
>   - All three lines start near RMSE 1.5 at t=0 and rapidly decrease, then flatten out, converging to values between 0.5 and 0.6 at t=30. The lines are very close, with ALINE and Psi-marginal slightly outperforming QUEST+ in the later steps, but the overall performance is similar.
>
> - **Panel (b): Stimuli Values for Threshold & Slope**
>   This is a scatter plot illustrating stimuli values over the number of steps, representing a query strategy.
>   - The x-axis is labeled "Number of Steps t" and ranges from 0 to 30, with major ticks at 10, 20, and 30.
>   - The y-axis is labeled "Stimuli Values" and ranges from -5 to 5, with major ticks at -5, 0, and 5.
>   - A legend in the top right identifies:
>     - "True threshold" as a horizontal grey dashed line. This line is positioned at approximately Y = -0.5.
>     - "Query points" as orange circular markers.
>   - The query points initially show a wider spread (e.g., from Y=3 to Y=-3) for lower 't' values. As 't' increases, these points converge towards the "True threshold" line, becoming tightly clustered around Y = -0.5 by t=30. The color of the query points gradually darkens from light orange to dark orange as 't' increases.
>
> **Bottom Row: Guess Rate & Lapse Rate**
>
> - **Panel (c): RMSE for Guess Rate & Lapse Rate**
>   This is a line graph showing RMSE as a function of the number of steps, similar in type to panel (a).
>
>   - The x-axis is labeled "Number of Steps t" and ranges from 0 to 30, with major ticks at 10, 20, and 30.
>   - The y-axis is labeled "RMSE ↓" and ranges from 0.15 to 0.30, with major ticks at 0.20, 0.25, and 0.30.
>   - The legend is identical to panel (a), showing "QUEST+" (blue dashed line, squares), "Psi-marginal" (green dash-dot line, triangles), and "ALINE (ours)" (orange solid line, stars).
>   - "QUEST+" starts around RMSE 0.28 and decreases slowly to approximately 0.22 at t=30.
>   - "Psi-marginal" and "ALINE (ours)" also start around RMSE 0.28 but decrease much more steeply and consistently, reaching values below 0.15 by t=30. These two lines are very close to each other and significantly outperform QUEST+.
>
> - **Panel (d): Stimuli Values for Guess Rate & Lapse Rate**
>   This is a scatter plot illustrating stimuli values over the number of steps, similar in type to panel (b).
>   - The x-axis is labeled "Number of Steps t" and ranges from 0 to 30, with major ticks at 10, 20, and 30.
>   - The y-axis is labeled "Stimuli Values" and ranges from -5 to 5, with major ticks at -5, 0, and 5.
>   - The legend is identical to panel (b), showing "True threshold" as a horizontal grey dashed line and "Query points" as orange circular markers. The "True threshold" line is positioned at approximately Y = -2.5.
>   - The query points form two distinct clusters. One cluster is above the "True threshold" line, starting around Y=4 and converging towards Y=3.5. The other cluster is below the "True threshold" line, starting around Y=-4.5 and converging towards Y=-4.5. Unlike panel (b), these query points do not converge to the "True threshold" line but rather maintain two separate bands of values. The color of the query points gradually darkens from light orange to dark orange as 't' increases.

Figure 5: Results on psychometric model. RMSE (mean $\pm 95 \%$ CI) when targeting (a) threshold \& slope and (c) guess \& lapse rates, with ALINE's corresponding query strategies shown in (b) and (d).

function used here is characterized by four parameters: threshold, slope, guess rate, and lapse rate; see Section C.3.1 for details. Different research questions in psychophysics necessitate focusing on different parameter subsets. For instance, studies on perceptual sensitivity primarily target precise estimation of threshold and slope, while investigations into response biases or attentional phenomena might focus on the guess and lapse rates. This is where Aline's unique flexible querying strategy can be used to target specific parameter subsets of interest.
We compare Aline with two established adaptive psychophysical methods: QUEST+ [70], which targets all parameters simultaneously, and Psi-marginal [58], which can marginalize over nuisance parameters to focus on a specified subset, a non-amortized gold-standard method for flexible acquisition. We evaluate scenarios targeting either the threshold and slope parameters or the guess and lapse rates. Details of the baselines and the experimental setup are in Section C.3.2.
Figure 5 shows the results. When targeting threshold and slope (Figure 5a), which are generally easier to infer, Aline achieves results comparable to baselines. When targeting guess and lapse rates (Figure 5c), QUEST+ performs sub-optimally as its experimental design strategy is dominated by the more readily estimable threshold and slope parameters. In contrast, both Psi-marginal and Aline lead to significantly better inference than QUEST+ when explicitly targeting guess and lapse rates. Moreover, Aline offers a $10 \times$ speedup over these non-amortized methods (See Section D). We also visualize the query strategies adopted by Aline in the two cases: when targeting threshold and slope (Figure 5b), stimuli are concentrated near the estimated threshold. Conversely, when targeting guess and lapse rates (Figure 5d), Aline appropriately selects 'easy' stimuli at extreme values where mistakes can be more readily attributed to random behavior (governed by lapses and guesses) rather than the discriminative ability of the subject (governed by threshold and slope). To further demonstrate Aline's runtime flexibility, we conduct two additional investigations detailed in Section D.4. First, we show that a single pre-trained ALINE model can dynamically switch its acquisition target midexperiment. Second, we validate its ability to generalize to novel combinations of targets that were not seen during training, showing the effectiveness of the query-target cross-attention mechanism.

# 5 Conclusion

We introduced Aline, a unified amortized framework that seamlessly integrates active data acquisition with Bayesian inference. Aline dynamically adapts its strategy to target selected inference goals, offering a flexible and efficient solution for Bayesian inference and active data acquisition.

Limitations \& future work. Currently, Aline operates with pre-defined, fixed priors, necessitating re-training for different prior specifications. Future work could explore prior amortization [15, 71], to allow for dynamic prior conditioning. As Aline estimates marginal posteriors, extending this to joint posterior estimation, potentially via autoregressive modeling [11, 35], is a promising direction. Note that, at deployment, we may encounter observations that differ substantially from the training data, leading to degradation in performance. This issue can potentially be tackled by combining Aline with robust approaches such as [22, 36]. Lastly, Aline's current architecture is tailored to fixed input dimensionalities and discrete design spaces, a common practice with TNPs [53, 50, 76, 37]. Generalizing Aline to be dimension-agnostic [45] and to support continuous experimental designs [64] are valuable avenues for future research.

---

#### Page 11

# Acknowledgements

DH, LA and SK were supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI, 359207). The authors acknowledge the research environment provided by ELLIS Institute Finland. LA was also supported by Research Council of Finland grants 358980 and 356498. SK was also supported by the UKRI Turing AI World-Leading Researcher Fellowship, [EP/W002973/1]. AB was supported by the Research Council of Finland grant no. 362534. The authors wish to thank Aalto Science-IT project, and CSC-IT Center for Science, Finland, for the computational and data storage resources provided.

## References

[1] Andersson, T. R., Bruinsma, W. P., Markou, S., Requeima, J., Coca-Castro, A., Vaughan, A., Ellis, A.-L., Lazzara, M. A., Jones, D., Hosking, S., et al. (2023). Environmental sensor placement with convolutional gaussian neural processes. Environmental Data Science, 2:e32. 7
[2] Arango, S. P., Jomaa, H. S., Wistuba, M., and Grabocka, J. (2021). Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 24
[3] Arrow, K. J., Chenery, H. B., Minhas, B. S., and Solow, R. M. (1961). Capital-labor substitution and economic efficiency. The Review of Economics and Statistics, 43(3):225-250. 9, 21
[4] Ashman, M., Diaconu, C., Kim, J., Sivaraya, L., Markou, S., Requeima, J., Bruinsma, W. P., and Turner, R. E. (2024a). Translation equivariant transformer neural processes. In International Conference on Machine Learning, pages 1924-1944. PMLR. 3
[5] Ashman, M., Diaconu, C., Weller, A., and Turner, R. E. (2024b). In-context in-context learning with transformer neural processes. In Symposium on Advances in Approximate Bayesian Inference, pages 1-29. PMLR. 3
[6] Barlas, Y. Z. and Salako, K. (2025). Performance comparisons of reinforcement learning algorithms for sequential experimental design. arXiv preprint arXiv:2503.05905. 27
[7] Berry, S. M., Carlin, B. P., Lee, J. J., and Muller, P. (2010). Bayesian adaptive methods for clinical trials. CRC press. 2
[8] Blau, T., Bonilla, E., Chades, I., and Dezfouli, A. (2023). Cross-entropy estimators for sequential experiment design with reinforcement learning. arXiv preprint arXiv:2305.18435. 2, 4, 6
[9] Blau, T., Bonilla, E. V., Chades, I., and Dezfouli, A. (2022). Optimizing sequential experimental design with deep reinforcement learning. In International conference on machine learning, pages 2107-2128. PMLR. 2, 3, 4, 6, 9, 21, 22, 23
[10] Broderick, T., Boyd, N., Wibisono, A., Wilson, A. C., and Jordan, M. I. (2013). Streaming variational bayes. Advances in neural information processing systems, 26. 1
[11] Bruinsma, W., Markou, S., Requeima, J., Foong, A. Y., Andersson, T., Vaughan, A., Buonomo, A., Hosking, S., and Turner, R. E. (2023). Autoregressive conditional neural processes. In The Eleventh International Conference on Learning Representations. 3, 5, 10
[12] Bruinsma, W., Requeima, J., Foong, A. Y., Gordon, J., and Turner, R. E. (2020). The gaussian neural process. In Third Symposium on Advances in Approximate Bayesian Inference. 3
[13] Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., and Riddell, A. (2017). Stan: A probabilistic programming language. Journal of statistical software, 76:1-32. 1
[14] Chaloner, K. and Verdinelli, I. (1995). Bayesian experimental design: A review. Statistical science, pages 273-304. 3
[15] Chang, P. E., Loka, N., Huang, D., Remes, U., Kaski, S., and Acerbi, L. (2025). Amortized probabilistic conditioning for optimization, simulation and inference. In International Conference on Artificial Intelligence and Statistics. PMLR. 2, 3, 4, 5, 7, 8, 9, 10, 21

---

#### Page 12

[16] Chen, X., Wang, C., Zhou, Z., and Ross, K. W. (2021). Randomized ensembled double qlearning: Learning fast without a model. In International Conference on Learning Representations. 23
[17] Cranmer, K., Brehmer, J., and Louppe, G. (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055-30062. 4
[18] Doucet, A., De Freitas, N., Gordon, N. J., et al. (2001). Sequential Monte Carlo methods in practice, volume 1. Springer. 1
[19] Feng, L., Hajimirsadeghi, H., Bengio, Y., and Ahmed, M. O. (2023). Latent bottlenecked attentive neural processes. In The Eleventh International Conference on Learning Representations. 3
[20] Feng, L., Tung, F., Hajimirsadeghi, H., Bengio, Y., and Ahmed, M. O. (2024). Memory efficient neural processes via constant memory attention block. In International Conference on Machine Learning, pages 13365-13386. PMLR. 3
[21] Filstroff, L., Sundin, I., Mikkola, P., Tiulpin, A., Kylmäoja, J., and Kaski, S. (2024). Targeted active learning for bayesian decision-making. Transactions on Machine Learning Research. 1
[22] Forster, A., Ivanova, D. R., and Rainforth, T. (2025). Improving robustness to model misspecification in bayesian experimental design. In 7th Symposium on Advances in Approximate Bayesian Inference Workshop Track. 10
[23] Foster, A., Ivanova, D. R., Malik, I., and Rainforth, T. (2021). Deep adaptive design: Amortizing sequential bayesian experimental design. In International Conference on Machine Learning, pages 3384-3395. PMLR. 2, 3, 4, 5, 9, 16, 22, 27
[24] Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth, T., and Goodman, N. (2019). Variational bayesian optimal experimental design. Advances in Neural Information Processing Systems, 32. 2, 6, 21
[25] Foster, A., Jankowiak, M., O’Meara, M., Teh, Y. W., and Rainforth, T. (2020). A unified stochastic gradient approach to designing bayesian-optimal experiments. In International Conference on Artificial Intelligence and Statistics, pages 2959-2969. PMLR. 9, 22
[26] Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., and Eslami, S. A. (2018a). Conditional neural processes. In International conference on machine learning, pages 1704-1713. PMLR. 2, 3, 5
[27] Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D. J., Eslami, S., and Teh, Y. W. (2018b). Neural processes. arXiv preprint arXiv:1807.01622. 2, 3
[28] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). Bayesian Data Analysis. CRC Press. 1, 3
[29] Gilaie-Dotan, S., Kanai, R., Bahrami, B., Rees, G., and Saygin, A. P. (2013). Neuroanatomical correlates of biological motion detection. Neuropsychologia, 51(3):457-463. 9
[30] Giovagnoli, A. (2021). The bayesian design of adaptive clinical trials. International journal of environmental research and public health, 18(2):530. 1
[31] Gloeckler, M., Deistler, M., Weilbach, C. D., Wood, F., and Macke, J. H. (2024). All-in-one simulation-based inference. In International Conference on Machine Learning, pages 1573515766. PMLR. 2, 3, 4
[32] Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In Teh, Y. W. and Titterington, M., editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 249-256, Chia Laguna Resort, Sardinia, Italy. PMLR. 22
[33] Gordon, J., Bruinsma, W. P., Foong, A. Y., Requeima, J., Dubois, Y., and Turner, R. E. (2020). Convolutional conditional neural processes. In International Conference on Learning Representations. 3

---

#### Page 13

[34] Greenberg, D., Nonnenmacher, M., and Macke, J. (2019). Automatic posterior transformation for likelihood-free inference. In International conference on machine learning, pages 2404-2414. PMLR. 2, 3, 4
[35] Hassan, C., Loka, N., Li, C.-Y., Huang, D., Chang, P. E., Yang, Y., Silvestrin, F., Kaski, S., and Acerbi, L. (2025). Efficient autoregressive inference for transformer probabilistic models. arXiv preprint arXiv:2510.09477. 10
[36] Huang, D., Bharti, A., Souza, A., Acerbi, L., and Kaski, S. (2023a). Learning robust statistics for simulation-based inference under model misspecification. Advances in Neural Information Processing Systems, 36:7289-7310. 10
[37] Huang, D., Guo, Y., Acerbi, L., and Kaski, S. (2025). Amortized bayesian experimental design for decision-making. Advances in Neural Information Processing Systems, 37:109460-109486. 6, $7,8,10$
[38] Huang, D., Haussmann, M., Remes, U., John, S., Clarté, G., Luck, K., Kaski, S., and Acerbi, L. (2023b). Practical equivariances via relational conditional neural processes. Advances in Neural Information Processing Systems, 36:29201-29238. 3
[39] Hung, Y. H., Lin, K.-J., Lin, Y.-H., Wang, C.-Y., Sun, C., and Hsieh, P.-C. (2025). Boformer: Learning to solve multi-objective bayesian optimization via non-markovian rl. In The Thirteenth International Conference on Learning Representations. 7
[40] Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M. U., and Rainforth, T. (2021). Implicit deep adaptive design: Policy-based experimental design without likelihoods. Advances in Neural Information Processing Systems, 34. 2, 3, 4, 21
[41] Ivanova, D. R., Hedman, M., Guan, C., and Rainforth, T. (2024). Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design. ICLR 2024 Workshop on Data-centric Machine Learning Research (DMLR). 2, 21
[42] Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami, A., Rosenbaum, D., Vinyals, O., and Teh, Y. W. (2019). Attentive neural processes. In International Conference on Learning Representations. 3
[43] Kontsevich, L. L. and Tyler, C. W. (1999). Bayesian adaptive estimation of psychometric slope and threshold. Vision research, 39(16):2729-2737. 24
[44] Krause, A., Guestrin, C., Gupta, A., and Kleinberg, J. (2006). Near-optimal sensor placements: Maximizing information while minimizing communication cost. In Proceedings of the 5th international conference on Information processing in sensor networks, pages 2-10. 2
[45] Lee, H., Jang, C., Lee, D. B., and Lee, J. (2025). Dimension agnostic neural processes. In The Thirteenth International Conference on Learning Representations. 10
[46] Li, C.-Y., Toussaint, M., Rakitsch, B., and Zimmer, C. (2025). Amortized safe active learning for real-time decision-making: Pretrained neural policies from simulated nonparametric functions. arXiv preprint arXiv:2501.15458. 2, 4
[47] Lindley, D. V. (1956). On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, 27(4):986-1005. 3
[48] Lookman, T., Balachandran, P. V., Xue, D., and Yuan, R. (2019). Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. $n p j$ Computational Materials, 5(1):21. 2
[49] Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Öcal, K., Nonnenmacher, M., and Macke, J. H. (2017). Flexible statistical inference for mechanistic models of neural dynamics. Advances in neural information processing systems, 30. 2, 3, 4
[50] Maraval, A., Zimmer, M., Grosnit, A., and Bou Ammar, H. (2024). End-to-end meta-bayesian optimisation with transformer neural processes. Advances in Neural Information Processing Systems, 36. 7, 8, 10

---

#### Page 14

[51] Mittal, S., Bracher, N. L., Lajoie, G., Jaini, P., and Brubaker, M. (2025). Amortized in-context bayesian posterior estimation. arXiv preprint arXiv:2502.06601. 4
[52] Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., and Hutter, F. (2022). Transformers can do bayesian inference. In International Conference on Learning Representations. 2, 3, 5
[53] Nguyen, T. and Grover, A. (2022). Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In International Conference on Machine Learning, pages 1656916594. PMLR. 2, 3, 5, 7, 10
[54] Papamakarios, G. and Murray, I. (2016). Fast $\varepsilon$-free inference of simulation models with bayesian conditional density estimation. Advances in neural information processing systems, 29. $2,3,4$
[55] Pasek, J. and Krosnick, J. A. (2010). Optimizing survey questionnaire design in political science: Insights from psychology. 1
[56] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011). Scikit-learn: Machine learning in python. the Journal of machine Learning research, 12:2825-2830. 21, 27
[57] Powers, A. R., Mathys, C., and Corlett, P. R. (2017). Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors. Science, 357(6351):596-600. 1, 9
[58] Prins, N. (2013). The psi-marginal adaptive method: How to give nuisance parameters the attention they deserve (no more, no less). Journal of vision, 13(7):3-3. 2, 10, 24
[59] Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., and Bürkner, P.-C. (2023a). Jana: Jointly amortized neural approximation of complex bayesian models. In Uncertainty in Artificial Intelligence, pages 1695-1706. PMLR. 2, 3, 4
[60] Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., and Bürkner, P.-C. (2023b). Bayesflow: Amortized bayesian workflows with neural networks. Journal of Open Source Software, 8(89):5702. 2, 3, 4
[61] Rainforth, T., Foster, A., Ivanova, D. R., and Bickford Smith, F. (2024). Modern bayesian experimental design. Statistical Science, 39(1):100-114. 1, 3
[62] Rasmussen, C. E. and Williams, C. K. (2006). Gaussian Processes for Machine Learning. MIT Press. 8
[63] Ryan, E. G., Drovandi, C. C., McGree, J. M., and Pettitt, A. N. (2016). A review of modern computational algorithms for bayesian optimal design. International Statistical Review, 84(1):128154. 3
[64] Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning, pages 1889-1897. PMLR. 7, 10
[65] Shen, W., Dong, J., and Huan, X. (2025). Variational sequential optimal experimental design using reinforcement learning. Computer Methods in Applied Mechanics and Engineering, 444:118068. 2, 4, 6, 9, 22, 27
[66] Sheng, X. and Hu, Y.-H. (2004). Maximum likelihood multiple-source localization using acoustic energy measurements with wireless sensor networks. IEEE transactions on signal processing, 53(1):44-53. 9, 21
[67] Smith, F. B., Kirsch, A., Farquhar, S., Gal, Y., Foster, A., and Rainforth, T. (2023). Predictionoriented bayesian active learning. In International Conference on Artificial Intelligence and Statistics, pages 7331-7348. PMLR. 3, 4, 5, 7, 8, 17, 20
[68] Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12. 6

---

#### Page 15

[69] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.2
[70] Watson, A. B. (2017). Quest+: A general multidimensional bayesian adaptive psychometric method. Journal of Vision, 17(3):10-10. 1, 10, 24
[71] Whittle, G., Ziomek, J., Rawling, J., and Osborne, M. A. (2025). Distribution transformers: Fast approximate bayesian inference with on-the-fly prior adaptation. arXiv preprint arXiv:2502.02463. 10
[72] Wichmann, F. A. and Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. Perception \& Psychophysics, 63(8):1293-1313. 8, 9
[73] Xu, C., Hülsmeier, D., Buhl, M., and Kollmeier, B. (2024). How does inattention influence the robustness and efficiency of adaptive procedures in the context of psychoacoustic assessments via smartphone? Trends in Hearing, 28:23312165241288051. 9
[74] Yu, K., Bi, J., and Tresp, V. (2006). Active learning via transductive experimental design. In Proceedings of the 23rd international conference on Machine learning, pages 1081-1088. 8, 20
[75] Zammit-Mangion, A., Sainsbury-Dale, M., and Huser, R. (2024). Neural methods for amortised parameter inference. arXiv e-prints, pages arXiv-2404. 2, 4
[76] Zhang, X., Huang, D., Kaski, S., and Martinelli, J. (2025). Pabbo: Preferential amortized black-box optimization. In The Thirteenth International Conference on Learning Representations. 7,10

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
> `(x_1, y_1)` > `:` > `(x_t, y_t)` > `x_1^q` > `:` > `x_N^q` > `x_1^*` > `:` > `x_M^*`
>
> The column labels, from left to right, are:
> `(x_1, y_1)` > `...` > `(x_t, y_t)` > `x_1^q` > `...` > `x_N^q` > `x_1^*` > `...` > `x_M^*`
>
> The shading pattern indicates allowed attention:
>
> - **Rows `(x_1, y_1)` through `(x_t, y_t)` (first 3 visible rows):** All squares in these rows are shaded purple, indicating full attention to all columns.
> - **Rows `x_1^q` through `x_N^q` (middle 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^q` through `x_N^q`. The squares corresponding to `x_1^*` through `x_M^*` columns are unshaded white.
> - **Rows `x_1^*` through `x_M^*` (last 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^*` through `x_M^*`. The squares corresponding to `x_1^q` through `x_N^q` columns are unshaded white.
>
> **Panel (b):**
> This panel presents a 7x7 grid of squares, similar to panel (a) with '...' indicating omitted elements.
> The row labels, from top to bottom, are:
> `(x_1, y_1)` > `:` > `(x_t, y_t)` > `x_1^q` > `:` > `x_N^q` > `l_1`
>
> The column labels, from left to right, are:
> `(x_1, y_1)` > `...` > `(x_t, y_t)` > `x_1^q` > `...` > `x_N^q` > `l_1`
>
> The shading pattern indicates allowed attention:
>
> - **Rows `(x_1, y_1)` through `(x_t, y_t)` (first 3 visible rows):** All squares in these rows are shaded purple, indicating full attention to all columns.
> - **Rows `x_1^q` through `x_N^q` (middle 3 visible rows):** Squares in these rows are shaded purple for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)` and for columns corresponding to `x_1^q` through `x_N^q`. The square corresponding to the `l_1` column is unshaded white.
> - **Row `l_1` (last visible row):** Squares in this row are shaded purple only for columns corresponding to `(x_1, y_1)` through `(x_t, y_t)`. The squares corresponding to `x_1^q` through `x_N^q` and `l_1` columns are unshaded white.

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
>
> - **X-axis:** The horizontal axis ranges from -5.0 to 5.0, with major tick marks and numerical labels at -5.0, -2.5, 0.0, 2.5, and 5.0.
> - **Y-axis:** The vertical axis ranges from -2 to 2, with major tick marks and numerical labels at -2, 0, and 2.
> - **Plot Line:** A single, solid blue line represents the function in each subplot.
>
> Above each subplot, a unique title is provided, following the format "Sample X: ls = Y, scale = Z". The values for 'ls' and 'scale' vary across the samples, visually correlating with the characteristics of the plotted functions. Generally, lower 'ls' values correspond to more erratic and jagged functions with frequent oscillations, while higher 'ls' values indicate smoother functions with broader curves or gentler slopes. Similarly, higher 'scale' values tend to correspond to functions with larger amplitudes, reaching closer to the y-axis limits, whereas lower 'scale' values indicate functions with smaller amplitudes, staying closer to the x-axis.
>
> The specific titles and visual characteristics of each sample are as follows:
>
> **Row 1:**
>
> - **Sample 1:** "Sample 1: ls = 1.20, scale = 0.33". A relatively smooth curve with gentle undulations, mostly staying between -1 and 1.
> - **Sample 2:** "Sample 2: ls = 1.51, scale = 0.29". A very smooth, gently sloping curve, starting slightly above 0 and gradually decreasing, then slightly increasing, remaining close to the x-axis.
> - **Sample 3:** "Sample 3: ls = 1.92, scale = 0.69". A smooth, wave-like curve with one prominent peak and trough, oscillating between approximately -1.5 and 1.5.
> - **Sample 4:** "Sample 4: ls = 0.38, scale = 0.81". A highly erratic and jagged function with numerous rapid fluctuations, covering nearly the full y-range from -2 to 2.
>
> **Row 2:**
>
> - **Sample 5:** "Sample 5: ls = 0.34, scale = 0.18". An erratic function with small amplitude oscillations, mostly staying between -0.5 and 0.5.
> - **Sample 6:** "Sample 6: ls = 1.01, scale = 0.68". A smooth, S-shaped curve with a significant dip below -1 and a rise above 1.
> - **Sample 7:** "Sample 7: ls = 0.86, scale = 0.99". A complex, moderately erratic function with several distinct peaks and troughs, reaching close to the y-axis limits.
> - **Sample 8:** "Sample 8: ls = 0.65, scale = 0.44". A moderately smooth, wavy function with several oscillations, mostly between -1 and 1.
>
> **Row 3:**
>
> - **Sample 9:** "Sample 9: ls = 0.78, scale = 0.74". A smooth, multi-peak and trough curve, resembling a sine wave with varying amplitudes, mostly between -1.5 and 1.5.
> - **Sample 10:** "Sample 10: ls = 1.69, scale = 0.70". A relatively smooth function with gentle, broad undulations, mostly between -1 and 1.
> - **Sample 11:** "Sample 11: ls = 1.95, scale = 0.53". A very smooth, almost flat line that gently curves downwards from left to right, staying close to the x-axis.
> - **Sample 12:** "Sample 12: ls = 0.51, scale = 0.89". A highly periodic and smooth wave-like pattern with multiple prominent peaks and troughs, resembling a sine wave with large amplitude, reaching close to the y-axis limits.
>
> **Row 4:**
>
> - **Sample 13:** "Sample 13: ls = 0.61, scale = 0.34". A moderately erratic function with several small oscillations, mostly between -1 and 1.
> - **Sample 14:** "Sample 14: ls = 0.54, scale = 0.51". A moderately erratic function with more pronounced peaks and troughs than Sample 13, covering a wider y-range.
> - **Sample 15:** "Sample 15: ls = 0.88, scale = 0.41". A relatively smooth, gently undulating function, mostly staying between -1 and 1.
> - **Sample 16:** "Sample 16: ls = 0.24, scale = 0.46". A highly erratic and jagged function with frequent, rapid fluctuations, similar to Sample 4 but with slightly smaller amplitude.
>
> **Row 5:**
>
> - **Sample 17:** "Sample 17: ls = 1.02, scale = 0.17". A very smooth, almost flat line with minimal undulations, staying very close to the x-axis, mostly between -0.5 and 0.5.
> - **Sample 18:** "Sample 18: ls = 1.19, scale = 0.54". A smooth, wave-like curve with a prominent trough and peak, resembling a single cycle of a wave, mostly between -1.5 and 1.
> - **Sample 19:** "Sample 19: ls = 0.16, scale = 0.72". A highly erratic and jagged function with very frequent and large fluctuations, covering the full y-range from -2 to 2, appearing to be the most erratic among all samples.
> - **Sample 20:** "Sample 20: ls = 1.62, scale = 0.66". A relatively smooth curve, starting slightly above 0, dipping below 0, and then rising again, resembling a gentle 'U' shape.

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

| Parameter               | Location Finding |       CES |
| :---------------------- | :--------------: | --------: |
| VI gradient steps       |       1000       |      1000 |
| VI learning rate        |    $10^{-3}$     | $10^{-3}$ |
| Design gradient steps   |       2500       |      2500 |
| Design learning rate    |    $10^{-3}$     | $10^{-3}$ |
| Contrastive samples $L$ |       500        |        10 |
| Expectation samples     |       500        |        10 |

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

| Parameter                |  Location Finding |               CES |
| :----------------------- | ----------------: | ----------------: |
| Critics                  |                 2 |                 2 |
| Random subsets           |                 2 |                 2 |
| Contrastive samples $L$  |          $10^{5}$ |          $10^{5}$ |
| Training epochs          |          $10^{5}$ |  $5 \cdot 10^{4}$ |
| Discount factor $\gamma$ |               0.9 |               0.9 |
| Target update rate       |         $10^{-3}$ | $5 \cdot 10^{-3}$ |
| Policy learning rate     |         $10^{-4}$ | $3 \cdot 10^{-4}$ |
| Critic learning rate     | $3 \cdot 10^{-4}$ | $3 \cdot 10^{-4}$ |
| Buffer size              |          $10^{7}$ |          $10^{6}$ |

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

|               |      GP-RS      |      GP-US      |      GP-VR      |     GP-EPIG     |           ACE-US            |        Aline (ours)         |
| :------------ | :-------------: | :-------------: | :-------------: | :-------------: | :-------------------------: | :-------------------------: |
| rpart (6D)    | $0.07 \pm 0.03$ | $0.04 \pm 0.02$ | $0.04 \pm 0.02$ | $0.05 \pm 0.02$ | $\mathbf{0 . 0 1} \pm 0.00$ | $\mathbf{0 . 0 1} \pm 0.00$ |
| svm (8D)      | $0.22 \pm 0.11$ | $0.11 \pm 0.05$ | $0.12 \pm 0.07$ | $0.15 \pm 0.08$ |       $0.04 \pm 0.01$       | $\mathbf{0 . 0 3} \pm 0.01$ |
| ranger (9D)   | $0.10 \pm 0.02$ | $0.07 \pm 0.01$ | $0.08 \pm 0.02$ | $0.08 \pm 0.02$ | $\mathbf{0 . 0 2} \pm 0.01$ | $\mathbf{0 . 0 2} \pm 0.01$ |
| xgboost (16D) | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ | $0.09 \pm 0.02$ |       $0.04 \pm 0.01$       | $\mathbf{0 . 0 3} \pm 0.01$ |

> **Image description.** The image presents a multi-panel line graph consisting of three subplots arranged horizontally, each illustrating predictive performance in terms of RMSE (Root Mean Squared Error) across different active learning benchmark functions. Each subplot shows multiple colored lines, representing different methods, with translucent shaded areas indicating confidence intervals. A common legend is placed below the three panels.
>
> Each of the three panels shares a common y-axis labeled "RMSE ↓", with the downward arrow suggesting that lower values are better. The x-axis in each panel represents an increasing numerical value, likely corresponding to the number of queries or iterations, though not explicitly labeled.
>
> The panels are titled as follows:
>
> 1.  **Gramacy 1D (Left Panel):**
>
>     - The y-axis ranges from 0.0 to 0.5.
>     - The x-axis ranges from 0 to 30, with major ticks at 5-unit intervals.
>     - Six distinct lines are plotted:
>       - **GP-RS (grey circles):** Starts at approximately 0.5 and gradually decreases, leveling off around 0.3 with a relatively wide shaded confidence interval.
>       - **GP-US (brown squares):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.15 with a narrow shaded confidence interval.
>       - **GP-VR (blue upward triangles):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.1 with a narrow shaded confidence interval.
>       - **GP-EPIG (purple diamonds):** Starts at approximately 0.5 and decreases sharply, leveling off around 0.1 with a narrow shaded confidence interval.
>       - **ACE-US (green crosses):** Starts at approximately 0.5 and decreases, leveling off around 0.1 with a narrow shaded confidence interval.
>       - **ALINE (ours) (orange stars):** Starts at approximately 0.5 and shows the steepest initial decrease, reaching the lowest RMSE value of approximately 0.05 by the end of the x-axis range, with a narrow shaded confidence interval.
>
> 2.  **Branin 2D (Middle Panel):**
>
>     - The y-axis ranges from 0.0 to 0.9.
>     - The x-axis ranges from 0 to 50, with major ticks at 10-unit intervals.
>     - Similar to the first panel, six lines are plotted, all showing a decreasing trend:
>       - **GP-RS (grey circles):** Starts at approximately 0.9 and decreases, leveling off around 0.2 with a relatively wide shaded confidence interval.
>       - **GP-US (brown squares):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>       - **GP-VR (blue upward triangles):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>       - **GP-EPIG (purple diamonds):** Starts at approximately 0.9 and decreases sharply, leveling off around 0.05 with a narrow shaded confidence interval.
>       - **ACE-US (green crosses):** Starts at approximately 0.9 and decreases, leveling off around 0.1 with a narrow shaded confidence interval.
>       - **ALINE (ours) (orange stars):** Starts at approximately 0.9 and shows the steepest initial decrease, reaching the lowest RMSE value of approximately 0.03 by the end of the x-axis range, with a narrow shaded confidence interval.
>
> 3.  **Three Hump Camel 2D (Right Panel):**
>     - The y-axis ranges from 0.0 to 2.5.
>     - The x-axis ranges from 0 to 50, with major ticks at 10-unit intervals.
>     - The six lines are plotted, generally showing a decreasing trend, but with higher RMSE values compared to the other panels:
>       - **GP-RS (grey circles):** Starts at approximately 2.5 and decreases, leveling off around 0.5 with a relatively wide shaded confidence interval.
>       - **GP-US (brown squares):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.2 with a narrow shaded confidence interval.
>       - **GP-VR (blue upward triangles):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.3 with a narrow shaded confidence interval.
>       - **GP-EPIG (purple diamonds):** Starts at approximately 2.5 and decreases sharply, leveling off around 0.3 with a narrow shaded confidence interval.
>       - **ACE-US (green crosses):** Starts at approximately 2.5 and decreases, leveling off around 0.6 with a narrow shaded confidence interval.
>       - **ALINE (ours) (orange stars):** Starts at approximately 2.5 and decreases, leveling off around 0.7 with a narrow shaded confidence interval.
>
> A horizontal legend below the plots identifies each method by its color and marker:
>
> - `GP-RS` (grey line with circles)
> - `GP-US` (brown line with squares)
> - `GP-VR` (blue line with upward triangles)
> - `GP-EPIG` (purple line with diamonds)
> - `ACE-US` (green line with crosses)
> - `ALINE (ours)` (orange line with stars)

Figure A3: Predictive performance in terms of RMSE on three other active learning benchmark functions. Results show the mean and $95 \%$ confidence interval (CI) across 100 runs. Notably, on the Three Hump Camel function, the performance of amortized methods like Aline and ACE-US is limited, as its output scale significantly differs from the pre-training distribution, highlighting a scenario of distribution shift.

area for future work, such as training Aline on a broader prior distribution of functions, potentially leading to more universally capable models.

Acquisition visualization for AL. To qualitatively understand the behavior of our model, we visualize the query strategy employed by Aline for AL on a randomly sampled synthetic function Figure A4. This visualization illustrates how Aline iteratively selects query points to reduce uncertainty and refine its predictive posterior.

Hyperparameter inference visualization. We now visualize the evolution of Aline's estimated posterior distributions for the underlying GP hyperparameters for a randomly drawn 2D synthetic GP function (see Figure A5). The posteriors are shown after 1, 15, and 30 active data acquisition steps. As Aline strategically queries more informative data points, its posterior beliefs about these generative parameters become increasingly concentrated and accurate.

Inference time. To assess the computational efficiency of Aline, we report the inference times for the AL tasks in Table A4. The times represent the total duration to complete a sequence of 30 steps for 1D functions and 50 steps for 2D functions, averaged over 10 independent runs. As both Aline and ACE-US perform inference via a single forward pass per step once trained, they are significantly faster compared to traditional GP-based methods.

Table A4: Comparison of inference times (seconds) for different AL methods on 1D (30 steps) and 2D (50 steps) tasks. Values are averaged over 10 runs (mean $\pm$ standard deviation).

| Methods | Inference time (s) |                 |
| :------ | :----------------: | :-------------: |
|         |   1D \& 30 steps   | 2D \& 50 steps  |
| GP-US   |  $0.62 \pm 0.09$   | $1.72 \pm 0.23$ |
| GP-VR   |  $1.41 \pm 0.14$   | $4.03 \pm 0.18$ |
| GP-EPIG |  $1.34 \pm 0.11$   | $3.43 \pm 0.24$ |
| ACE-US  |  $0.08 \pm 0.00$   | $0.19 \pm 0.02$ |
| ALINE   |  $0.08 \pm 0.00$   | $0.19 \pm 0.02$ |

---

#### Page 26

> **Image description.** This image presents a grid of 20 individual line graphs, arranged in 4 rows and 5 columns, each labeled sequentially from "Step 1" to "Step 20". These graphs illustrate a sequential query strategy, likely for a Gaussian Process (GP) model, over 20 iterative steps.
>
> Each individual graph shares common visual elements:
>
> - **Axes**: A horizontal x-axis labeled "x" ranging from -4 to 4, and a vertical y-axis labeled "y" with varying ranges (e.g., -1.5 to 1.5 in early steps, narrowing to -0.5 to 1.0 in later steps).
> - **Legend (visible in Step 1 only)**:
>   - "Prediction": A solid blue line representing the model's current prediction.
>   - "Ground Truth": A dashed red line representing the true underlying function.
>   - "Targets": Small, dark red circular markers, indicating specific points of interest or the true values at certain locations.
>   - "Context": Small, green circular markers, representing data points that have already been queried and observed by the model.
>   - "Next Query": A dashed red vertical line, indicating the x-coordinate where the model proposes to query next.
> - **Uncertainty Region**: A light blue shaded area around the "Prediction" line, representing the model's uncertainty or confidence interval.
>
> **Progression across the 20 steps:**
>
> - **Steps 1-5**: In the initial steps, the "Prediction" (blue line) deviates significantly from the "Ground Truth" (red dashed line), and the "Uncertainty" (blue shaded region) is broad across the entire x-range. The "Next Query" (red dashed vertical line) is strategically placed in regions of high uncertainty or where the prediction is poor. For example, in Step 1, the query is around x=-2.5. By Step 5, the prediction starts to align better with the ground truth, and the uncertainty begins to reduce around the initial "Context" points.
> - **Steps 6-10**: As more "Context" points (green dots) accumulate from previous queries, the "Prediction" line increasingly conforms to the "Ground Truth" curve. The "Uncertainty" region visibly narrows in areas where context points are dense. The "Next Query" continues to target regions where uncertainty remains high or where the prediction still needs refinement.
> - **Steps 11-15**: The model's "Prediction" closely matches the "Ground Truth" across most of the x-range. The "Uncertainty" region is significantly reduced, becoming very thin in many areas, indicating high confidence. The y-axis range in these plots has also narrowed, reflecting a more precise prediction. The "Next Query" points are still visible, suggesting continuous refinement.
> - **Steps 16-20**: In the final steps, the "Prediction" line is almost perfectly aligned with the "Ground Truth" line, and the "Uncertainty" region is minimal across the entire range of x. The model has learned the underlying function very well. The "Next Query" continues to be placed, likely for minor adjustments or to confirm the model's high confidence. The density of "Context" points (green dots) has increased substantially, covering the ground truth curve densely.
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
>
> - A white background with a light gray grid.
> - A y-axis labeled "Density", with numerical values ranging from 0.0 up to varying maximums depending on the plot.
> - A blue curve outlining a shaded purple area, representing the "ALINE Posterior" distribution.
> - A vertical dashed red line, representing the "True Value".
> - The x-axis label varies depending on the parameter being estimated.
>
> **Column 1: t = 1**
>
> - **Top plot (Lengthscale 1 Posterior):** The x-axis is "Lengthscale 1 Value" (0 to 3). The blue curve is broad and relatively flat, peaking around 1.5-2.0. The red dashed line is at x = 1.0. A legend in the top-left plot clarifies "ALINE Posterior" (blue curve) and "True Value" (red dashed line).
> - **Middle plot (Lengthscale 2 Posterior):** The x-axis is "Lengthscale 2 Value" (0 to 3). The blue curve is broad and somewhat bimodal, with peaks around 1.0 and 2.0-2.5. The red dashed line is at x = 2.6.
> - **Bottom plot (Scale Posterior):** The x-axis is "Scale Value" (0.00 to 1.00). The blue curve is broad, with a sharp peak around 0.15-0.20, then gradually decreasing. The red dashed line is at x = 0.15.
>
> **Column 2: t = 15**
>
> - **Top plot (Lengthscale 1 Posterior):** The x-axis is "Lengthscale 1 Value" (0 to 3). The blue curve is noticeably narrower and taller than at t=1, peaking sharply around 1.2-1.3. The red dashed line remains at x = 1.0.
> - **Middle plot (Lengthscale 2 Posterior):** The x-axis is "Lengthscale 2 Value" (0.5 to 3.0). The blue curve is narrower and taller, still somewhat bimodal but more concentrated, with peaks around 1.5 and 2.2. The red dashed line is at x = 2.6.
> - **Bottom plot (Scale Posterior):** The x-axis is "Scale Value" (0.00 to 1.00). The blue curve is significantly narrower and taller, peaking sharply around 0.15-0.20. The red dashed line is at x = 0.15.
>
> **Column 3: t = 30**
>
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
>
> - **Data Points ($\text{x}_t$)**: Numerous small, circular points are scattered across the plot. A dense cluster of these points, ranging in color from light orange to dark reddish-brown, is concentrated around the central high-density region of the contour plot, specifically around (0.5, 0.6). A few additional, lighter orange points are sparsely distributed further out, for example, near (0.0, 0.0), (0.9, 0.1), and (0.9, 0.9).
> - **True Location ($\theta$)**: A single, prominent blue star symbol is positioned within the dense cluster of data points, also centered around (0.5, 0.6), indicating a specific target location.
>
> Two vertical colorbars are positioned to the right of the main plot:
>
> - **Left Colorbar (Posterior log probability density)**: This colorbar displays a gradient from black at the bottom to white at the top, with intermediate shades of gray and light blue-teal, corresponding to the contour colors. It is labeled vertically as "Posterior $\text{log } q(\theta|\mathcal{D}_T)$". The scale ranges from -1800 at the bottom to 0 at the top, with tick marks at -1800, -1500, -1200, -900, -600, -300, and 0.
> - **Right Colorbar (Time step)**: This colorbar shows a gradient from light orange at the bottom to dark reddish-brown at the top. It is labeled vertically as "Time step $\text{t}$". The scale ranges from 0 at the bottom to 30 at the top, with tick marks at 0, 5, 10, 15, 20, 25, and 30. The color of the circular data points on the main plot corresponds to this time step color scale, with earlier steps being lighter orange and later steps being darker brown.
>
> A small legend box is located in the top-right corner of the main plot, explaining the symbols:
>
> - An orange dot next to "$\text{x}_t$"
> - A blue star next to "$\theta$"
>
> The visual arrangement suggests that the contour plot represents a probability distribution, the blue star marks a true value, and the colored dots represent sampled or queried locations over time, with their concentration around the star indicating an inference process converging towards the true location.

Figure A6: Visualization of Aline's design policy and resulting posterior for the Location Finding task. The contour plot shows the log posterior probability density of the source location $\theta$ (true location marked by blue star) after $T=30$ steps. Queried locations, with color indicating the time step of acquisition, demonstrating a concentration of queries around the true source.

> **Image description.** A series of three line graphs, arranged horizontally, displays marginal posterior distributions for different parameters. Each graph features a white background, black axes, and labels, and contains one or more colored lines representing probability distributions, along with a dashed red vertical line indicating a "true parameter value" as per the context.
>
> 1.  **Left Graph (Parameter $\rho$)**:
>
>     - **Title**: $\rho$
>     - **Y-axis**: Labeled "$p(\theta)$", ranging from 0 to 10, with major ticks at 0, 5, and 10.
>     - **X-axis**: Labeled "$\rho$", ranging from 0.0 to 1.0, with major ticks at 0.0, 0.5, and 1.0.
>     - **Content**: A single blue line forms a distinct, bell-shaped curve, peaking sharply at approximately x=0.7 and reaching a maximum y-value just above 10. The curve is narrow, indicating a concentrated distribution. A dashed red vertical line is positioned precisely at the peak of the blue curve, around x=0.7.
>
> 2.  **Middle Graph (Parameter $\alpha$)**:
>
>     - **Title**: $\alpha$
>     - **Y-axis**: Unlabeled but implicitly representing $p(\theta)$, ranging from 0 to 40, with major ticks at 0, 20, and 40.
>     - **X-axis**: Labeled "$\alpha$", ranging from 0.0 to 1.0, with major ticks at 0.0, 0.5, and 1.0.
>     - **Content**: This graph shows three very narrow, sharp peaks, each representing a distribution.
>       - A blue line peaks around x=0.3, reaching a y-value slightly above 40.
>       - A green line peaks slightly to the right of the blue line, around x=0.35, also reaching a y-value slightly above 40.
>       - An orange line peaks further to the right, around x=0.4, also reaching a y-value slightly above 40.
>         All three peaks are extremely narrow and tall, indicating highly concentrated distributions. Three dashed red vertical lines are present, each aligned with the peak of one of the colored curves, indicating their respective "true parameter values."
>
> 3.  **Right Graph (Parameter $u$)**:
>     - **Title**: $u$
>     - **Y-axis**: Unlabeled but implicitly representing $p(\theta)$, ranging from 0 to 2, with major ticks at 0, 1, and 2.
>     - **X-axis**: Labeled "$\log(u)$", ranging from -5.0 to 5.0, with major ticks at -5.0, 0.0, and 5.0.
>     - **Content**: A single blue line forms an extremely narrow, almost vertical spike, peaking around x=-4.0 and reaching a y-value just above 2. This signifies a very highly concentrated distribution. A dashed red vertical line is positioned precisely at the peak of the blue curve, around x=-4.0.

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
