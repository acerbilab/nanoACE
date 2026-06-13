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
