# EFFICIENT AUTOREGRESSIVE INFERENCE FOR TRANSFORMER PROBABILISTIC MODELS - Backmatter

---

## ACKNOWLEDGEMENT

This work was supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence, FCAI) and ELISE Networks of Excellence Centres (EU Horizon:2020 grant agreement 951847). The project is also supported by the EuroHPC Joint Undertaking and its members including top- up funding by Ministry of Education and Culture. CH and SK were supported by Business Finland (VirtualLab4Pharma, grant agreement 3597/31/2023) and the European Union (Horizon Europe, grant agreement 101214398, ELLIOT). NL was funded by Business Finland (project 3576/31/2023) and LUMI AI Factory (EU Horizon Europe Joint Undertaking and its members including top- up funding by Ministry of Education and Culture). YY was supported by the Ministry of Education and Culture's Doctoral Education Pilot under Decision No. VN/3137/2024- OKM- 6 (The Finnish Doctoral Program Network in Artificial Intelligence, AI- DOC). SK was supported by UKRI Turing AI World- Leading Researcher Fellowship (EP/W002973/1). LA was supported by Research Council of Finland grants 356498 and 358980, the latter also supporting FS. The authors also acknowledge the research environment provided by ELLIS Institute Finland.

We acknowledge Verda for providing computational resources. Additionally, we acknowledge CSC - IT Center for Science, Finland, for computational resources provided by the LUMI supercomputer, owned by the EuroHPC Joint Undertaking and hosted by CSC and the LUMI consortium (LUMI projects 462000943 and 462000874). Access was provided through the Finnish LUMI- OKM allocation. We also acknowledge the computational resources provided by the Aalto Science- IT Project from Computer Science IT.

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the granting authority. Neither the European Union nor the granting authority can be held responsible for them.

> **Image description.** A horizontal graphic displaying acknowledgments for funding, featuring two distinct sections: one for the European Union and one for the EuroHPC Joint Undertaking.
>
> On the left side of the image, the European Union funding acknowledgment is presented. This section includes the official European Union flag logo—a circle of twelve gold stars on a deep blue background. Above the logo, the text "Co-funded by" is displayed, and below the logo, the text "the European Union" is written.
>
> On the right side of the image, the EuroHPC acknowledgment is shown. This section features a stylized logo consisting of a circular arrangement of blue lines, suggesting a network or a globe. To the right of this graphic, the text "EuroHPC" is prominently displayed in a large, bold font, with the phrase "Joint Undertaking" written in a smaller font directly beneath it.
>
> The two sections are arranged side-by-side, separated by a significant amount of white space, indicating two separate sources of support.

## ETHICS STATEMENT

This work uses only publicly available datasets and synthetic simulators, with no sensitive data involved. The methods are for research purposes and pose no foreseeable ethical risks. We have followed the ICLR Code of Ethics.

## REPRODUCIBILITY STATEMENT

We will release the code, including the training and evaluation pipelines, as well as configuration files, in the repository linked below. All experiments use public datasets or, when applicable, a simulator for synthetic data. Algorithmic details are presented in Algorithms 1 and 2, and all hyperparameters and training schedules are specified in the configuration files and documented in the appendix. Ablation studies are also reported in the appendix. We do not release pretrained weights, and no special data licenses or usage constraints apply.

## REFERENCES

Luigi Acerbi. Variational Bayesian Monte Carlo. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2018.

Luigi Acerbi. Variational Bayesian Monte Carlo with noisy likelihoods. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2020.

Luigi Acerbi, Kalpana Dokka, Dora E Angelaki, and Wei Ji Ma. Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS Computational Biology, 14(7):e1006110, 2018.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block Diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, 2025.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518):859- 877, 2017.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few- shot learners. Advances in Neural Information Processing Systems, 2020.

Wessel P Bruinsma, James Requeima, Andrew YK Foong, Jonathan Gordon, and Richard E Turner. The Gaussian neural process. In 3rd Symposium on Advances in Approximate Bayesian Inference, 2021.

Wessel P Bruinsma, Stratis Markou, James Requeima, Andrew YK Foong, Tom R Andersson, Anna Vaughan, Anthony Buonomo, J Scott Hosking, and Richard E Turner. Autoregressive conditional neural processes. In International Conference on Learning Representations, 2023.

Paul E Chang, Nasrulloh Loka, Daolang Huang, Ulpu Remes, Samuel Kaski, and Luigi Acerbi. Amortized probabilistic conditioning for optimization, simulation and inference. In International Conference on Artificial Intelligence and Statistics. PMLR, 2025.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean- Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International Conference on Machine Learning. PMLR, 2020.

Tri Dao. FlashAttention- 2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2023.

Nicola De Cao, Wilker Aziz, and Ivan Titov. Block neural autoregressive flow. In Uncertainty in artificial intelligence. PMLR, 2020.

Justin Deschenaux, Lan Tran, and Caglar Gulcehre. Partition generative modeling: Masked modeling without masks, 2025. URL https://arxiv.org/abs/2505.18883.

Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, and Fergus Simpson. Neural diffusion processes. In International Conference on Machine Learning. PMLR, 2023.

Lasse Elsemüller, Hans Olischläger, Marvin Schmitt, Paul- Christian Bürkner, Ullrich Koethe, and Stefan T. Radev. Sensitivity- aware amortized bayesian inference. Transactions on Machine Learning Research, 2024.

Leo Feng, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Efficient queries transformer neural processes. In NeurIPS 2022 Workshop on Meta- Learning, 2022.

Leo Feng, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Latent bottleneck attentive neural processes. In International Conference on Learning Representations, 2023.

Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Memory efficient neural processes via constant memory attention block. In International Conference on Machine Learning. PMLR, 2024.

Andrew YK Foong, Wessel P Bruinsma, Jonathan Gordon, Yann Dubois, James Requeima, and Richard E Turner. Meta- learning stationary stochastic process prediction with convolutional neural processes. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2020.

Marta Garnelo, Dan Rosenbaum, Chris J Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J Rezende, and SM Ali Eslami. Conditional neural processes. In International Conference on Machine Learning. PMLR, 2018a.Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Ali Eslami, and Yee Whye Teh. Neural processes. In ICML 2018 Workshop on Theoretical Foundations and Applications of Deep Generative Models, 2018b.Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. Made: Masked autoencoder for distribution estimation. In International Conference on Machine Learning. PMLR, 2015. Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H Macke. All- in- one simulation- based inference. In International Conference on Machine Learning. PMLR, 2024. Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1382- 1390, 2022. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2020. Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations, 2023. Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319- 326, 2025. Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022. Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter- efficient transfer learning for nlp. In International Conference on Machine Learning. PMLR, 2019. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen- Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low- rank adaptation of large language models. In International Conference on Learning Representations. PMLR, 2022. Chin- Wei Huang, David Krueger, Alexandre Lacoste, and Aaron Courville. Neural autoregressive flows. In International Conference on Machine Learning. PMLR, 2018. Bobby Huggins, Chengkun Li, Marlon Tobaben, Mikko J. Aarons, and Luigi Acerbi. PyVBMC: Efficient Bayesian inference in Python. Journal of Open Source Software, 8(86):5428, 2023. doi: 10.21105/joss.05428. Kazuki Irie. Why are positional encodings nonessential for deep autoregressive transformers? revisiting a petroglyph. Findings of the Association for Computational Linguistics: ACL 2025, pp. 551- 559, 2024. Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning. PMLR, 2021. Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2016. David C Knill and Alexandre Pouget. The Bayesian brain: The role of uncertainty in neural coding and computation. Trends in Neurosciences, 27(12):712- 719, 2004.

Konrad P Körding, Ulrik Beierholm, Wei Ji Ma, Steven Quartz, Joshua B Tenenbaum, and Ladan Shams. Causal inference in multisensory perception. PLoS One, 2(9):e943, 2007.

Jose Lara- Rangel, Nanze Chen, and Fengzhe Zhang. Exploring pseudo- token approaches in transformer neural processes. arXiv preprint arXiv:2504.14416, 2025.

Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In international conference on artificial intelligence and statistics. PMLR, 2011.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set Transformer: A framework for attention- based permutation- invariant neural networks. In International conference on machine learning. PMLR, 2019.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning. PMLR, 2023.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben- Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Shuze Liu, Trevor Holland, Wei Ji Ma, and Luigi Acerbi. Distilling noise characteristics and prior expectations in multisensory causal inference. 2025.

Sulin Liu, Peter J Ramadge, and Ryan P Adams. Generative marginalization models. In International Conference on Machine Learning. PMLR, 2024.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

Stratis Markou, James Requeima, Wessel P Bruinsma, Anna Vaughan, and Richard E Turner. Practical conditional neural processes via tractable dependent predictions. In International Conference on Learning Representations, 2022.

Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, and Marcus A Brubaker. Exploring exchangeable dataset amortization for bayesian posterior inference. In ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling, 2023.

Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, and Marcus Brubaker. Amortized in- context Bayesian posterior estimation. arXiv preprint arXiv:2502.06601, 2025.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022.

Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. PFNs4BO: In- context learning for bayesian optimization. In International Conference on Machine Learning. PMLR, 2023.

Kevin P. Murphy. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.

Kevin P Murphy. Probabilistic Machine Learning: Advanced Topics. MIT press, 2023.

Ryan L Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Janossy pooling: Learning deep permutation- invariant functions for variable- size inputs. In International Conference on Learning Representations, 2019.

Tung Nguyen and Aditya Grover. Transformer Neural Processes: Uncertainty- aware meta learning via sequence modeling. In International Conference on Machine Learning. PMLR, 2022.

George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017.

Massimiliano Patacchiola, Aliaksandra Shysheya, Katja Hofmann, and Richard E Turner. Transformer neural autoregressive flows. In ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling, 2024.

Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations. PMLR, 2022.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in- context learning on large data. In International Conference on Machine Learning. PMLR, 2025.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. Tabiclv2: A better, faster, scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139, 2026.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre- training. 2018. URL https://openai.com/index/language- unsupervised/.

Carl Edward Rasmussen and Christopher KI Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

Arik Reuter, Tim GJ Rudner, Vincent Fortuin, and David Rügamer. Can transformers learn full Bayesian inference in context? International Conference on Machine Learning, 2025.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T Chiu, and Volodymyr Kuleshov. The diffusion duality. In International Conference on Machine Learning, 2025.

Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng, Zhengzhong Liu, Eric Xing, John Thickstun, and Arash Vahdat. Esoteric language models: Bridging autoregressive and masked diffusion llms, 2026. URL https://arxiv.org/abs/2506.01928.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, 2024.

Francesco Silvestrin, Chengkun Li, and Luigi Acerbi. Stacking Variational Bayesian Monte Carlo. Transactions on Machine Learning Research, 2025.

Jascha Sohl- Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning. PMLR, 2015.

Yang Song, Jascha Sohl- Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score- based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. HART: Efficient visual generation with Hybrid AutoRegressive Transformer. In International Conference on Learning Representations, 2025.

Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In International Conference on Machine Learning. PMLR, 2014.

Benigno Uria, Marc- Alexandre Côté, Karol Gregor, Iain Murray, and Hugo Larochelle. Neural autoregressive distribution estimation. Journal of Machine Learning Research, 17(205):1- 37, 2016.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017.

George Whittle, Juliusz Ziomek, Jacob Rawling, and Michael A Osborne. Distribution transformers: Fast approximate Bayesian inference with on- the- fly prior adaptation. arXiv preprint arXiv:2502.02463, 2025.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast- dLLM: Training- free acceleration of diffusion LLM by enabling KV cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

Xiao Lei Zhang, Henri Begleiter, Bernice Porjesz, Wenyu Wang, and Ann Litke. Event related potentials during object recognition tasks. Brain Research Bulletin, 38(6):531- 538, 1995.

Xiyuan Zhang, Danielle C Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W Mahoney, et al. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In Advances in Neural Information Processing Systems, 2025.

Chunsheng Zuo, Pavel Guerzhoy, and Michael Guerzhoy. Position information emerges in causal transformers without positional encodings via similarity of nearby embeddings. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 9418- 9430, 2025.

## Table of Contents

A Method Details 18

A.1 Modules and notation 18

A.2 Training mask that implements (R1)-(R4) 18

A.3 Algorithms for autoregressive sampling and log-likelihood evaluation 19

## B Transformer Neural Process Baselines Details 20

B.1 TNP-D 20

B.2 TNP-ND 20

B.3 TNP-A 20

## C Computational Efficiency Details 21

C.1 Scaling with Batch Size 21

C.2 Impact of Custom Triton Kernel 22

C.3 Comparison to Open-Source Baselines 22

C.4 Training Time Scaling 23

C.5 Impact of Attention Patterns on Training Speed 24

C.6 Memory Usage 25

## D Experimental Details 26

D.1 Model Configuration 26

D.2 Datasets 27

D.3 Multisensory causal inference model and experiment details 28

D.4 Tabular model details 31

D.5 Evaluation Details 32

## E Additional Log-Predictive Density Results on Synthetic and EEG Tasks 34

E.1 Predictive Power of Different Heads 34

E.2 Results of Larger \(M\) 34

E.3 EEG Forecasting w/ and w/o Target Permutation 35

## F Additional Multisensory Causal Inference Model Results 35

## G Additional Tabular Foundation Model Results 35

## H Ablations and Extra Experiments 36

H.1 Comparison to Non-Permutation-Invariant Transformers 36

H.2 Positional Embeddings Ablation 36

H.3 Number of Samples Order Averaging Ablation 36

H.4 Extension to Latent Bottlenecked Attentive Neural Processes Model 37

H.5 Buffer Size Ablation 38

I Use of Large Language Models 38

---

*Transcribed with OCR and VLMs; text, equations, and figure descriptions may contain mistakes.*
