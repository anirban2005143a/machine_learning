# **Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

**Rakshit Naidu**<sup>* 1 2</sup> **Harshita Diddee**<sup>* 3</sup> **Ajinkya Mulay**<sup>* 4</sup> **Aleti Vardhan**<sup>* 2</sup> **Krithika Ramesh**<sup>* 2</sup> **Ahmed Zamzam**<sup>5</sup> 

## **Abstract** 

In recent years, machine learning techniques utilizing large scale datasets have achieved remarkable performance. Differential privacy, by means of adding noise, provides strong privacy guarantees for such learning algorithms. The cost of differential privacy is often a reduced model accuracy and a lowered convergence speed. This paper investigates the impact of differential privacy on learning algorithms in terms of their carbon footprint due to either longer run-times or failed experiments. Through extensive experiments, further guidance is provided on choosing the noise levels which can strike a balance between desired privacy levels and reduced carbon emissions. 

## **1. Introduction** 

With the rising availability of large-scale, diverse datasets, performance of Machine Learning (ML) models have experienced a significant boost across a multitude of domains. This boost is also associated with the availability of extremescale datasets, which is heavily linked to individual user contributions achieved via crowd-sourcing. ML algorithms often perform operations directly on raw user data leading to a host of privacy violations. Differential Privacy (DP) (Dwork & Roth, 2014; Abadi et al., 2016) makes progress in this domain by providing strong privacy guarantees for such contributing individuals. This guarantee is achieved by means of noise addition, which can be done at various stages of the ML pipeline including : (1) _Local DP:_ Addition to the raw data (Cormode et al., 2018) (2) _Gradient DP:_ Addition to gradients after clipping (Abadi et al., 2016) (3) _Addition to Output & Objective DP:_ Addition to the final ML model or the loss function (Chaudhuri et al., 2011). 

> *Equal contribution 1Carnegie Mellon University 2Manipal Institute of Technology 3Bharati Vidyapeeth’s College of Engineering 4Purdue University 5The National Renewable Energy Laboratory. Correspondence to: Rakshit Naidu _<_ rnemakal@andrew.cmu.edu _>_ , Harshita Diddee _<_ harshitadd@gmail.com _>_ , Ajinkya Mulay _<_ mulay@purdue.edu _>_ . 

### **1.1. Impact on Climate Change** 

It is well-known that the computational resource investment requisite for training ML models generates a carbon footprint. This footprint is amplified in privacy-preserving setups where it is harder to reach consistent accuracy due to the addition of noise. Extended and failed runs (especially on larger datasets) actively contribute to an increase in the carbon footprint of ML experiments (Strubell et al., 2019). Therefore, an analysis of the climatic impact of this privacy modulation is critical. While the existing DP literature studies several performance aspects affected by varying privacy requirements, it lacks a comprehensive quantification of the carbon footprint of DP and how it is affected by variable privacy levels. Since DP also provides a mathematical paradigm to quantify the privacy budget of training ML models while tracking the privacy usage across multiple runs, this paper aims at quantifying the Carbon Emissions (CE) associated with varying privacy budgets of differentially private networks. In order to study impact of DP on these emissions, we implement _Gradient DP_ (DP-SGD (Abadi et al., 2016)) for natural language processing, image classification, and reinforcement learning domains to identify the privacy implications, model performance and most crucially the carbon footprint of each algorithm. As per our knowledge this is the first attempt to quantitatively benchmark the carbon footprint of differentially private ML models. 

### **1.2. Differential Privacy** 

_Definition 1_ : Given a randomized mechanism _A_ : _D →R_ (with domain _D_ and range _R_ ) and any two neighboring datasets _d_ 1 _, d_ 2 _∈D_ ( _i.e._ they differ by a single individual data element), _A_ is said to be ( _ε, δ_ )-differentially private for any subset _S ⊆R_<sup>1</sup> . 



$$
Pr [A (d1) \in{}{}S] \leq{}{}eε \cdot{}{} Pr [A (d2) \in{}{}S] + δ (1)
$$

Here, _ϵ ≥_ 0 _, δ ≥_ 0. A _δ_ = 0 case corresponds to pure differential privacy, while both _ϵ_ = 0 _, δ_ = 0 leads to an infinitely high privacy domain. Finally, _ϵ_ = _∞_ provides no privacy guarantees. For practical purposes we want 

1In this work, we exclusively use Gaussian noise 



**Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

_ϵ ≤_ 5 _, δ ≪ N_ <u>1</u><sup>where</sup><sup>_N_isthenumberofsamplesinthe</sup> dataset. (Dwork & Roth, 2014) 

The privacy of differentially private models can be quantified with parameters such as epsilon ( _ε_ ) and delta ( _δ_ ). Utilizing DP-SGD (Abadi et al., 2016), that is, adding noise to the gradients at each step during training using a clipping factor ( _S_ ) and noise multiplier ( _z_ ), the amount of noise added to the model can be linked to to the degree of privacy that the model can achieve. Theoretically, a lower value of _ε_ indicates a higher degree of privacy and this increased privacy degree is understandably, achieved at the expense of model performance due to the addition of the noise. The practical implication of this, however, includes a direct impact on the computational resources required to achieve model performance. Reduced privacy requirements allow the addition of noise with limited power, and hence, models can achieve appropriate performance without any significant resource expense. On the other hand, high privacy requirements necessitate adding a significantly large magnitude of noise which may directly lead to an increase in the number of training passes that the model has to iterate over to achieve the same accuracy. Further, noise addition may even lead to the non-convergence of some systems in the worst case. 

### **1.3. Related Work** 

Works such as (Strubell et al., 2019; Toews, 2020) discuss how conventional Machine Learning models impact carbon footprint. In particular, (Strubell et al., 2019) discusses how training a single Deep Learning model generates the total lifetime carbon footprint of nearly five cars (as mentioned in (Toews, 2020)) which is more than 17 times the amount of CO2 emissions generated by an average American per year. Regarding DP, there has been very little considerations on how Privacy-Preserving Machine Learning (PPML) impacts climate change. In (Qiu et al., 2021), a comprehensive study is presented on how local client-side models in Federated learning (FL) could potentially hold quality data required to understand climate change given data privacy concerns due to recent policies like GDPR (Skendziˇ c et al.´ , 2018). However, running local models on multiple client devices and aggregating them globally at the server level requires additional infrastructure in place, thereby causing a detrimental effect on carbon emissions. 

### **1.4. Contributions and Impacts** 

In this paper, we provide the first benchmark to quantitatively assess how DP-noise affect carbon emissions in three different tasks : (1) a Natural Language Processing (NLP) task using news classification (2) a Computer Vision (CV) task using the MNIST dataset and (3) a Reinforcement Learning (RL) task using the Cartpole control problem. Intuitively, when DP noise is added to ML pipelines, the carbon 

emissions should increase as the energy required for computations increase due to rising number of epochs required for convergence. In order to quantify how the addition of noise plays into climate change, we track carbon emissions in the models using the _codecarbon_ tool (Schmidt et al., 2021), a joint effort from authors of (Lacoste et al., 2019) and (Lottick et al., 2019). We record the average accuracy of several runs of the considered ML task to assess the behavior of DP-noise. 

Noise for masking data has been widely used in adversarial machine learning (Kurakin et al., 2016). Given the rise in Privacy-enhancing Technologies and privacy policies, noise addition has now become prevalent in the context of DP. We envisage this work to provide an insight on how much noise could result in varying amounts of CO2 emissions. Hence, our work takes a peek at how the addition of noise could impact a number of industries from healthcare to finance and justice, where sensitive data is commonly in use. 

## **2. Experimental Results** 

### **2.1. BERT** 

In these set of experiments, we evaluate the performance of two experiments on Bidirectional Encoder Representations from Transformers or BERT (Devlin et al., 2019). The model is fine-tuned for topic-classification of news articles. The primary objective of these experiments is to observe the carbon emissions and energy usage of vanilla BERT and DP-BERT (over different privacy levels). 

A randomly down-sampled subset (15,000 samples) of the AG News Classification (Anand, 2020) is used for this task with a 80/20 train-test split. We use BERT with the AdamW optimizer with the _bert-base-cased_ tokenizer (batch size ( _B_ ) of 32, Learning Rate being 0.0005) to conduct the following experiments for this task. 

### 2.1.1. CARBON EMISSIONS AND ENERGY CONSUMED 

The aim of this experiment is to analyse any possible association between different levels of privacy and carbon emissions. We run these experiments for 10 epochs each and present our results in Table 1 (averaged over 3 runs). Curiously, the carbon emissions for the _ϵ_ = 0 _._ 5 case is comparable to the EU’s 2021 passenger vehicle standard (Bandivadekar, 2013). The difference between a private model ( _ϵ_ = 0 _._ 5) and a non-private model ( _ϵ_ = _∞_ ) is approximately 1g of CO2, which is equivalent to the emissions from five Google searches (Sterling, 2009). 



**Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

|**Epsilon(**_ε_**)**|**CE(g)**|**EC(Wh)**|**Accuracy (%)**|
|---|---|---|---|
|0.5|26.7_±_0.63|49.9_±_1.2|48.5_±_1.39|
|2|26.3_±_0.49|49.3_±_0.9|52.0_±_0.73|
|5|26.1_±_0.1|48.9_±_0.9|52.3_±_0.36|
|15|25.9_±_0.09|48.5_±_0.1|54.2_±_1.40|
|_∞_**(Non-Private)**|25.2_±_0.00|47.1_±_0.27|58.5_±_5.29|



_Table 1._ **DP-BERT:** Emission-Accuracy trends over change in _ϵ_ for reaching 52% accuracy. 

In congruence with existing literature, the accuracy of the differentially private BERT increases consistently with the increase in epsilon. Interestingly, with the increase in the epsilon value – both, CE and EC decrease, though not by a very significant margin. Given that the range of the chosen _ε_ varies considerably, the consequent difference in the carbon emission is not proportionally varied. The practical implication of this invariance can be seen as incurring nearly the same carbon footprint for two versions of a model with different degrees of privacy. 

![Figure](assets/figure_0001_page_0003.svg)_Table 2._ Observing the number of epochs needed to achieve the threshold test accuracy ( _T_ ) with different privacy levels 

### 2.1.2. RESOURCE EXPENSE ANALYSIS 

The main aim of this experiment is to evaluate how many resources, in terms of consequent carbon and energy emissions are expended in order to achieve a target or threshold accuracy with different degrees of privacy. As defined in the previous set of experiments, we compute the accuracies over _ε_ = 0 _._ 5 _,_ 2 _,_ 5 _,_ 15.We set the target/threshold accuracy ( _T_ ) to 52% as shown in Table 2. 

It can be inferred from Table 2 that the Carbon Emission and Energy Usage required to attain the maximum experimental value of privacy is nearly 18 times the carbon emission required to attain the same threshold accuracy with a non-privacy preserving variant of the model. The practical consequence of this experiment dictates that enhancing the degree of privacy of the model, can incur a huge compute cost, which can invariably increase the carbon footprint of the model’s training pipeline. 

Additionally, From Figure 1, which present the accuracy curves for the experiment, it is quite evident that the vanilla variant (i.e a model without DP-noise) achieves the threshold accuracy with a significantly smaller carbon footprint than all the footprint of its privacy-preserving variants. 

![Figure](assets/figure_0002_page_0003.svg)_Figure 1._ **BERT with Gaussian DP:** Training and Testing accuracy trends over change in _ϵ_ where the threshold accuracy ( _T_ ) is set to 52%. 

|**Epsilon(**_ε_**)**|**CE(g)**|**EC(Wh)**|
|---|---|---|
|0.5*****|10.53_±_2.21|40.41_±_0.93|
|2*****|10.6_±_2.43|40.5_±_0.53|
|5|7.85_±_1.84|29.93_±_0.46|
|15|1.61_±_0.37|6.17_±_0.27|
|_∞_**(Non-Private)**|0.08_±_7e-04|0.38_±_3.3e-03|



_Table 3._ **MNIST:** Emission trends over change in _ϵ_ for reaching 70% accuracy ( ***** _70% accuracy not reached even after 200 epochs.)_ 

|**Epsilon(**_ε_**)**|**CE(g)**|**EC(Wh)**|
|---|---|---|
|0.5*|15.34|70.216|
|2|12.48|57.108|
|5|3.12|14.307|
|15|2.03|9.297|
|_∞_**(Non-Private)**|0.36|1.678|



_Table 4._ **CIFAR:** Emission trends over change in _ϵ_ for reaching 55% accuracy in a single run ( ***** _55% accuracy not reached even after 30 epochs.)_ 

### **2.2. MNIST & CIFAR** 

We evaluate our approach on the MNIST dataset (LeCun & Cortes, 2010) with a batch size of 128 using DP-SGD (Abadi et al., 2016). We use a simple multi-layer perceptron (MNIST 2NN) with a two hidden layers of 200 units each (parameters = 199,210) as the network from (McMahan et al., 2017). Our goal is to observe the trend in the CO2 emissions by allowing the model to train and reach _X_ accu- 



**Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

racy with different values of _ε_ (different levels of privacy). We compute the accuracies over _ε_ = 0 _._ 5 _,_ 2 _,_ 5 _,_ 15 as shown in Fig. 2. We set the target/threshold accuracy ( _T_ ) to 70% so that most of the privacy-variant models can achieve under 200 iterations. In Fig. 2 we see that only models with _ε_ = 5 _,_ 15 reach 70% accuracy within 200 epochs. Fig. 2 shows a clear trend on how increasing levels of privacy in ML models increases the amount of computation required to reach _T_ , thereby releasing higher carbon emissions. 

For the CIFAR-10 experiments, we use the ResNet18 model pre-trained on the ImageNet dataset. This deep variant of CNN is chosen instead of the simple MNIST 2NN in order to capture more complex and realistic features. We compute the accuracies over _ε_ = 0 _._ 5 _,_ 2 _,_ 5 _,_ 15 as shown in Fig. 3. We set the target/threshold accuracy ( _T_ ) to 55% as most of the private models can achieve under 30 iterations. We run the CIFAR experiments only for 30 iterations as we observe that there is little to no significant improvement beyond this. In Fig. 3, We observe similar trends as Fig. 2 on how increasing privacy levels involve higher computations to reach _T_ which in turn, release higher carbon emissions. 

![Figure](assets/figure_0003_page_0004.svg)_Figure 2._ **MNIST with Gaussian DP:** Training and test accuracy trends during training for multiple _ϵ_ values. 

### **2.3. Cartpole** 

For the reinforcement learning experiments, we trained a DQN over OpenAI Gym’s Cartpole-v0 environment. Due to page restrictions we defer the discussion of the RL experiments to the Appendix B. 

![Figure](assets/figure_0004_page_0004.svg)_Figure 3._ **CIFAR-10 with Gaussian DP:** Training and test accuracy trends during training for multiple _ϵ_ values. 

## **3. Conclusion** 

We demonstrate and highlight the prominent impact of Privacy-Preserving Machine Learning (PPML) on carbon emissions over three ML domains, namely, CV, NLP and RL. We observe that the stronger privacy regime, _i.e,_ a lower _ϵ_ value, ML algorithms always result in higher levels of carbon emissions in the CV and NLP domains. Curiously, results for RL are less obviously explained and we defer these discussions for future work. We conclude that alongside the challenge of obtaining state-of-the-art performance, PPML needs to reduce the number of epochs required to reach the desired performance. This leads us to the following critical questions which we leave as open questions for the future: (1) Can we reduce the number of iterations (including hyperparameter tuning) required to reach a privacy-utility ratio? (2) How much does the size of ML models affect the carbon emissions and the overall performance under PPML? 

## **4. Acknowledgements** 

We thank Fatemehsadat Mireshghallah for her valuable inputs and discussions. 

## **References** 

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. _Proceedings of the 2016 ACM SIGSAC Conference on Computer and_ 



#### **Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

- _Communications Security_ , Oct 2016. doi: 10 _._ 1145/ 2976749 _._ 2978318. URL http://dx _._ doi _._ org/ 10 _._ 1145/2976749 _._ 2978318. 

   - LeCun, Y. and Cortes, C. MNIST handwritten digit database. 2010. URL http://yann _._ lecun _._ com/ exdb/mnist/. 

- Anand, A. Ag news classification, 2020. URL https://www _._ kaggle _._ com/amananandrai/ ag-news-classification-dataset. 

- Bandivadekar, A. One (vehicle efficiency) table to rule them all. https://theicct _._ org/blogs/staff/ one-vehicle-efficiency-table-rulethem-all, 2013. Accessed: 2021-5-31. 

- Barto, A. G., Sutton, R. S., and Anderson, C. W. Neuronlike adaptive elements that can solve difficult learning control problems. _IEEE Transactions on Systems, Man, and Cybernetics_ , SMC-13(5):834–846, 1983. doi: 10 _._ 1109/ TSMC _._ 1983 _._ 6313077. 

- Chaudhuri, K., Monteleoni, C., and Sarwate, A. D. Differentially private empirical risk minimization. _Journal of Machine Learning Research_ , 12(3), 2011. 

- Cormode, G., Jha, S., Kulkarni, T., Li, N., Srivastava, D., and Wang, T. Privacy at scale: Local differential privacy in practice. In _Proceedings of the 2018 International Conference on Management of Data_ , pp. 1655–1658, 2018. 

- Dabney, W., Rowland, M., Bellemare, M. G., and Munos, R. Distributional reinforcement learning with quantile regression, 2017. 

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10 _._ 18653/ v1/N19-1423. URL https://www _._ aclweb _._ org/ anthology/N19-1423. 

- Dwork, C. and Roth, A. The algorithmic foundations of differential privacy. _Found. Trends Theor. Comput. Sci._ , 9(3–4):211–407, August 2014. ISSN 1551-305X. doi: 10 _._ 1561/0400000042. URL https://doi _._ org/ 10 _._ 1561/0400000042. 

- Kurakin, A., Goodfellow, I., and Bengio, S. Adversarial machine learning at scale. _arXiv preprint arXiv:1611.01236_ , 2016. 

- Lacoste, A., Luccioni, A., Schmidt, V., and Dandres, T. Quantifying the carbon emissions of machine learning. _Workshop on Tackling Climate Change with Machine Learning at NeurIPS 2019_ , 2019. 

- Lottick, K., Susai, S., Friedler, S. A., and Wilson, J. P. Energy usage reports: Environmental awareness as part of algorithmic accountability. _Workshop on Tackling Climate Change with Machine Learning at NeurIPS 2019_ , 2019. 

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep networks from decentralized data, 2017. 

- Qiu, X., Parcollet, T., Fernandez-Marqu´ es, J., de Gusm´ ao,˜ P. P. B., Beutel, D. J., Topal, T., Mathur, A., and Lane, N. D. A first look into the carbon footprint of federated learning. _CoRR_ , abs/2102.07627, 2021. URL https: //arxiv _._ org/abs/2102 _._ 07627. 

- Schmidt, V., Goyal, K., Joshi, A., Feld, B., Conell, L., Laskaris, N., Blank, D., Wilson, J., Friedler, S., and Luccioni, S. CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing. 2021. doi: 10 _._ 5281/zenodo _._ 4658424. 

- Skendziˇ c, A., Kova´ ciˇ c, B., and Tijan, E.´ General data protection regulation — protection of personal data in an organisation. In _2018 41st International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO)_ , pp. 1370–1375, 2018. doi: 10 _._ 23919/MIPRO _._ 2018 _._ 8400247. 

- Sterling, G. Calculating the carbon footprint of a google search. 2009. URL https: //searchengineland _._ com/calculatingthe-carbon-footprint-of-a-googlesearch-16105. 

- Strubell, E., Ganesh, A., and McCallum, A. Energy and policy considerations for deep learning in NLP. _CoRR_ , abs/1906.02243, 2019. URL http://arxiv _._ org/ abs/1906 _._ 02243. 

- Toews, R. Deep learning’s carbon emissions problem. _Forbes_ , 06 2020. URL https://www _._ forbes _._ com/ sites/robtoews/2020/06/17/deeplearnings-climate-change-problem/?sh= 6bcb1b9f6b43. 

- Wang, B. and Hegde, N. Privacy-preserving q-learning with functional noise in continuous state spaces, 2019. 



**Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

## **A. Hyperparameter Tuning** 

### **A.1. BERT** 

We set out to understand the impact of noise introduced by DP on the model convergence up until a threshold test accuracy. We primarily tune the epsilon values (for the privacy guarantee) and the number of epochs. For a fair comparison we use constant hyperparameters for each epsilon so as to to minimize the scope of confounding variables (optimally tuned hyperparameters per epsilon value) for BERT (Table 1). 

Succeeding that though, since a lot of differentially private models are deployed in edge-focused setups where numerous clients can take hyperparameter optimization decisions locally - the experiments (Table 2) with the thresholding accuracy were conducted. On grounds of ablation, a subset of these locally available hyperparamters were chosen with the objective to evaluate the change in carbon and energy emissions when these hyperparameters are altered: the privacy guarantee (meant to be a globally decided parameter in a distributed setup) and the number of epochs (meant to be a locally decided parameter) are changed. 

### **A.2. MNIST & CIFAR** 

Similar to the BERT approach, we use consistent hyperparameters across different _ε_ -privacy levels for the MNIST dataset. 

However, due to serious model performance degradation on the CIFAR-10 dataset we use an alternate approach. We tweak a set of hyperparameters to achieve optimal performance at the respective _ε_ -privacy levels when the model performs poorly. For instance, we observe a major drop in performance for only the non-private model when a constant learning rate of 10<sup>_−_3</sup> is used for both the private and non-private settings. So, we instead choose a learning rate of 10<sup>_−_6</sup> for the non-private model which leads to optimal performance. 

In our experiments, we also use the RMS-PROP optimizer which leads to more stable results & faster convergence during training. 

## **B. Reinforcement Learning: Experiments & Discussion** 

### **B.1. Cartpole** 

The Cartpole-v0 environment (Barto et al., 1983) consists of an un-actuated joint to a cart. There are two possible actions which involve a force of +1 or -1 being applied to the cart along a friction-less track. The pole starts upright, with the goal of preventing it from falling over. For every time-step that the pole is upright, a reward of +1 is added to the total 

reward. However, if the pole exceeds 15 degrees from the vertical, or if the cart moves more than 2.4 units from the center, the episode ends. 

![Figure](assets/figure_0005_page_0006.svg)_Figure 4._ **CartPole with Gaussian DP:** Episodes vs Rewards for the mean reward every 100 episodes 

![Figure](assets/figure_0006_page_0006.svg)_Figure 5._ **Acrobot with Gaussian DP:** CO2 vs Epsilon values post training after a 1000 episodes 

The DQN’s configuration (including the hyperparameters) is the same as the one used in (Wang & Hegde, 2019), and we observed results similar to this paper, with one variant of DP model slightly outperforming the baseline as shown in 4. It consists of a single hidden layer with 16 neurons. For our non-private experiment we obtained a mean reward of 19.94 and carbon emissions of 0.22 g on average (over a 1000 episodes). We provide results of the private variants in Fig. 4. Our setup included multiple experiments. 

- Noise addition to DQN’s output layer only. (1) 

- Noise addition to both, the DQN’s output layer and its parameters. The noise added to the parameters is the averaged noise sampled from the _noisebuffer_ function 



**Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning** 

![Figure](assets/figure_0007_page_0007.svg)_Figure 6._ **Acrobot with Gaussian DP:** Episodes vs Rewards for the mean reward every episode for a 1000 episodes 

vacy guarantees lead to higher carbon emissions. As a sanity check, we can observe (from Table 8) that the mean reward increases with higher epsilons (lower privacy requirement) due to reduced noise levels. 

|**Epsilon**<sup>_~~∗~~_</sup><br>(_ϵ_<sup>_∗_</sup>_∝_15_ϵ_)|**Sigma**<br>(_σ ∝_<sup>1</sup><br>_ϵ_ <sup>)</sup>|**Mean Reward**|**CE (g)**|
|---|---|---|---|
|1|<br>15|4.5 ± 0.6|1.03 ± 0.06|
|3|5|2.2 ± 0.2|0.96 ± 0.03|
|7.5|2|19.9 ± 0.5|1.14 ± 0.06|
|30|0.5|19.4 ± 0.1|1.15 ± 0.06|



_Table 5._ **CartPole:** Emission trends over change in _ϵ_<sup>_∗_</sup> post 1000 episodes in (1) following (Wang & Hegde, 2019) 

|**Epsilon**<sup>_~~∗~~_</sup>|**Sigma**<br>|**Mean Reward**|**CE (g)**|
|---|---|---|---|
|(_ϵ_<sup>_∗_</sup>_∝_15_ϵ_)|(_σ ∝_<sup>1</sup><br>_ϵ_ <sup>)</sup>|||
|1|<br>15|2.3 ± 0.9|0.41 ± 0.01|
|3|5|10.2 ± 0.8|0.5 ± 0.02|
|7.5|2|7.6 ± 0.7|0.45 ± 0.02|
|30|0.5|13.8 ± 0.1|0.48 ± 0.03|



_Table 6._ **CartPole:** Emission trends post 1000 episodes in (2) 

### during the forward pass. (2) 

We varied the value of the variance _σ_ of the distribution to observe its impact on the function approximated by the DQN. As expected, with increasing noise addition to the model ( _i.e._ , increasing value of _σ_ ), we notice a drop in the average reward. Subsequently, the increased computations lead to higher carbon emissions. We observe that there is a significant increase in CE from Table 6 to Table 7 when the number of episodes increase. 

### **B.2. Acrobot** 

We alternatively run experiments on the Acrobot-v1 environment. Acrobot-v1 has 2 joints and links and the joint between the links is actuated. The environment’s initial position has the links in a resting state, hanging downward, with the objective being to swing the end of the lowermost link up to a specified height. There are 3 potential actions, to apply positive torque, to apply negative torque, or to do nothing. The agent attempts to maximize its reward within _500_ time-steps, with each time-step potentially incurring a punishment of _-1_ , thus making _-500_ the worst possible reward. If the agent reaches the given height, a reward of _0_ is returned. The state returned consists of the joint angular velocities and the sin and cos values of the two rotational joint angles. 

|**Epsilon**<sup>_~~∗~~_</sup><br>(_ϵ_<sup>_∗_</sup>_∝_15_ϵ_)|**Sigma**<br>(_σ ∝_<sup>1</sup><br>_ϵ_ <sup>)</sup>|**Mean Reward**|**CE (g)**|
|---|---|---|---|
|1|15|13.2 ± 0.3|3.51 ± 0.26|
|3|5|13.7 ± 0.9|2.31 ± 0.28|
|7.5|2|18.1 ± 0.1|2.72 ± 0.23|
|30|0.5|19.8 ± 0.6|4.0 ± 0.31|



_Table 7._ **CartPole:** Emission trends post 5000 episodes in (2) 

|**Epsilon**|**Mean Reward**|**CO2**|
|---|---|---|
|0.5|-150.7 ± 18.69|3.25 ± 0.372|
|2|-138.0 ± 11.15|3.0 ± 0.245|
|5|-120.6 ± 4.64|0.98 ± 0.035|
|15|-117.5 ± 2.45|0.92 ± 0.023|
|0|-110.5 ± 1.22|0.33 ± 0.023|



_Table 8._ **Acrobot-v1** : Emission trends post a 1000 Episodes 

For the purpose of training the Acrobot environment, we used distributed reinforcement learning with quantile regression (Dabney et al., 2017). Quantile regression models the distribution of the returned rewards instead of only considering the mean of the distribution. We use the Opacus library in PyTorch to add the necessary DP guarantees. From Fig.5, we observe a consistent upward trend of the CO2 emissions with the epsilon value. This directly implies that better pri- 

