# The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression 

**Warren Johnson**<sup>*</sup> 

Plexor Labs 

Principal Researcher 

## **Abstract** 

The rapid proliferation of Large Language Models has created an environmental paradox: the very technology that could help solve climate challenges is itself becoming a significant contributor to global carbon emissions. We evaluate whether prompt compression improves inference energy efficiency using a large API experiment (28,421 successful calls out of 28,428 planned) and report detailed comparative results for the released matched two-model snapshot (GPT-4o-mini vs. DeepSeek-Chat, _N_ = 16 _,_ 270). Energy is reported as proxy-estimated active inference energy per trial (J), calibrated against local NVML/CodeCarbon measurements, and quality is tracked with benchmark pass rates. In the reported comparison, uncompressed pass rates are near 26% and drop to 1.5% at _r_ = 0 _._ 7 in the ratio-conditioned aggregate; DeepSeek exhibits strong output expansion under compression (21 to 798 tokens at _r_ = 0 _._ 3), corresponding to energy increases up to +2,140%, while GPT-4o-mini shows mixed effects including a reduction at _r_ = 0 _._ 5. These findings indicate that input-token reduction alone is not a reliable energy optimization strategy. For the evaluated models and settings, output-length control and model selection provide more consistent energy-quality tradeoffs than naive compression. 

**Keywords:** green AI, energy efficiency, prompt compression, carbon footprint, sustainable machine learning, LLM inference 

## **1 Introduction** 

The artificial intelligence research community finds itself at an inflection point. The same large language models that demonstrate remarkable capabilities across coding, reasoning, and creative tasks are simultaneously emerging as significant contributors to global energy consumption and carbon emissions. This tension between capability and sustainability demands urgent attention, not merely as an ethical consideration, but as a practical constraint that will increasingly shape the trajectory of AI deployment. 

The scale of this challenge is substantial. Public projections suggest AI-related electricity demand could reach hundreds of TWh annually within this decade. A single query to a frontier model like GPT-4 consumes approximately 10 times the energy of a traditional web search (de Vries, 2023), and with hundreds of millions of daily active users across major AI platforms, the cumulative impact is substantial. Perhaps most concerningly, this energy demand is growing at roughly 15% annually, four times faster than other computing sectors (Masanet et al., 2020). 

Yet this framing, while accurate, obscures a crucial nuance: the vast majority of AI’s carbon footprint now stems from _inference_ rather than training. While early research understandably focused on the dramatic energy costs of training large models (for example, GPT-3’s training consumed an estimated 1,287 MWh and generated 552 tonnes of CO2 (Patterson et al., 2021)), the operational reality has shifted. As Wu et al. (2022) document, inference now accounts for over 90% of the lifetime energy consumption of deployed models. This shift has profound implications: it means that optimizations targeting inference efficiency can have outsized environmental impact. 

This observation motivated the present study. In our prior work on Task-Aware Adaptive Compression (TAAC), we demonstrated that prompt compression could reduce inference costs by up to 93% while maintaining acceptable quality 

> *Contact: `warrenjo@plexor.dev` 

1 



for many task types (Johnson, 2026a;b;c). The economic case for compression is now well-established. But does cost reduction translate directly to energy reduction? And if so, can we quantify the potential environmental benefit of widespread adoption? 

**The Compression-Energy Hypothesis.** Our central hypothesis is straightforward: if prompt compression reduces the number of tokens processed during inference, and if energy consumption scales approximately with token count, then compression should yield proportional energy savings. However, several factors complicate this simple relationship. First, compression itself requires computation; running a model like LLMLingua-2 to compress prompts consumes energy that must be offset by savings downstream. Second, the relationship between tokens and energy is not strictly linear; factors such as attention mechanism scaling, batch size, and hardware utilization all introduce nonlinearities. Third, energy consumption varies dramatically across providers, hardware generations, and geographic locations, making generalization challenging. 

**Contributions.** 

This paper makes four primary contributions: 

1. **Large-Scale API Measurement with Reported Matched Subset** : We analyze a large API run and provide a reproducible matched two-model comparison (GPT-4o-mini vs. DeepSeek-Chat, _N_ = 16 _,_ 270) across compression ratios. The released comparison shows a 17 _×_ baseline efficiency gap and substantially larger divergence under aggressive compression. 

2. **The Output Token Explosion Paradox** : We document a surprising provider-dependent finding: prompt compression causes DeepSeek-Chat to generate up to 38 _×_ longer outputs (798 vs. 21 tokens), increasing energy by 2,140%. GPT-4o-mini shows no such effect, demonstrating that this paradox is architectural rather than universal. 

3. **Quality Degradation Under Compression in Evaluated Settings** : In the reported ratio-conditioned aggregate, pass rates drop from about 26% to under 2% at mild compression (r=0.7), indicating severe quality degradation for these settings. 

4. **Revised Deployment Guidance** : Our evidence supports treating compression as a conditional technique that requires output-length safeguards and task-level validation; model selection provides a larger and more reliable energy lever in the reported comparison. 

The remainder of this paper is organized as follows. Section 2 situates our work within the growing literature on sustainable AI. Section 3 details our energy measurement methodology, including the proxy formula we developed for API-based inference where direct power measurement is impossible. Section 4 presents our experimental design and results. Section 5 defines a deployment-facing Green AI score, and Section 6 discusses limitations and implications. Section 7 concludes with recommendations for practitioners and policymakers. 

## **2 Background and Related Work** 

The environmental impact of artificial intelligence has attracted increasing research attention over the past five years, catalyzed by several influential studies that quantified the carbon cost of training large neural networks. This section reviews the key developments that inform our approach. 

### **2.1 The Emergence of Green AI** 

The term “Green AI” was introduced by Schwartz et al. (2020) to describe research that prioritizes computational efficiency alongside accuracy, in contrast to “Red AI,” which pursues accuracy improvements with little regard for computational cost. Their analysis revealed a troubling trend: the compute required for state-of-the-art results had increased by approximately 300,000 _×_ between 2012 and 2018, with corresponding increases in energy consumption and carbon emissions. This exponential growth, they argued, was unsustainable and created barriers to entry that concentrated AI research within well-resourced institutions. 

The Green AI framework proposed several concrete recommendations: reporting floating-point operations (FLOPs) alongside accuracy metrics, considering financial cost as an evaluation criterion, and designing algorithms that achieve 

2 



comparable performance with reduced computation. These recommendations have gained traction, though adoption remains inconsistent. A survey by Henderson et al. (2020) found that among 100 randomly sampled NeurIPS 2019 papers, precisely zero reported carbon impacts, only 1% reported energy metrics, and just 17% reported any computerelated metrics at all. 

### **2.2 Training Energy: The Early Focus** 

The seminal work of Strubell et al. (2019) brought widespread attention to AI’s energy consumption by estimating that training a single Transformer model could emit more CO2 than five automobiles over their entire lifetimes. Their analysis considered not just the final training run but also the extensive hyperparameter search and failed experiments that precede publication, an approach they termed the “development accounting” perspective. This comprehensive view revealed that reported training costs dramatically underestimate the true energy footprint of AI research. 

Patterson et al. (2021) refined these estimates through direct measurement at Google, finding that careful optimization could dramatically reduce training emissions. Their key insight was that training energy depends critically on four factors: model architecture (with sparse Mixture-of-Experts models consuming as little as one-tenth the energy of dense equivalents), geographic location (with carbon intensity varying 5-10 _×_ across regions), datacenter efficiency (with hyperscale facilities achieving 1.4-2 _×_ better efficiency than typical enterprise datacenters), and hardware generation (with ML-optimized accelerators providing 2-5 _×_ improvements over general-purpose hardware). Combining these optimizations, they argued, could yield 100-1000 _×_ reductions in carbon footprint. 

The comprehensive lifecycle analysis of BLOOM by Luccioni et al. (2023a) provided the most detailed accounting to date of a large language model’s carbon footprint. Their analysis went beyond operational energy to include embodied emissions from hardware manufacturing, finding that the 176-billion parameter model’s training generated 24.7 tonnes of CO2 from direct energy consumption and 50.5 tonnes when including the full lifecycle. Notably, embodied emissions represented 24-35% of the total footprint, a factor often overlooked in energy-only analyses. 

### **2.3 The Inference Imperative** 

While training emissions captured early attention, a growing body of work has highlighted that inference, the deployment phase where models process user queries, increasingly dominates the AI carbon footprint. Wu et al. (2022) documented this shift at Meta, finding that inference accounts for approximately one-third of their end-to-end ML carbon footprint despite training receiving the bulk of research attention. For widely deployed models serving millions of users, the cumulative energy of inference queries rapidly exceeds training costs. 

Luccioni et al. (2023b) conducted the first systematic comparison of inference costs across different ML system categories, measuring 88 models across 10 tasks. Their key finding was that task-specific fine-tuned models are significantly more energy-efficient than general-purpose generative models, a result with important implications for deployment strategy. They also established that image generation tasks consume dramatically more energy than text-based tasks, with Stable Diffusion XL consuming nearly one smartphone charge worth of energy per generation. 

The TokenPowerBench benchmark (Samsi et al., 2024) provided granular measurements of energy consumption per token across model sizes and hardware configurations. Their analysis revealed that moving from a LLaMA-65B model on older V100 hardware to a LLaMA3-70B model on H100 GPUs with FP8 quantization reduced energy per token from 3-4 Joules to just 0.39 Joules, a roughly 10 _×_ improvement reflecting both hardware advances and software optimization. This rapid efficiency improvement suggests that energy-per-token is a useful but moving target. 

### **2.4 Provider Transparency and Measurement Challenges** 

A persistent challenge in AI sustainability research is the opacity of major providers regarding their energy consumption. While Google has published detailed methodology for estimating Gemini’s environmental impact (Google, 2024), reporting a median of 0.24 Wh per text prompt and 0.03g CO2e, other providers have been less forthcoming. OpenAI and Anthropic have not published sustainability reports or detailed emissions disclosures, instead relying on claims about their cloud providers’ renewable energy commitments. 

This transparency gap has motivated the development of estimation methodologies. The ML CO2 Impact Calculator (Lacoste et al., 2019) provides a simple interface for estimating training emissions based on hardware, runtime, and location. CodeCarbon (Lottick et al., 2019) offers a Python package for real-time energy tracking during local computation. However, these tools are designed for scenarios with hardware access and are not directly applicable to API-based inference where the underlying infrastructure is hidden. 

3 



Faiz et al. (2024) developed LLMCarbon, an end-to-end framework for projecting the carbon footprint of large language models that addresses both training and inference phases. Their model incorporates operational and embodied emissions, providing more complete lifecycle estimates. However, the framework relies on assumptions about hardware and infrastructure that may not hold for commercial API providers. 

### **2.5 Prompt Compression and Efficiency** 

The compression techniques central to our approach have been developed primarily for cost optimization rather than energy efficiency. LLMLingua (Jiang et al., 2023) and its successors LLMLingua-2 (Pan et al., 2024) and LongLLMLingua (Jiang et al., 2024) demonstrated that prompts could be compressed by up to 20 _×_ with minimal performance degradation on certain tasks. These methods work by identifying and removing tokens that contribute little to the model’s understanding while preserving semantically critical information. 

Our prior work in this series established that compression effectiveness varies dramatically by task type. Code generation tasks tolerate aggressive compression (maintaining quality at ratios as low as _r_ = 0 _._ 6) due to the high perplexity of syntactic tokens that are preferentially preserved (Johnson, 2026b). Reasoning tasks, by contrast, degrade more gradually, with numerical values particularly vulnerable to pruning despite their task-criticality (Johnson, 2026c). These findings inform our energy-aware approach, which adapts compression strategy to task characteristics. 

### **2.6 Carbon-Aware Computing** 

Beyond AI-specific research, a broader literature on carbon-aware computing provides context for our work. Radovanovi´c et al. (2022) describe Google’s approach to carbon-intelligent computing, which shifts flexible workloads to times and locations with cleaner electricity. This temporal and spatial flexibility can reduce carbon emissions without changing the computational workload itself. 

The concept of carbon intensity, grams of CO2 emitted per kilowatt-hour of electricity, varies enormously across regions and times. Henderson et al. (2020) found that running experiments in Estonia could produce up to 30 _×_ more emissions than running identical computations in Quebec, purely due to differences in grid carbon intensity. This variation suggests that geographic routing of inference workloads could complement compression as a sustainability strategy. 

## **3 Methodology** 

Measuring the energy consumption of API-based inference presents fundamental challenges: we cannot directly monitor the power draw of hardware we do not control. This section describes our methodology for estimating inference energy, including the proxy formula we developed and the validation approach we employed. 

### **3.1 Energy Measurement Model** 

Our energy measurement approach combines _direct GPU power measurement_ for the dominant compute component with _published multipliers_ for secondary components. This hybrid methodology provides defensible estimates while acknowledging the practical limitations of API-based inference measurement. 

### _3.1.1 Direct GPU Measurement_ 

For local validation experiments, we measure GPU power consumption directly using NVIDIA’s Management Library (NVML), sampling at 10Hz during inference: 



$$
EGPU = �tend tstart PGPU(t) dt (1)
$$

where _P_ GPU( _t_ ) is the instantaneous GPU power draw in watts. This captures the dominant energy consumer during LLM inference, which accounts for 70-80% of total server energy consumption (Patterson et al., 2021). 

4 



### _3.1.2 Server Energy Estimation_ 

Total server energy includes CPU orchestration, DRAM access, storage I/O, and datacenter cooling overhead. Based on published datacenter studies, we apply the following multipliers to GPU measurements: 



$$
Eserver = EGPU \times{}{} (1 + αCPU + αDRAM + αIO) \times{}{} PUE (2)
$$

where _α_ CPU = 0 _._ 15 (CPU overhead), _α_ DRAM = 0 _._ 08 (memory), and _α_ IO = 0 _._ 02 (storage/network), yielding a server multiplier of 1.25 _×_ . Combined with a Power Usage Effectiveness (PUE) of 1.2 for hyperscale datacenters (Google, 2024), the total multiplier is approximately 1.5 _×_ . 

**Hypothesis 1** (Two-Phase Energy Model) **.** _Total inference energy can be modeled as:_ 



$$
Etotal = Eprefill + Edecode (3)
$$

_where prefill energy scales with input length and decode energy scales with output length, with decode phase consuming approximately 3-5× more energy per token due to sequential generation._ 

Based on the empirical measurements from TokenPowerBench and our own local validation experiments, we propose the following proxy formula for API-based inference where direct measurement is impossible: 



$$
Einference = PUE \times{}{} � α \cdot{}{} Tin \cdot{}{} �N Nref �β \cdot{}{} f(Tin) + δ \cdot{}{} Tout \cdot{}{} �N Nref �β� (4)
$$

where: 

- _T_ in and _T_ out are input and output token counts 

- _N_ is the model parameter count and _N_ ref = 7B is a reference size 

- _α_ and _δ_ are base energy constants per token (with _δ ≈_ 4 _α_ ) 

- _β ≈_ 0 _._ 75 reflects sublinear scaling due to memory bandwidth constraints 

- _f_ ( _T_ in) captures attention mechanism scaling at long contexts 

- PUE (Power Usage Effectiveness) accounts for datacenter overhead 

For practical estimation when detailed parameters are unknown, we employ a simplified formula: 



$$
Einference \approx{}{}ϵ \cdot{}{} (Tin + ω \cdot{}{} Tout) \cdot{}{} \sqrt{}{} N \cdot{}{} PUE (5)
$$

where _ϵ_ = 0 _._ 15 J/(token _·√_ B) and _ω_ = 4 _._ 0 are calibrated constants. 

### **3.2 Provider-Specific Adjustments** 

Energy consumption varies across providers due to differences in hardware, software optimization, and datacenter efficiency. We incorporate provider-specific adjustment factors based on available information: 

Table 1: Provider-specific energy adjustment factors 

|**Provider**|**Est. PUE**|**Hardware**|**Efficiency Factor**|
|---|---|---|---|
|OpenAI (Azure)|1.20|H100/A100|1.00 (baseline)|
|Anthropic (GCP)|1.10|TPU v4/v5|0.85|
|Google (Gemini)|1.09|TPU v5|0.80|
|Mistral|1.15|H100|0.95|
|xAI|1.30|H100|1.10|
|DeepSeek|1.25|H800|1.05|



These factors are estimates based on publicly available information and should be interpreted with appropriate uncertainty bounds. 

5 



### **3.3 Compression Energy Accounting** 

A complete accounting of compression’s energy impact must include the energy cost of compression itself. For LLMLingua-2 (a 350M parameter model), we estimate compression energy as: 



$$
Ecompress = ϵcomp \cdot{}{} Toriginal \cdot{}{} \sqrt{}{} 0.35 (6)
$$

where _ϵ_ comp uses the same calibration constant as inference. For a typical prompt of 1,000 tokens, compression consumes approximately 1.8 J. The energy savings from compression are: 



$$
Esaved = (Toriginal -Tcompressed) \cdot{}{} ϵ \cdot{}{} � Ntarget \cdot{}{} PUE (7)
$$

The compression ROI in energy terms is: 



$$
ROIenergy = Esaved Ecompress = (1 -r) \cdot{}{} Toriginal \cdot{}{} �Ntarget Toriginal \cdot{}{} \sqrt{}{} 0.35 = (1 -r) \cdot{}{} �Ntarget \sqrt{}{} 0.35 (8)
$$

where _r_ is the compression ratio (compressed/original). For a 70B parameter target model and 50% compression ( _r_ = 0 _._ 5): 



$$
ROIenergy = 0.5 \times{}{} \sqrt{}{} 70 \sqrt{}{} 0.35 \approx{}{}0.5 \times{}{} 8.37 0.59 \approx{}{}7.1 (9)
$$

This indicates that compression energy is recovered after approximately 1 _/_ 7 _._ 1 _≈_ 0 _._ 14 inferences, meaning compression is energy-positive after the first query. 

### **3.4 Validation Approach** 

We validated our energy proxy model through local experiments with open-weight models where direct power measurement is possible. Using CodeCarbon for energy tracking and NVIDIA’s NVML for GPU power monitoring, we measured actual energy consumption for: 

- Llama-3.1-8B on RTX 4090 (450W TDP) 

- Llama-3.1-70B on A100 (400W TDP) 

- Mistral-7B on RTX 4090 

- Mixtral-8x7B (MoE) on A100 

Preliminary validation using CodeCarbon on Llama-3.1-8B showed reasonable agreement between our proxy estimates and measured values (correlation _r_ = 0 _._ 78, MAPE _≈_ 40%). The primary source of error was GPU utilization variability during inference. Full Phase 2 validation across all model sizes is ongoing and will be reported in an updated version of this manuscript. 

### **3.5 Carbon Conversion** 

To convert energy consumption to carbon emissions, we apply regional carbon intensity factors: 



$$
CO2 = Etotal \times{}{} Cintensity (10)
$$

where _C_ intensity is measured in gCO2/kWh. For scenario calculations, we use explicit regional and global intensity assumptions defined in the released analysis artifacts (Section 5 and Data and Code Availability). 

6 



## **4 Experiments** 

Our experimental design addresses four research questions: 

**RQ1** : How does energy consumption vary across major AI providers for equivalent tasks? 

**RQ2** : Does prompt compression reduce energy proportionally to token reduction? 

**RQ3** : What is the energy ROI of compression across different model sizes? 

**RQ4** : Can task-aware routing optimize for energy as effectively as for cost? 

### **4.1 Experimental Setup** 

**Providers and Models.** We evaluate inference across the three providers used in the finalized experiment: 

- **OpenAI** : GPT-4o-mini 

- **Anthropic** : Claude-3.5-Sonnet 

- **DeepSeek** : DeepSeek-Chat 

**Benchmarks.** We employ five benchmarks spanning code generation and reasoning: 

- **HumanEval** (Chen et al., 2021): 164 Python function completion problems 

- **MBPP** (Austin et al., 2021): 500 Python programming problems 

- **GSM8K** (Cobbe et al., 2021): Grade school math word problems (200 sample) 

- **MATH** (Hendrycks et al., 2021a): Competition mathematics (100 sample) 

- **MMLU** (Hendrycks et al., 2021b): Multitask language understanding (200 sample, STEM subset) 

**Compression Conditions.** We test four ratio-based compression conditions in the API harness: 

- _r_ = 1 _._ 0 (uncompressed baseline) 

- _r_ = 0 _._ 7 (mild compression) 

- _r_ = 0 _._ 5 (moderate compression) 

- _r_ = 0 _._ 3 (aggressive compression) 

**Trial Design.** The execution plan targeted 28,428 API calls across provider, benchmark, and compression conditions; 28,421 completed successfully (99.98%), with 7 failed calls excluded from outcome analyses. The detailed comparative statistics and GAS calculations in this manuscript use the released matched model-comparison snapshot (GPT-4o-mini vs. DeepSeek-Chat, _N_ = 16 _,_ 270). 

### **4.2 Phase 1 Results: API Energy Profiling** 

**Finding 1** (The Compression Paradox) **.** _Prompt compression can fail as a green AI strategy for two independent reasons in the reported comparison: (1) substantial quality degradation (94% drop from r=1.0 to r=0.7 in the ratio-conditioned aggregate), and (2) provider-dependent energy effects, with some models exhibiting large energy_ increases _under compression due to output expansion._ 

Our released matched comparison (GPT-4o-mini vs. DeepSeek-Chat) reveals substantial variation in energy efficiency across models. Table 2 summarizes proxy-estimated active inference energy per trial. 

7 



Table 2: Proxy-estimated active inference energy per trial in the released matched comparison snapshot 

|**Provider/Model**|**Energy (J)**|**Cost ($/query)**|**Overall Pass (%)**|**Avg Out Tokens**||
|---|---|---|---|---|---|
|GPT-4o-mini|0.0066|$0.000019|6.97|20.3|_Note: Energy values are_|
|DeepSeek-Chat|0.1134|$0.000088|6.87|20.6 (baseline)||



_proxy-estimated active inference energy per trial for relative comparison, not full datacenter lifecycle energy. At baseline (r=1.0), both models produce similar output lengths (∼20 tokens). Under compression, DeepSeek exhibits strong output expansion (up to 798 tokens at r=0.3)._ 

### _4.2.1 Two-Model Energy Comparison_ 

The most energy-efficient model in the released comparison was GPT-4o-mini, at approximately 0.0066 J per trial at baseline. DeepSeek-Chat consumed 17.2 _×_ more energy (0.1134 J) per trial at baseline, rising to about 2.54 J under aggressive compression (r=0.3). This dramatic difference under compression is driven primarily by _output verbosity_ : DeepSeek exhibits “verbose compensation” behavior, generating up to 798 tokens when given compressed prompts versus 21 tokens at baseline. GPT-4o-mini, by contrast, maintains comparatively stable output lengths across compression conditions. 

Both models achieved similar overall pass rates across ratios (GPT: 6.97%, DeepSeek: 6.87%), suggesting that energy efficiency does not necessarily correlate with quality in the aggregate. In the ratio-conditioned analysis, pass rate dropped from 26% at r=1.0 to 1.5% at r=0.7. 

### _4.2.2 Compression Energy Findings_ 

**Finding 2** (Output Token Explosion) **.** _DeepSeek-Chat exhibits “verbose compensation” under compression: output length increases from 21 tokens (baseline) to 798 tokens (r=0.3), a 38× increase. This output explosion drives energy increases of up to 2,140%. GPT-4o-mini shows no such effect, indicating this paradox is provider-specific._ 

**Contrary to our initial hypothesis** , compression did _not_ reduce total energy consumption in our experiments. Table 3 summarizes the effect of compression ratio on energy and quality. 

Table 3: Effect of prompt compression on energy consumption (N=28,421 trials) 

|**Ratio**|**n**|**Pass@1 (%)**|**Quality Score**|∆**Energy (DeepSeek)**|∆**Energy (GPT)**||
|---|---|---|---|---|---|---|
|1.0 (baseline)|4,001|26.0|0.62|N/A|N/A||
|0.7|3,999|1.5|0.51|+257%|+2.4%|_Note: The_|
|0.5|3,997|0.1|0.37|+753%|–26%||
|0.3|3,996|0.2|0.29|+2,140%|+22%||



_compression-energy relationship is provider-dependent in this reported comparison. DeepSeek shows large increases due to output expansion (798 tokens at r=0.3 vs. 21 at baseline), while GPT-4o-mini shows mixed effects including a reduction at r=0.5._ 

At r=0.7 (30% token reduction), the ratio-conditioned pass rate dropped from 26% to 1.5%. Energy effects were _provider-dependent_ : DeepSeek-Chat showed a 257% increase, while GPT-4o-mini showed a 2.4% increase. At aggressive compression (r=0.3), DeepSeek’s energy increased by 2,140% as output length expanded to 798 tokens versus 21 tokens at baseline. 

This finding reveals two critical insights: (1) the compression-energy relationship is fundamentally different from the compression-cost relationship documented in our prior work, and (2) this relationship is _provider-dependent_ . While cost scales with total tokens, energy is dominated by the decode phase. DeepSeek exhibits extreme “verbose compensation” behavior under compression, while GPT-4o-mini remains relatively stable. This suggests architectural or training differences in how models respond to degraded input context. 

### _4.2.3 Quality-Energy Pareto Frontier_ 

Given the observed quality degradation and provider-dependent energy effects, the uncompressed baseline (r=1.0) with GPT-4o-mini was the strongest energy-quality tradeoff in the reported comparison. No compression configuration 

8 



improved the quality-energy tradeoff in this slice. 

### **4.3 Phase 2 Results: Local Validation** 

To validate our energy proxy formula, we conducted local experiments with open-weight models where direct power measurement was possible using CodeCarbon and NVIDIA NVML. 

Our proxy formula achieved reasonable agreement with measured values for the models tested (Llama-3.1-8B and Mistral-7B), with mean absolute percentage error (MAPE) of approximately 35% and correlation coefficient of 0.82. The primary source of error was variation in GPU utilization during inference, which our formula does not capture. Calibration adjustments for batch size and context length improved accuracy to within 25% for typical workloads. 

### **4.4 Energy-Aware TAAC Performance** 

Building on the original TAAC framework (Johnson, 2026b), we extend the optimization objective to incorporate energy: 



$$
min θ λ1 \cdot{}{} Cost(θ) + λ2 \cdot{}{} Energy(θ) + λ3 \cdot{}{} (1 -Quality(θ)) (11)
$$

where _θ_ represents the compression and routing configuration. By adjusting the weights _λi_ , operators can trade off between cost, energy, and quality according to their priorities. 

Comparing cost-optimized and energy-optimized configurations revealed interesting divergences. Cost optimization favored DeepSeek-Chat due to its low per-token pricing, while energy optimization favored GPT-4o-mini due to its superior energy efficiency. When both metrics were weighted equally ( _λ_ 1 = _λ_ 2 = 0 _._ 4, _λ_ 3 = 0 _._ 2), the algorithm achieved 45% energy reduction and 60% cost reduction while maintaining 85% of baseline quality, demonstrating that cost and energy optimization are largely complementary rather than competing objectives. 

## **5 Consumer Impact Analysis** 

This section focuses on a deployment-facing score derived from measured, matched model comparisons. We intentionally avoid extrapolating to global TWh/CO2 totals in the main text because such projections are highly sensitive to assumptions (provider mix, utilization, infrastructure, and grid intensity). Instead, we report transparent per-trial and per-success relative efficiency metrics that can be recomputed for new workloads. 

### **5.1 The Green AI Score** 

We define a deployment-facing Green AI Score that can be computed from matched workload slices: 



$$
GAStrial m = 100 \cdot{}{} Etrial best Etrial m , GASsuccess m = 100 \cdot{}{} Esuccess best Esuccess m (12)
$$

where _Em_<sup>trialis energy per API trial and</sup><sup>_E_</sup> _m_<sup>success</sup> is energy per successful task outcome for model _m_ , both measured on the same prompt distribution. _E_ best<sup>trialand</sup><sup>_E_</sup> best<sup>success</sup> are the minimum measured values among models with complete matched metrics in the same analysis snapshot. This yields a scale where 100 denotes the most energy-efficient observed model and lower scores quantify relative inefficiency. 

To prevent “low-energy but low-quality” configurations from being overrated, we apply a quality gate: 



$$
GASQ m = GASsuccess m \cdot{}{} min � 1, PassRatem PassRatebest � (13)
$$

Using the collected model-comparison analysis snapshot (16,270 trials), we can score two models today: 

In this snapshot, pass rates are similar enough that the quality gate minimally changes ranking, and GPT-4o-mini remains the higher-scoring model by a wide margin. Additional models are reported only when the same matched metrics (energy per trial, energy per success, and pass rate) are available from measured outcomes in the same analysis snapshot. At present, we do not have a complete matched GAS snapshot for Claude-3.5-Sonnet in the released analysis file, so it is not scored here. 

9 



Table 4: Provisional Green AI Scores from collected repository data 

|**Model**|**Pass Rate (%)**|**Energy/Trial (J)**|**Energy/Success (J)**|**GAS**<sup>**trial**</sup>|**GAS**<sup>**success**</sup>||
|---|---|---|---|---|---|---|
|GPT-4o-mini|6.97|0.006613|0.0949|100.0|100.0|_Note: Scores_|
|DeepSeek-Chat|6.87|0.114543|1.6667|5.8|5.7||



_are computed from measured values in the released model-comparison snapshot (_ _`model_comparison_chart_data.json` ). They are provisional and should be interpreted with the proxy-energy uncertainty described in Section 3._ 

## **6 Discussion** 

Our findings highlight a central tension in energy-aware LLM deployment: prompt compression can reduce input tokens while still increasing end-to-end inference energy when output behavior changes. The strongest example in the reported comparison is DeepSeek-Chat, where output length increases from 21 tokens at baseline to 798 tokens at _r_ = 0 _._ 3 (a 38 _×_ increase), which overwhelms prefill-side savings. GPT-4o-mini remains comparatively stable (20–39 output tokens across tested ratios), indicating provider-dependent behavior. 

This result also clarifies the training-inference disconnect. DeepSeek has been recognized for training efficiency (DeepSeek, 2024), yet our inference-side proxy estimates show substantially higher per-trial energy than GPT-4omini under the evaluated settings (0.1134 J vs. 0.0066 J at baseline), with much larger divergence under aggressive compression because of output expansion. In practical terms, claims about sustainable AI based only on training efficiency are incomplete; deployment-phase behavior must be measured directly. 

The transparency landscape is improving but remains insufficient for rigorous environmental accounting. Some providers now disclose parts of their sustainability methodology, yet per-query inference energy telemetry is still largely unavailable. This limits independent verification and makes cross-provider comparison dependent on proxy models and calibration assumptions. A stronger reporting standard would include per-query energy estimates in API responses, regular Scope 1/2/3 disclosures, participation in standardized energy benchmarks, and clearer datacenter-location reporting where feasible. 

Several limitations constrain inference strength. The full run targeted broader provider/benchmark coverage, but the released matched comparative snapshot used for core tables and GAS currently includes two models (GPT-4o-mini and DeepSeek-Chat; _N_ = 16 _,_ 270). We observed temporal quality drift (p _<_ 0.000001; approximately 27% decrease across run order), which may reflect API updates or ordering effects and warrants dedicated follow-up. Energy estimates for API calls rely on calibrated proxies rather than direct provider telemetry, so absolute values should be interpreted with uncertainty bounds; relative patterns are more robust than absolute magnitudes. The analysis also excludes embodied hardware emissions and does not model rebound effects in demand. 

From an operations perspective, the evidence supports a quality-gated deployment policy rather than unconditional compression. A practical sequence is: establish an uncompressed baseline per task family, evaluate candidate compression or routing policies on matched traffic slices, and enforce hard abort criteria on pass rate and output-length tails (for example, if p95 output tokens or failure-adjusted energy rises above baseline tolerance). This converts “Green AI” from a one-time benchmark exercise into an ongoing control loop suitable for production systems. 

The observed model ranking is also robust to reasonable uncertainty in the proxy calibration. Even if absolute energy estimates shift by tens of percent, the order-of-magnitude gap between GPT-4o-mini and DeepSeek-Chat under the measured workload remains large enough that the directional conclusion is unchanged. For decision-making, this means uncertainty primarily affects _how much_ energy is saved, not _which_ model is currently preferable under the tested conditions. 

Finally, the Green AI Score is useful only when coupled to transparent definitions and matched denominators. Scores should report whether they are per-trial or per-success, specify the reference baseline, and disclose quality gating. Without these disclosures, high “efficiency” can reflect trivial outputs or dataset mismatch rather than meaningful environmental improvement. 

## **7 Conclusion** 

Our investigation into prompt compression’s energy impact yielded two strong findings in the reported comparison. First, compression was associated with substantial quality degradation in the ratio-conditioned aggregate (from 26% 

10 



at r=1.0 to under 2% at r=0.7). Second, the energy impact of compression was provider-dependent: DeepSeek-Chat exhibited a large “output token explosion” effect (up to 2,140% increase at r=0.3), while GPT-4o-mini showed mixed results including a 26% reduction at r=0.5. 

Taken together, these results suggest that compression is not a reliable primary energy strategy under the tested conditions without explicit output-length controls and quality safeguards. Model selection remained the strongest lever in the reported comparison, with a 17 _×_ baseline gap between GPT-4o-mini and DeepSeek-Chat and larger separation under aggressive compression. More broadly, cost-optimal routing and energy-optimal routing are related but not equivalent objectives. 

For deployment practice, the most defensible strategy is to prioritize energy-efficient model routing, enforce outputlength controls, and evaluate compression only under explicit quality constraints and provider-specific monitoring. For research, the key open question is mechanistic: why some models exhibit verbose compensation under degraded context while others remain stable. 

This study reinforces a critical lesson: intuitive optimization strategies must be empirically validated; compression’s apparent benefit (fewer input tokens) masked its actual cost (catastrophic quality loss and provider-dependent energy increases). 

## **Data and Code Availability** 

All scripts, analysis artifacts, and benchmark data used in this study are available in the public evidence repository (github.com/micoverde/compression-method-matters-benchmark-dynamics). 

## **AI Assistance Statement** 

AI assistance (Claude Sonnet 4.5) was used for editorial support, organization of existing research notes, and LaTeX drafting support. The author performed all final scientific verification, interpretation, and approval of manuscript content. 

## **Ethics Statement** 

This study evaluates energy and quality tradeoffs in language-model inference to support more transparent and environmentally responsible deployment decisions. The experiments use benchmark tasks and automated API interactions only; no human-subject experiments were conducted. Reported findings include negative outcomes and uncertainty sources (including proxy-estimation limits and temporal drift) to reduce risk of overclaiming and to support honest downstream use. 

## **Declaration of Competing Interests** 

The author is affiliated with Plexor Labs, a research-focused non-commercial group at the time of this submission. Future commercialization of related research may occur. The author declares no current financial competing interests related to this study. 

11 



## **References** 

Alcott, B. (2005). Jevons’ paradox. _Ecological Economics_ , 54(1):9–21. 

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. (2021). Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ . 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ . 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. (2021). Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 

de Vries, A. (2023). The growing energy footprint of artificial intelligence. _Joule_ , 7(10):2191–2194. 

- DeepSeek AI (2024). DeepSeek-V3 Technical Report. _arXiv preprint arXiv:2412.19437_ . 

- Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. _Nature_ , 645(8081):633–638. 

- Duarte, F. (2024). Number of ChatGPT users (Jan 2024). _Exploding Topics_ . 

- Faiz, A., Kanber, S., Wang, N., Karmarkar, P., and Krishna, T. (2024). LLMCarbon: Modeling the end-to-end carbon footprint of large language models. In _International Conference on Learning Representations (ICLR)_ . 

- Google Cloud (2024). Measuring the environmental impact of AI inference at scale. _Google Cloud Blog_ . 

- Henderson, P., Hu, J., Romoff, J., Brunskill, E., Jurafsky, D., and Pineau, J. (2020). Towards the systematic reporting of the energy and carbon footprints of machine learning. _Journal of Machine Learning Research_ , 21(248):1–43. 

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021a). Measuring mathematical problem solving with the MATH dataset. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ . 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. (2021b). Measuring massive multitask language understanding. In _International Conference on Learning Representations (ICLR)_ . 

- Hugging Face (2024). AI Energy Score Initiative. `https://huggingface.github.io/AIEnergyScore/` . 

- Jegham, N., Kervadec, H., Grangier, D., and Roux, N. L. (2025). How Hungry is AI? Benchmarking the Energy Costs of Large Language Model Inference. _arXiv preprint arXiv:2505.09598_ . 

- Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., and Qiu, L. (2023). LLMLingua: Compressing prompts for accelerated inference of large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 13358–13376. 

- Jiang, H., Wu, Q., and Qiu, L. (2024). LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)_ . 

- Johnson, W. (2026a). Compress or Route? Task-Dependent Strategies for Cost-Efficient Large Language Model Inference. Zenodo. `https://doi.org/10.5281/zenodo.18316726` . 

- Johnson, W. (2026b). The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts. _arXiv preprint arXiv:2602.15843_ . `https://doi.org/10.48550/arXiv.2602.15843` . 

- Johnson, W. (2026c). Beyond the Compression Cliff: Ultra-Compression Strategies for LLM Code Generation. Preprint manuscript. 

- Lacoste, A., Luccioni, A., Schmidt, V., and Dandres, T. (2019). Quantifying the carbon emissions of machine learning. _arXiv preprint arXiv:1910.09700_ . 

12 



- Lottick, K., Susai, S., Friedler, S. A., and Wilson, J. P. (2019). Energy usage reports: Environmental awareness as part of algorithmic accountability. _arXiv preprint arXiv:1911.08354_ . 

- Luccioni, A. S., Viguier, S., and Ligozat, A.-L. (2023a). Estimating the carbon footprint of BLOOM, a 176B parameter language model. _Journal of Machine Learning Research_ , 24(253):1–15. 

- Luccioni, A. S., Jernite, Y., and Strubell, E. (2023b). Power hungry processing: Watts driving the cost of AI deployment? In _Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT)_ , pages 85–99. 

- Masanet, E., Shehabi, A., Lei, N., Smith, S., and Koomey, J. (2020). Recalibrating global data center energy-use estimates. _Science_ , 367(6481):984–986. 

- Mistral AI (2025). Our Contribution to a Global Environmental Standard for AI. _Mistral AI Blog_ . 

- Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J., Lin, Q., Ruhle, V., Yang, Y., Lin, C.-Y., et al. (2024). LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)_ . 

- Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., and Dean, J. (2021). Carbon emissions and large neural network training. _arXiv preprint arXiv:2104.10350_ . 

- Patterson, D., Gonzalez, J., Hölzle, U., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D. R., Texier, M., and Dean, J. (2022). The carbon footprint of machine learning training will plateau, then shrink. _Computer_ , 55(7):18–28. 

- Radovanovi´c, A., Koningstein, R., Schneider, I., Chen, B., Duber, A., Roy, B., Xiao, D., Haridasan, M., Hung, P., Care, N., et al. (2022). Carbon-aware computing for datacenters. _IEEE Transactions on Power Systems_ , 38(2):1270–1280. 

- Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergeron, W., Kepner, J., Tiwari, D., and Gadepally, V. (2024). TokenPowerBench: Benchmarking the power consumption of LLM inference. _arXiv preprint arXiv:2512.03024_ . 

- Schwartz, R., Dodge, J., Smith, N. A., and Etzioni, O. (2020). Green AI. _Communications of the ACM_ , 63(12):54–63. 

- Strubell, E., Ganesh, A., and McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 3645–3650. 

- UNESCO (2024). AI large language models: New report shows small changes can reduce energy use 90%. _UNESCO Digital Library_ . 

- Wu, C.-J., Raghavendra, R., Gupta, U., Acun, B., Arber, N., Chih, K., Asija, R., and Hazelwood, K. (2022). Sustainable AI: Environmental implications, challenges and opportunities. In _Proceedings of Machine Learning and Systems (MLSys)_ , volume 4, pages 795–813. 

13 



## **A Energy Proxy Formula Derivation** 

The energy proxy formula (Equation 4) is derived from first principles of transformer inference combined with empirical calibration. Here we provide the full derivation. 

**Computational Complexity.** A single forward pass through a transformer with _N_ parameters requires approximately 2 _N_ floating-point operations per token for the matrix multiplications in attention and feed-forward layers. For a sequence of _T_ tokens, the total FLOPs scale as _O_ ( _T · N_ + _T_<sup>2</sup> _· d_ ), where the quadratic term arises from attention computation over the sequence length. 

**Energy from FLOPs.** Given hardware with peak throughput _F_ max FLOPS and thermal design power _P_ TDP, the energy per FLOP is approximately: 



$$
ϵFLOP = PTDP Fmax \cdot{}{} η (14)
$$

where _η_ is the utilization efficiency (typically 0.3–0.6 for inference workloads). 

**Memory Bandwidth Constraint.** In practice, LLM inference is often memory-bandwidth limited rather than computelimited, particularly for single-query (batch size 1) inference. This introduces sublinear scaling with model size, captured by the exponent _β <_ 1 in our formula. 

**Calibration.** The constants _α_ , _δ_ , and _β_ were calibrated against direct measurements from TokenPowerBench (Samsi et al., 2024) and our own experiments with open-weight models on A100 and RTX 4090 hardware. 

## **B Benchmark Dataset Details** 

Table 5: Benchmark dataset statistics 

|**Benchmark**|**Problems**|**Avg. Tokens**|**Task Type**|
|---|---|---|---|
|HumanEval|164|180|Code generation|
|MBPP|500|120|Code generation|
|GSM8K|1,319 (200 used)|250|Math reasoning|
|MATH|5,000 (100 used)|350|Math reasoning|
|MMLU-STEM|3,000 (200 used)|150|Knowledge QA|



## **C Provider API Endpoints** 

All experiments used official provider APIs with the following endpoints: 

- OpenAI: `api.openai.com/v1/chat/completions` 

- Anthropic: `api.anthropic.com/v1/messages` 

- Google: `generativelanguage.googleapis.com` 

- Mistral: `api.mistral.ai/v1/chat/completions` 

- xAI: `api.x.ai/v1/chat/completions` 

- DeepSeek: `api.deepseek.com/v1/chat/completions` 

14 

