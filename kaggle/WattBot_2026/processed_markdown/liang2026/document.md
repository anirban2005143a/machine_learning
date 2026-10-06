1 

# **GPU-to-Grid: Voltage Regulation via GPU Utilization Control** 

Zhirui Liang, Jae-Won Chung, Mosharaf Chowdhury, Jiasi Chen, Vladimir Dvorkin 

**_Abstract_ —While the rapid expansion of data centers poses challenges for power grids, it also offers new opportunities as flexible loads. Existing power system research often abstracts data centers as aggregate resources, while computer system research focuses on GPU energy efficiency and largely ignores grid impacts. To bridge this gap, we develop a GPU-to-Grid framework that couples device-level GPU control with power system objectives. We study distribution-level voltage regulation enabled by LLM inference flexibility, using batch size as a datacenter-side control knob that trades off GPU power consumption, inference latency, and token throughput. We first formulate the problem as an optimization problem and then realize it as an online feedback optimization controller, implemented by the data center operator using its own empirical GPU power-performance model and real-time measurements from both the GPU and grid systems. Our key insight is that reducing GPU power alleviates lower-voltage violations, while increasing GPU power mitigates upper-voltage violations; this challenges the common belief that minimizing GPU power is always beneficial to power grids.**<sup>1</sup> 

**_Index Terms_ —Distribution voltage regulation, data centers, GPU flexibility, batch size, online feedback optimization.** 

## I. INTRODUCTION 

The rapid expansion of AI workloads is driving a sharp rise in data center electricity demand and GPU power density. Globally, data centers consumed about 415 TWh in 2024 and are projected to more than double by 2030, with AI as a key growth driver [3]. At the device level, state-of-the-art accelerators already approach server-scale power intensities (e.g., NVIDIA’s H100 SXM5 GPU specifies up to 700 W per GPU [4]), so large GPU clusters can add multi-megawatt loads over short deployment timelines. This surge creates significant challenges for power-system operation and planning: load growth is geographically concentrated, often constrained by latency and reliability requirements, and can substantially change regional demand trajectories [5]. 

At the same time, grid-connected data centers also offer new opportunities for power system operation. Similar to vehicle-to-grid (V2G) technologies [6], large-scale data centers can act as flexible demand-side resources by adjusting power consumption in response to grid conditions. Recent work in the power system literature has explored the use of data center flexibility for services such as peak shaving, frequency regulation, and voltage support [7]–[9]. However, most existing studies model data center flexibility at an aggregate level and do not explicitly capture how such flexibility is realized at the device level. In particular, the mechanisms by which GPU workloads provide controllable power adjustments, along with the associated inference latency and throughput constraints, are often abstracted or neglected. These GPU-level considerations are critical in practice, as they enable fine-grained and fast 

All authors are with the University of Michigan, Ann Arbor, MI, USA. 1Open-sourced as part of the OpenG2G library [1], [2]. 

power control while limiting the achievable magnitude and speed of power adjustments. 

From the computer systems perspective, significant effort has been devoted to improving the energy efficiency of GPUs through workload-aware control. Both training and inference energy consumption can be reduced by tuning available control knobs, such as GPU frequency scaling, power caps, and batch size selection, which directly affect utilization, throughput, and latency [10]–[12]. While these methods optimize energy or performance under fixed workload objectives, they do not account for grid needs. From the power system standpoint, there are operating conditions under which increased GPU power consumption is desirable, for example during periods of high renewable generation or when overvoltage arises in distribution networks. Enabling GPUs to act as grid-supportive resources therefore requires closing the loop between grid conditions and device-level control decisions. 

To bridge this gap, we propose a GPU-to-Grid (G2G) framework that couples device-level GPU control with grid-level feedback. The framework integrates models of GPU powerperformance trade-offs with real-time grid signals, enabling GPUs to operate as grid-supportive resources while respecting computing constraints. Such grid feedback signals may take the form of voltage measurements, frequency deviations, or price signals, depending on the grid service being provided. 

In this paper, we demonstrate the G2G framework using one grid service, distribution-level voltage regulation, and one GPU control knob, the batch size of LLM inference tasks. The controller is implemented by the data-center operator, which updates batch size locally in response to limited grid voltage and LLM latency measurements. Fig. 1 shows the overall architecture. Users submit stochastic inference requests to heterogeneous LLM models served by dedicated GPUs. The resulting batch-size decisions affect user latency, token throughput, GPU power consumption, and distributionnetwork voltages. Thus, at each control interval, the controller balances voltage constraints, latency requirements, and datacenter throughput objectives within the proposed G2G framework. 

Several studies have investigated grid services using devicelevel models of data center resources. For instance, Chen et al. [13] employ GPUs for voltage regulation via dynamic voltage and frequency scaling (DVFS), but assumes a linear relationship between GPU frequency and power and adopts a simple droop-based control that ignores inference latency and throughput constraints. In contrast, Colangelo et al. [14] demonstrate grid-interactive AI data centers in a field deployment, showing that workload control and DVFS can reduce power consumption during peak periods while maintaining quality of service; however, the grid feedback in that work is limited to high-level exogenous signals such as conges- 



2 

![Figure](assets/figure_0001_page_0002.svg)Fig. 1. GPU-to-Grid (G2G) framework for voltage regulation. The aggregated GPU behavior is simulated based on real measurement data in [15], and grid feedback is provided by the power system simulator (e.g., OpenDSS [16]). 

tion or peak-demand indicators, rather than physical states of power systems. This work addresses these limitations by using real GPU measurement data to model the nonlinear relationships between control knobs and performance metrics, and by proposing an online-feedback-optimization framework that explicitly incorporates voltage, latency, and throughput. By closing the loop with physical grid measurements and avoiding reliance on demand-response signals, the proposed approach enables fast distribution voltage regulation. 

## II. ANALYSIS OF GPU MEASUREMENT DATA 

## _A. Data Source and Inference Workload Characterization_ 

Our work is based on real measurements from software, hardware, and workloads that are representative of modern AI data center operations. Specifically, we used the ML.ENERGY Benchmark v3.0 data [12], [15], which provides detailed GPU power consumption, latency, and throughput measurements over time for various batch size<sup>2</sup> configurations of the large language model (LLM) inference server. 

Measurements were collected on vLLM [17] v0.11.1 on NVIDIA H100 80GB SXM5 GPUs connected with NVSwitch, both of which are representative of modern AI data centers. Workloads include dense Transformer [18]-based LLMs (Meta Llama 3.1 family [19]) responding to ChatGPT-style conversational queries and mixture-of-experts (MoE) [20] LLMs (Qwen 3 family [21]) answering challenging problems with reasoning, as summarized in Table I. The models used span a variety of architectures, tasks, sizes, number of GPUs, and parameter precisions, providing substantial diversity in power consumption and performance characteristics. 

Fig. 2 compares the GPU power consumption trajectories over time across different batch sizes for the Llama 3.1 405B and Qwen3 235B A22B models. Both models exhibit a consistent trend: larger batch sizes result in higher average GPU 

> 2In this work, batch size refers to the LLM inference server’s maximum batch size configuration, which is sustained during steady state request serving in a well-utilized datacenter. 

TABLE I 

LLMS STUDIED IN THIS PAPER 

|**Model name**|**Type**<br>**Active **|**params**<sup>_†_</sup>|**Precision**|**#GPUs**|
|---|---|---|---|---|
|Llama 3.1 8B|Dense|8B|BF16|1|
|Llama 3.1 70B|Dense|70B|BF16|4|
|Llama 3.1 405B|Dense|405B|FP8|8|
|Qwen3 30B A3B|MoE|3B|BF16|2|
|Qwen3 235B A22B|MoE|22B|BF16|8|



> _†_ MoE models dynamically activate only a subset of total parameters. 

![Figure](assets/figure_0002_page_0002.svg)power consumption. This shared property is the foundation of batch size-based control across model architectures. 

## _B. Tradeoff between Latency and Throughput_ 

Inference serving performance is commonly characterized by two key metrics: (1) latency, which quantifies the response time of individual inference requests, and (2) throughput, which measures the total number of requests completed per unit time. When it comes to LLMs, Inter-Token Latency (ITL) is a common latency metric, defined as the time spent to generate each output token after the previous one; a long ITL manifests as an AI chat service speaking very slowly, degrading user experience [22]. For throughput, token throughput is commonly reported, defined as the number of tokens generated per unit time; a low token throughput means that the server is not able to serve as many users at the same time, making users wait longer to get responses. 

In LLM serving, tokens are generated in _batches_ [23]; _b_ requests run inference together in the GPU, and when the _whole_ batch has completed execution by the GPU, each request gets one new token generated (thus _b_ new tokens are generated simultaneously). When the _batch size b_ is increased, the raw amount of computation needed to execute inference for that batch increases. This naturally takes more time for the GPU to complete, thereby increasing ITL. On the other hand, with a larger batch size, the GPU’s various software and hardware overheads are better amortized and the GPU’s utilization increases, making it capable of completing _more_ computations per unit time, increasing token throughput. A side effect of increased GPU utilization is increased power draw, as shown in Fig. 2. The relationship between ITL, token throughput, and batch size for a single LLM text generation iteration can be captured by the following equation: 

Batch Size (tokens) Inter-Token Latency (s) 

Token Throughput (tokens/s) = 



3 

Ideally, data center operators would want both high throughput and low latency. However, these objectives are inherently in tension because increasing batch size improves throughput while simultaneously increasing latency. We analyze and model this tradeoff relationship and the impact of batch size using real measurement data in Section II-C, and build our optimization model based on this relationship in Section III. 

## _C. Relationship between Performance Metrics and Batch Size_ 

We use GPU measurement data to empirically model the relationship between batch size _b_ and key performance metrics: (i) total GPU power consumption _p_ (in watts), (ii) mean intertoken latency _l_ (in seconds), and (iii) token throughput _r_ (in tokens per second). These relationships are represented using logistic functions, which capture the transition from underutilized to resource-saturated GPU operation as batch size increases. The logistic functions also provide inexpensive analytic gradients for online batch-size optimization. 

Define the logarithmic batch size variable _x_ ≜ log2( _b_ ). Then the power consumption, latency, and throughput are modeled directly as functions of _x_ : 



$$
p(x) = Pmax 1 + exp(-kp(x -x0,p)) + p0, (1) l(x) = Lmax 1 + exp(-kl(x -x0,l)) + l0, (2)
$$



$$
r(x) = Rmax 1 + exp(-kr(x -x0,r)) + r0, (3)
$$

where _P_ max, _L_ max, and _R_ max denote the saturation magnitudes of power consumption, latency, and throughput, respectively; _kp_ , _kl_ , and _kr_ control the sharpness of the transitions; _x_ 0 _,p_ , _x_ 0 _,l_ , and _x_ 0 _,r_ represent the characteristic batch size thresholds at which these transitions occur; and _p_ 0, _l_ 0, and _r_ 0 are offset terms. 

The fitted relationships for the three Llama models are shown in Fig. 3, while the fitting results for the two Qwen models, which exhibit similar trends, are provided in Fig. 10 in Appendix A. The models are fitted for the average GPU measurements over the entire observation horizon of each experiment in the ML.ENERGY benchmark dataset. 

As batch size increases, GPU power consumption rises monotonically and eventually saturates. Inter-token latency also increases with batch size, with a nonlinear growth as the system approaches saturation. In contrast, token throughput initially increases rapidly with batch size, but exhibits diminishing marginal gains at larger batch sizes. These trends are consistent across model scales, although larger models operate at higher power and latency levels and reach saturation at smaller batch sizes. Overall, Fig. 3 demonstrates batch size is an effective control knob that induces predictable trade-offs among power, latency, and throughput with nonlinear impact. 

## III. GRID- AND USER-AWARE BATCH SIZE OPTIMIZATION 

## _A. Batch Size Optimization Model_ 

This section formulates GPU batch size control as an optimization problem, beginning with the power system model with data center load. Consider a data center connected to 

![Figure](assets/figure_0003_page_0003.svg)Fig. 3. Fitted relationships between batch size and performance metrics for three models in the Llama 3.1 family [19]. 

a single node in a three-phase distribution network with _M_ buses. Let **v** _t_ ≜ [ **v** _t_<sup>_A,_</sup><sup>**v**</sup> _t_<sup>_B,_</sup><sup>**v**</sup> _t_<sup>_C_]</sup><sup>_⊤∈_R3</sup><sup>_M_denotethestacked</sup> three-phase voltage magnitudes, where **v** _t_<sup>_ϕ_</sup> _∈_ R<sup>_M_</sup> collects voltages on phase _ϕ ∈{A, B, C}_ . The phase-wise active and reactive power consumptions are **p** _t_ ≜ [ _p_<sup>_A_</sup> _t_<sup>_, pB_</sup> _t_<sup>_, pC_</sup> _t_<sup>]</sup><sup>_⊤_</sup> and **q** _t_ ≜ [ _qt_<sup>_A, q_</sup> _t_<sup>_B, q_</sup> _t_<sup>_C_]</sup><sup>_⊤_,andaconstantpowerfactorPFis</sup> assumed for all phases, such that _qt_<sup>_ϕ_=tan(arccos(PF))</sup><sup>_pϕ_</sup> _t_<sup>.</sup> The mapping from **p** _t_ and **q** _t_ to **v** _t_ is well established in power system modeling and simulation frameworks [24]. 

Assume the data center runs inference workloads for _N_ distinct LLM models. Model _i_ is deployed using _wi_ identical replicas, each assigned the same number of GPUs, and the vector **w** ≜ [ _w_ 1 _, . . . , wN_ ]<sup>_⊤_</sup> collects replica counts. Under replica-based scaling, the time-averaged total power consumption and aggregate token throughput of model _i_ scale approximately linearly with _wi_ , while instantaneous power may deviate from linear scaling due to temporal misalignment of short-term fluctuations across replicas. Because replicas operate in parallel, each model’s ITL does not scale with _wi_ . Accordingly, _pi_ , _ri_ , and _li_ denote the aggregate power, total token throughput, and average ITL of model _i_ , respectively. 

For simplicity, we assume that all GPUs assigned to the same model are assumed to share a common batch size configuration. Let **b** ≜ [ _b_ 1 _, . . . , bN_ ]<sup>_⊤_</sup> denote the batch size vector, where each _bi_ takes discrete values and is restricted to powers of two, with batch size updates applied almost immediately after the control signals are sent to GPUs. To enable continuous optimization, we introduce the relaxed decision variable **x** ≜ [ _x_ 1 _, . . . , xN_ ]<sup>_⊤_</sup> , where _xi_ approximates log2( _bi_ ). Since key GPU performance metrics scale smoothly in the log batch size domain, formulating the control problem in terms of **x** yields well-conditioned control actions. 

The optimal batch size configuration can be determined by solving the following problem over **x** at each control interval: 



$$
max x �N i=1 ri(xi) -γ ∥x -xt∥2 2 (4a) s.t. v \leq{}{}v � p(x), q � \leq{}{}v (λ, λ) (4b) li(xi) \leq{}{}Lth,i, ∀i (µi) (4c) xi \leq{}{}xi \leq{}{}xi, ∀i, (4d)
$$



4 

where _<u>x</u>_ _~~i~~_<sup>=log</sup> 2<sup><u>(</u></sup><sup>_<u>b</u>_</sup> _~~i~~_<sup>)and</sup> _<u>xi</u>_ = log2( _bi_ ) denote the lower and upper bounds on the relaxed batch size variable for model _i_ .<sup>3</sup> The optimization model’s objective in (4a) aims to maximize the aggregate token throughput across all LLM models, which aligns with a fundamental operational goal of modern data centers. The regularization term _γ ∥_ **x** _−_ **x** _t∥_ 2<sup>2,with</sup><sup>_γ>_0,</sup> penalizes large deviations of the current decision variable **x** from the previous control action **x** _t_ . This term discourages abrupt changes in batch size decisions across successive control intervals, thereby promoting smoother and more stable GPU operation and corresponding power trajectories. 

Moreover, the optimization model explicitly captures the coupling among three stakeholders: the power grid, the data center operator, and LLM service users. From the grid perspective, the voltage constraints in (4b) enforce three-phase voltage limits at all buses in the distribution system and are associated with dual variables **_<u>λ</u>_** _,_ **_λ_** _∈_ R<sup>3</sup> +<sup>_M_.Fromtheusers’perspective,</sup> the latency constraints in (4c) impose per-model quality-ofservice requirements, with dual variables _µi ≥_ 0. The mean inter-token latency threshold _L_ th _,i_ may vary across models to reflect heterogeneity in LLM architectures and servicelevel objectives. From the data center operator’s perspective, these constraints are jointly balanced against the throughputmaximization objective, enabling batch size decisions that simultaneously respect grid reliability and user experience. 

## _B. Batch Size Control via Online Feedback Optimization_ 

Since the batch size optimization in (4) is formulated as a continuous relaxation of an inherently integer-valued decision problem, discrepancies inevitably arise between the expected output and the realized behavior of the coupled user–GPU–grid system. Moreover, additional mismatches may be introduced by actuation delays, workload stochasticity, and unmodeled system dynamics. Online feedback optimization (OFO) inherently mitigates these issues by updating batch size decisions directly from real-time system measurements, rather than relying on exact model fidelity. This feedbackdriven structure renders OFO robust to modeling inaccuracies and implementation imperfections. 

We follow the standard OFO implementation in [25] to solve (4) and add a step for discrete actuation after that. At each control interval _t_ = 0 _,_ 1 _,_ 2 _, . . ._ , the OFO controller executes the following steps. 

_Step 1: Measurement:_ The controller measures the threephase voltage magnitudes at all buses, denoted by **v** ˆ _t_ , and the mean ITL of each LLM model, denoted by<sup>ˆ</sup> _li,t_ . 

_Step 2: Dual Variable Updates:_ The dual variables associated with the voltage and latency constraints are updated via projected gradient ascent: 



$$
λt+1 = � λt + ρv � v -ˆvt �� +, (5) λt+1 = � λt + ρv �ˆvt -v �� +, (6) µi,t+1 = � µi,t + ρl �ˆli,t -Lth,i �� +, ∀i, (7)
$$

> 3We use _<u>b</u>_ _~~i~~_<sup>=8,asgoinglowerhurtsthroughputsignificantlywithout</sup> lowering ITL. _bi_ is set as the largest batch size that fits in GPU memory. 

where _ρv >_ 0 and _ρl >_ 0 are dual step sizes, and [ _·_ ]+ denotes element-wise projection onto the nonnegative orthant. 

_Step 3: Primal Update in Log_ 2 _Batch Size Space:_ The relaxed primal decision variable **x** is updated via projected gradient descent: 



$$
xt+1 = Π[x,x] � xt -ρx\nabla{}{}xL � xt, λt+1, λt+1, µt+1 �� , (8)
$$

where _ρx >_ 0 is the primal step size, _L_ is the Lagrangian function associated with (4), and Π[ **<u>x</u>** _<u>,</u>_ **<u>x</u>** ]<sup>(</sup><sup>_·_)denoteselement-</sup> wise projection onto the box constraints _<u>x</u>_ _~~i~~_<sup>_≤xi≤_</sup> _<u>xi</u>_ . The derivation of _∇_ **x** _L_ is provided in Appendix B. This gradient captures the trade-offs among throughput maximization, latency constraints, voltage regulation, and penalties on large batch-size adjustments. 

_Step 4: Discrete Actuation (Mapping_ **x** _t_ +1 _to_ **b** _t_ +1 _):_ The OFO update produces a continuous decision **x** _t_ +1 _∈_ R<sup>_N_</sup> , whereas the GPU runtime requires discrete batch size settings. We therefore map each component to the nearest integer in log2 scale and convert back to batch size: 



$$
˜xi,t+1 = round(xi,t+1), bi,t+1 = 2˜xi,t+1, ∀i. (9)
$$

The resulting batch size vector **b** _t_ +1 = [ _b_ 1 _,t_ +1 _, . . . , bN,t_ +1]<sup>_⊤_</sup> is then applied to the GPU servers. 

In summary, OFO enables the data center operator to iteratively adjust GPU batch sizes using real-time voltage and latency feedback, without requiring an exact or static system model. This makes OFO particularly well suited for real-time G2G coordination under practical implementation constraints. 



IV. NUMERICAL EXPERIMENTS

## _A. Data Center Power Profile Generation_ 

A key challenge in numerical studies of data centers is the lack of publicly available, high-resolution power measurements that capture responses to inference-level control knobs such as batch size. Existing datasets (e.g., the MIT Supercloud Dataset [26]) characterize aggregate behavior but do not resolve control-induced power dynamics. To address this gap, we develop a cluster simulator based on real measurement data from [12], [15] to emulate realistic GPU responses (including power, ITL, and throughput) to batch-size control. 

Synthetic load generation is designed to capture both realism and diversity. To improve realism, we superimpose multiple replica-level GPU power traces with randomly shifted start times, rather than directly scaling a single trace. This represents asynchronous workload arrivals and avoids unrealistically amplified transients. We also model time-varying intertoken latency (ITL): because historical ITL measurements exhibit heavy-tailed behavior, we fit a weighted mixture of two lognormal distributions for each batch size, as shown in Fig. 4. At each control interval, replica-level ITLs are sampled and averaged to obtain the model-level ITL used for latency evaluation. To introduce diversity, we include both fast and slow power variations. Fast variations are produced by a temporary training workload running concurrently with inference, representing events such as training interruptions or resumptions. Slow variations are generated by gradually reducing the number of active LLM inference replicas, mimicking changes in request arrival rates over time. 



5 

![Figure](assets/figure_0004_page_0005.svg)Fig. 4. Fitted ITL distributions across batch sizes for the Llama 3.1 8B model. 

Fig. 5. Synthetic data center power and average ITL with a fixed batch size of 128 for all LLM models. 

The simulated data center has an aggregate capacity of approximately 5 MW and consists of 900 servers (8 GPUs each), evenly distributed across three phases. A constant base load of 0.5 MW per phase is included to represent ancillary infrastructure such as cooling, accounting for roughly 30% of total consumption [27]. We consider five heterogeneous LLM inference workloads (detailed in Table III in Appendix C), running together over a 60-minute horizon with 0.1 s resolution. A transient training workload is added over _t_ = 1000 to 2000 s, and inference demand is linearly reduced from _t_ = 2500 to 3000 s, producing both short-term variability and sustained power shifts that induce significant voltage dynamics in the distribution system. We use this workload pattern for subsequent evaluations. Fig. 5(a) shows the resulting power profile and average ITL for a benchmark case with fixed batch size 128. Also, as shown by Fig. 5(b), the variability of permodel average ITL increases as the number of active GPUs decreases (orange area) due to reduced statistical averaging across servers. 

## _B. GPU-to-Grid Simulation via OpenG2G_ 

We evaluate voltage impacts using the IEEE 13-bus distribution feeder [28], with the data center connected at Bus 671 and operating at a constant power factor of PF = 0 _._ 95, as shown in Fig. 6. Simulations are performed using the opensource OpenG2G library [1], [2], which in turn invokes OpenDSS [16], [29] for distribution system simulation. 

As a no-GPU-flexibility baseline, we simulate the synthetic data center load in Fig. 5(a), with voltage regulation provided only by step-voltage-regulator tap changes. Because frequent tap operations increase wear, maintenance needs, and outage risk, we impose a 30-minute minimum dwell time as a conservative limit on excessive mechanical actuation, with the 

![Figure](assets/figure_0005_page_0005.svg)Fig. 6. IEEE 13-bus distribution feeder with data center load at Bus 671. 

earliest tap operation allowed at _t_ = 25 min. Fig. 7 shows the resulting voltage trajectories on phases A–C. Although tap changes correct sustained deviations at _t_ = 25 min and _t_ = 55 min, the enforced delay causes temporary voltage violations after data center load changes, motivating GPU flexibility as a fast complementary voltage-regulation resource. 

## _C. Batch Size Optimization Results_ 

Voltage regulation with GPU flexibility is implemented using an OFO controller with primal step size _ρx_ = 0 _._ 1, dual step sizes _ρv_ = _ρl_ = 1, and objective weight _γ_ = 0 _._ 1, operating at a 1 s control interval. For all models, batch sizes are selected from the discrete set _{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _,_ 512 _}_ . The resulting voltage trajectories are shown in Fig. 8, and the corresponding GPU performance metrics are shown in Fig. 9. 

To interpret the batch size trajectories in Fig. 9, we categorize controller actions into three regimes: throughput-driven, voltage-driven, and latency-driven, reflecting how the OFO controller balances data center performance objectives against grid and users’ requirements. 

_Throughput-driven regions:_ When no constraints in (4) are active, or when constraint violations do not dominate the gradient _∇_ **x** _L_ in (8), the OFO controller maximizes aggregate token throughput across all models. As shown in Fig. 9(a), throughput-driven regions appear before and after the training window, ensuring performance maximization during non-critical intervals. In our implementation, per-replica throughput is normalized to a maximum of one to enable fair aggregation across models; in practice, operators may apply model-specific throughput weights to reflect service priorities. 

_Voltage-driven regions:_ Voltage-driven actions occur during the abrupt undervoltage event near _t ≈_ 1000 s and the gradual overvoltage event near _t ≈_ 3000 s. The corresponding stepwise changes in per-replica power, highlighted in Fig. 9(b), reflect aggressive batch size reductions and increases, respectively. Comparing Fig. 7 and Fig. 8, GPU batch size control enables faster and smoother voltage recovery than tap changers, which are constrained by slow mechanical actuation. Notably, the overvoltage case illustrates that increasing GPU 



6 

![Figure](assets/figure_0006_page_0006.svg)Fig. 7. Voltage trajectories in IEEE 13-bus system without GPU flexibility. The dashed lines indicate the voltage limits (0.95 and 1.05 pu). 

![Figure](assets/figure_0007_page_0006.svg)Fig. 8. Voltage trajectories in IEEE 13-bus system with GPU flexibility and no tap change. Voltages stay mostly within their limits. 

TABLE II 

VOLTAGE REGULATION PERFORMANCE COMPARISON 

|**C**|**Violation**|**Worst**|**Worst**|**Integral**|
|---|---|---|---|---|
|**ase**|**Time (s)**|_V_min **(pu)**|_V_max **(pu)**|**Viol. (pu·s)**|
|No control, no tap|994.3|0.9352|1.0431|30.86|
|Tap change only|1005.1|0.9352|1.0568|22.15|
|GPU control only|62.4|0.9452|1.0445|0.0848|



Note: Violation time is total voltage violation duration. Worst _V_ min/ _V_ max are extrema across all buses and phases. Integral viol. is the time integral of out-of-limit voltage deviations. 

power consumption provides valuable grid support, a result of interest to both power and computer systems communities. 

_Latency-driven regions:_ As shown in Fig. 9(c), ITL variability increases significantly after _t ≈_ 3000 s. This is because empirically, larger batch sizes are associated with broader ITL distributions, resulting in greater latency fluctuations and a higher risk of violating latency constraints. Consequently, the batch size decisions in the green shaded region in Fig. 9 are primarily driven by latency regulation. 

Finally, Table II quantitatively compares the voltage regulation performance of different cases. While tap-only control prolong voltage violations relative to the uncontrolled baseline due to actuation delays and overcorrection, GPU-based control reduces the integral voltage violation (capturing both the duration and magnitude of voltage deviations) by orders of magnitude without any tap operations during the simulation. This improvement arises from closed-loop feedback, which enables rapid correction of voltage deviations and avoids the prediction errors inherent in slow, open-loop voltage regulation devices. These results suggest that the inherent GPU flexibility may allow data centers to meet power system requirements without using additional flexible resources such as batteries. 

## V. CONCLUSION 

This paper demonstrates the potential of GPU-level control for distribution-level voltage regulation using real LLM inference data, proving the batch size is an effective control knob 

![Figure](assets/figure_0008_page_0006.svg)Fig. 9. OFO modulates batch size for each model to maximize throughput while meeting target inter-token latency constraints. 

for grid support by datacenters. Specifically, we propose an OFO framework that balances the requirements of the power grid, LLM service users, and data center operators by jointly considering voltage constraints, latency limits, and throughput objectives, while relying only on readily available grid measurements and avoiding the need for detailed grid information. A limitation of this study is that data center power, latency, and throughput dynamics are generated from pre-measured traces and fitted performance models, which may not capture all sources of variability present in real GPU operation. Future work includes extending to a hardware-in-the-loop setting, where GPU performance metrics are measured in real time and fully integrated into the control loop, enabling end-to-end validation under realistic operating conditions. 



7 

![Figure](assets/figure_0009_page_0007.svg)Fig. 10. Fitted relationships between batch size and performance metrics for two Qwen models. 

## ACKNOWLEDGMENT 

We thank the reviewers for their insightful feedback. Zhirui Liang is supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, and Jae-Won Chung is supported by the Kwanjeong Educational Foundation and the Rackham Predoctoral Fellowship. This work was supported in part by NSF grants CCF2450085 and CNS-2106184, DARPA ML2P Award HR001126-9-E190, and grants from Ford and the Laude Institute. 

The authors used AI tools to assist with narrative polishing, including spell-checking and streamlining arguments, as well as for coding and debugging. None of the narrative was directly produced by AI; it was used solely in an implementation tool and did not replace the authors. All models and ideas are the sole intellectual property of the authors, and no AI model contributed to their development. 

## APPENDIX 

## _A. Additional GPU Measurement Data Plots_ 

We present additional GPU measurement results for Qwen models to demonstrate that the key findings in the main text derived from Llama models generalize to other architectures. The fitted relationships between batch size and performance metrics for two Qwen models are shown in Fig. 10, which follow the same logistic functional form as (1)-(3). 

In addition, Fig. 11 shows the fitted ITL distributions across batch sizes for the Qwen3 235B A22B model. As in Fig. 4, each distribution is modeled as a weighted mixture of two lognormal distributions. One stall distribution represents short-duration decoding events concentrated around the mean latency, while the steady distribution captures longer-lasting components that dominate the tail of the distribution. However, for batch sizes larger than 64, the distributions exhibit greater overlap in this case, which means that the ITL is not as sensitive to batch size increase as the Llama 3.1B 8B model. 

## _B. Gradient Derivation with Respect to Batch Size_ 

The proposed formulation in (4) is fully differentiable with respect to **x** . Accordingly, the associated Lagrangian function 

![Figure](assets/figure_0010_page_0007.svg)Fig. 11. Fitted ITL distributions across batch sizes for the Qwen3 235B A22B model. 

can be written as 



$$
L(x, λ, λ, µ) = -�N i=1 ri(xi) + γ ∥x -xt∥2 2 + λ ⊤� v � p(x), q � -v � + λ⊤� v -v � p(x), q �� + �N i=1 µi � li(xi) -Lth,i � . (10)
$$

Let **_η_** ≜ **_λ_** _−_ **_<u>λ</u>_** _∈_ R<sup>3</sup><sup>_M_</sup> . The partial derivative of the Lagrangian with respect to _xi_ is 



$$
\partial{}{}L \partial{}{}xi = -dri(xi) dxi + 2γ(xi -xt,i) + µi dli(xi) dxi + η⊤\partial{}{}v \partial{}{}p \partial{}{}p(x) \partial{}{}xi . (11)
$$

Under a three-phase linearized distribution flow (LinDistFlow) approximation [30], and assuming that power injections at all non–data-center buses remain constant, the bus voltage magnitudes changes from time _t_ to _t_ + 1 can be expressed as approximately affine functions of the power consumptions at the data center bus: 



$$
vt+1 = vt -R ∆pt -X ∆qt, (12)
$$

where ∆ **p** _t_ = **p** _t_ +1 _−_ **p** _t_ and ∆ **q** _t_ = **q** _t_ +1 _−_ **q** _t_ denote the changes in active and reactive power consumptions at the data center bus. The sensitivity matrices **R** _,_ **X** _∈_ R<sup>3</sup><sup>_M×_3</sup> capture both within-phase and cross-phase voltage responses to variations in active and reactive data center load. Therefore, the voltage sensitivity with respect to active power becomes 



$$
H ≜\partial{}{}v \partial{}{}p = -R -tan(arccos(PF)) X \in{}{}R3M\times{}{}3. (13)
$$

Since model _i_ may be executed on GPUs connected to different phases _ϕ ∈ {A, B, C}_ of the power system, we introduce a phase-allocation weight vector **e** _i_ = [ _ei,A, ei,B, ei,C_ ]<sup>_⊤_</sup> _∈_ R<sup>3</sup> where _ei,ϕ_ denotes the fraction of GPUs assigned to model _i_ that are connected to phase _ϕ_ . Therefore, we have 



$$
\partial{}{}p(x) \partial{}{}xi = ei dpi(xi) dxi , (14)
$$

Given the logistic functions in (1), (2), and (3) which are the functions of power, latency, and throughput for one replica of model deployment, we obtain the gradient for the power, latency, and throughput of all replicas 



$$
dpi(xi) dxi = Pmaxkpwi exp(-kp(xi -x0,p)) (1 + exp(-kp(xi -x0,p)))2 , (15)
$$



8 

TABLE III 

LLM INFERENCE WORKLOADS AND MODEL-SPECIFIC PARAMETERS IN 

NUMERICAL EXPERIMENTS 

|**Model name**|**Replica **|**Count**|**GPUs per **|**replica**|_Lth_ **(s)**|
|---|---|---|---|---|---|
|Llama 3.1 8B||720||1|0.08|
|Llama 3.1 70B||180||4|0.10|
|Llama 3.1 405B||90||8|0.12|
|Qwen3-30B A3B||480||2|0.06|
|Qwen3 235B A22B||210||8|0.14|





$$
dli(xi) dxi = Lmaxkl exp(-kl(xi -x0,l)) (1 + exp(-kl(xi -x0,l)))2 , (16)
$$



$$
dri(xi) dxi = Rmaxkrwi exp(-kr(xi -x0,r)) (1 + exp(-kr(xi -x0,r)))2 . (17)
$$

In summary, we obtain the gradient of Lagrangian with respect to _xi_ as 



$$
\partial{}{}L \partial{}{}xi = 2γ(xi -xt,i) -Rmaxkrwi exp(-kr(xi -x0,r)) (1 + exp(-kr(xi -x0,r)))2 + η⊤Hei Pmaxkpwi exp(-kp(xi -x0,p)) (1 + exp(-kp(xi -x0,p)))2 + µiLmaxkl exp(-kl(xi -x0,l)) (1 + exp(-kl(xi -x0,l)))2 . (18)
$$

## _C. Simulation Setup_ 

The topology of the IEEE 13-bus feeder with a data center load is shown in Fig. 6. Bus 650 serves as the upstream substation and voltage reference, with its voltage regulated by the transmission system and thus weakly influenced by downstream load variations. Voltage regulation within the feeder is primarily provided by the step-voltage regulator between Bus 650 and Bus 632, whose tap operations produce discrete voltage changes at the regulator bus in response to sustained load variations. Thus, the voltage at the regulator bus reflects discrete changes corresponding to tap operations. 

In the numerical experiments, we consider the five LLM models listed in Table I. Each model is assigned an initial replica count and a latency threshold _Lth_ , with larger models serving fewer users and tolerating higher latency. The resulting configuration occupies 600 servers, providing sufficient GPU flexibility for voltage regulation. The reset 300 servers are used for training during the training window _t ∈_ [1000 _,_ 2000] s. 

## REFERENCES 

- [1] J.-W. Chung, Z. Liang, Y. Mao, J. Chen, M. Chowdhury, and V. Dvorkin, “OpenG2G: A simulation platform for AI datacenter-grid runtime coordination,” _arXiv preprint arXiv:2605.05519_ , 2026. 

- [2] “OpenG2G.” https://github.com/gpu2grid/openg2g. 

- [3] Masanet, Eric and Shehabi, Arman and Lei, Ning and Smith, Sarah and Koomey, Jonathan, “United states data center energy usage report,” Technical Report LBNL-2024-DataCenterReport, Lawrence Berkeley National Laboratory, 2024. 

   - [6] C. Guille and G. Gross, “A conceptual framework for the vehicle-to-grid (v2g) implementation,” _Energy policy_ , vol. 37, no. 11, pp. 4379–4390, 2009. 

   - [7] V. Dvorkin, “Agent coordination via contextual regression (agentconcur) for data center flexibility,” _IEEE Transactions on Power Systems_ , 2024. 

   - [8] Y. Fu, X. Han, K. Baker, and W. Zuo, “Assessments of data centers for provision of frequency regulation,” _Applied Energy_ , vol. 277, p. 115621, 2020. 

   - [9] Y. Xie, W. Cui, and A. Wierman, “Enhancing data center low-voltage ride-through,” _arXiv preprint arXiv:2510.03867_ , 2025. 

   - [10] J. You, J.-W. Chung, and M. Chowdhury, “Zeus: Understanding and optimizing gpu energy consumption of dnn training,” in _20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)_ , pp. 119–139, 2023. 

   - [11] J.-W. Chung, Y. Gu, I. Jang, L. Meng, N. Bansal, and M. Chowdhury, “Reducing energy bloat in large model training,” _Proceedings of the 30th ACM Symposium on Operating Systems Principles_ , 2024. 

   - [12] J.-W. Chung, J. J. Ma, R. Wu, J. Liu, O. J. Kweon, Y. Xia, Z. Wu, and M. Chowdhury, “The ML.ENERGY benchmark: Toward automated inference energy measurement and optimization,” in _NeurIPS Datasets and Benchmarks_ , 2025. 

   - [13] Y. Chen and B. Zhang, “Voltage regulation in distribution systems with data center loads,” _arXiv preprint arXiv:2507.06416_ , 2025. 

   - [14] P. Colangelo, A. K. Coskun, J. Megrue, C. Roberts, S. Sengupta, V. Sivaram, E. Tiao, A. Vijaykar, C. Williams, D. C. Wilson, _et al._ , “Ai data centres as grid-interactive assets,” _Nature Energy_ , pp. 1–8, 2025. 

   - [15] “The ML.ENERGY benchmark.” https://github.com/ml-energy/ benchmark. 

   - [16] R. C. Dugan and T. E. McDermott, “An open source platform for collaborating on smart grid research,” tech. rep., Electric Power Research Institute (EPRI), 2011. 

   - [17] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with PagedAttention,” in _SOSP_ , 2023. 

   - [18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _NeurIPS_ , 2017. 

   - [19] A. . M. Llama Team, “The llama 3 herd of models,” _arXiv preprint arXiv:2407.21783_ , 2024. 

   - [20] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” _arXiv preprint arXiv:1701.06538_ , 2017. 

   - [21] Q. Team, “Qwen3 technical report,” _arXiv preprint arXiv:2505.09388_ , 2025. 

   - [22] J. Liu, J.-W. Chung, Z. Wu, F. Lai, M. Lee, and M. Chowdhury, “Andes: Defining and enhancing quality-of-experience in llm-based text streaming services,” _arXiv preprint arXiv:2404.16283_ , 2024. 

   - [23] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, “Orca: A distributed serving system for Transformer-Based generative models,” in _OSDI_ , 2022. 

   - [24] J. J. Grainger and W. D. Stevenson, _Power System Analysis_ . New York, NY, USA: McGraw–Hill, 1994. 

   - [25] L. Ortmann, A. Hauswirth, I. Caduff, F. D¨orfler, and S. Bolognani, “Experimental validation of feedback optimization in power distribution grids,” _Electric Power Systems Research_ , vol. 189, p. 106782, 2020. 

   - [26] S. Samsi, M. L. Weiss, D. Bestor, B. Li, M. Jones, A. Reuther, D. Edelman, W. Arcand, C. Byun, J. Holodnack, _et al._ , “The mit supercloud dataset,” in _2021 IEEE High Performance Extreme Computing Conference (HPEC)_ , pp. 1–8, IEEE, 2021. 

   - [27] International Energy Agency, “Energy demand from ai and data centers,” _IEA Report_ , 2025. 

   - [28] IEEE Distribution System Analysis Subcommittee, “Ieee 13 node test feeder,” tech. rep., IEEE Power & Energy Society, 2014. 

   - [29] M. J. O’Connell and contributors, “OpenDSSDirect.py: Direct python interface to opendss.” https://github.com/dss-extensions/OpenDSSDirect. py, 2020. 

   - [30] L. Gan and S. H. Low, “Convex relaxations and linear approximation for optimal power flow in multiphase radial networks,” in _2014 power systems computation conference_ , pp. 1–9, IEEE, 2014. 

- [4] NVIDIA Corporation, “Nvidia h100 tensor core gpu.” https://www. nvidia.com/en-us/data-center/h100/, 2024. 

- [5] X. Chen, X. Wang, A. Colacelli, M. Lee, and L. Xie, “Electricity demand and grid impacts of ai data centers: Challenges and prospects,” _arXiv preprint arXiv:2509.07218_ , 2025. 

