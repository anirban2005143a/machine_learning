Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures<sup>_⋆_</sup> 

Mauricio Fadel Argerich<sup>_a_,∗</sup> , Jonathan Fürst<sup>_b_</sup> and Marta Patiño-Martínez<sup>_a_</sup> 

_aUniversidad Politécnica de Madrid, C. de los Ciruelos, Boadilla del Monte, 28660, Madrid, Spain_ 

_bZurich University of Applied Sciences, Switzerland, Gertrudstrasse 15, Winterthur, 8400, Zurich, Switzerland_ 

|A R T I C L E I N F O|A B S T R A C T|
|---|---|
|_Keywords_:|While the large energy consumption of Large Language Models (LLMs) is recognized by the<br>fi|
|AI sustainability<br>Large Language Models<br>fi|community, system operators lack guidance for energy-efficient LLM inference deployments that<br>leverage energy trade-offs of heterogeneous hardware due to a lack of energy-aware benchmarks|
|AI efficiency|and data. In this work we address this gap with Watt Counts: the largest open-access dataset of|
|LLM inference|energy consumption of LLMs, with over 5,000 experiments for 50 LLMs across 10 NVIDIA Graphics|
|Energy-aware dataset|Processing Units (GPUs) in batch and server scenarios along with a reproducible, open-source|
|LLM deployment benchmark<br>GPU energy efficiency<br>Inference benchmark|benchmark that enables community submissions to expand this dataset. Leveraging this dataset, we<br>conduct a system-level study of LLM inference across heterogeneous GPU architectures and show<br>that GPU selection is crucial for energy efficiency outcomes and that optimal hardware choices<br>vary significantly across models and deployment scenarios, demonstrating the critical importance<br>of hardware-aware deployment in heterogeneous LLM systems. Guided by our data and insights,<br>we show that practitioners can reduce energy consumption by up to 70% in server scenarios with<br>negligible impact on user experience, and by up to 20% in batch scenarios.|



# **1. Introduction** 

While the high-energy demands of Large Language Models (LLMs) are well recognized, the energy footprint of LLM inference deployments is often overlooked and poorly understood [1]. Although LLM inference is generally less energy-intensive than training, its total energy consumption is often greater because it requires server deployments with continuous availability to respond to user queries [2]. LLM inference deployments span across a variety of environments including edge servers, on-premise infrastructure, and cloud deployments with heterogeneous hardware, making energyaware deployment both critical and complex [3]. High energy consumption of these deployments not only generates carbon emissions and increases electricity costs for domestic users and industry [4], but also slows progress towards the Paris Agreement and the UN Agenda 2030 targets on clean energy and climate action [5]. 

In particular, recent studies show that _Graphics Processing Units (GPUs) are the primary energy consumers in AI inference, accounting for up to 90% of total system power draw_ [6], [7], underscoring the need for their careful evaluation. This challenge is even greater in modern heterogeneous LLM deployments, where inference workloads are executed on diverse GPU architectures with varying memory capacities, compute characteristics, and power profiles. However, a lack of system-level evidence and tools to benchmark the energy consumption of LLM-GPU configurations means system operators must deal on their own with complex 

_⋆_ 

mauricio.fadel@alumnos.upm.es (M. Fadel Argerich); fues@zhaw.ch (J. Fürst); mpatino@fi.upm.es (M. Patiño-Martínez) 

(M. Fadel Argerich); (J. Fürst); (M. Patiño-Martínez) ORCID(s): 1 

energy-performance trade-offs arising from the added effects of LLM characteristics, GPU architecture, and deployment scenarios. 

To address these challenges, we introduce _Watt Counts_ , a large-scale, energy-aware benchmark and dataset designed to support deployment decisions for sustainable LLM inference systems. Our contributions are as follows: 

1. We publish the _largest energy-aware, open dataset of LLMs inference_ for system-level analysis of LLM inference across heterogeneous GPU architectures. Our dataset contains power draw, energy per token, throughput, and several additional performance metrics for 50 LLMs running on ten GPUs with five different NVIDIA architectures in batch and server scenarios. _This results in 5K+ single experiments on 370 different LLM-GPU pairs, and 14M+ power draw samples._ 

2. A _customizable, open-source, and easy-to-run benchmark for measuring the energy consumption of LLM inference deployments_ , enabling reproducible energy and performance measurement in bare metal and cloud instances, which does not require any external power meters. This benchmark allows users to easily contribute to our live dataset. Both the dataset and the source code for the benchmark will be released on Github under an MIT license upon acceptance and are provided in the supplementary material. 

3. A _comprehensive system-level study of LLM inference energy consumption_ , revealing how GPU architectural features (e.g., memory bandwidth, cache hierarchy, TDP), model architectural properties, and deployment scenario jointly determine energy efficiency and performance trade-offs. Guided by our data and insights, 

Page 1 of 13 

: _Preprint submitted to Elsevier_ 



we show that energy consumption can be reduced by up to 70% in server scenarios with negligible impact on user experience, and by up to 20% in batch scenarios. 

Together, these resources provide the foundation for data-driven decisions in the deployment of sustainable LLM systems that leverage heterogeneous GPU architectures to improve their scalability and, in particular, their energy efficiency. 

The rest of this paper is organized as follows. Section 2 surveys related work across related topical areas. Section 3 describes the benchmark methodology and design, including the energy and performance metrics to be analyzed, and Section 4 describes how the benchmark was used to create the Watt Counts dataset. Section 5 uses the dataset to develop a system-level study of LLM inference energy consumption. Section 6 addresses limitations and broader considerations of this work as well as future work opportunities, and Section 7 synthesizes the key findings. 

# **2. Related Work** 

The recent widespread use of LLMs has made understanding and optimizing their energy consumption a critical research topic. We review previous work in three key areas: (1) energy and carbon measurement studies during training and inference, (2) tools and benchmarks for energy estimation, and (3) energy-aware datasets for LLM inference. 

**Energy and carbon measurement studies during training and inference.** Several works have focused on measuring the energy consumption and carbon emissions of LLMs during training [8, 9, 10], as this phase is typically the most energy-intensive. However, the inference phase often has a greater impact than training, as LLMs spend most of their operational lifetime serving inference requests [11, 12, 13, 6]. In this line, Wu et al. analyze the environmental impact of experimentation, training, and inference of LLMs and recommendation models at Meta [14]. Jegham et al. estimate the energy consumption of several proprietary LLMs based on company-disclosed information since no energyaware data is available for these models [15]. Similarly, Kim et al. survey numerous optimization techniques for transformers, evaluating their impact on inference time and energy, and applying them to improve energy efficiency [16]. MLPerf Power [17] presents insights from over 1800 energy consumption measurements of various ML workloads, including LLMs, across different systems. These were obtained by running the MLPerf Inference benchmark [18] and recording system-level energy consumption with physical power meters. TokenPowerBench [19], a benchmark for measuring phase-level power consumption of LLM inference, focuses on model and inference engine comparison, evaluating 15+ open-source models only on H100 clusters for batch scenarios. At the time of writing, the dataset and benchmark implementation of TokenPowerBench are not public. 

**Tools and benchmarks for energy estimation.** In many settings, such as cloud computing, inference runs on systems that cannot be instrumented with physical power meters. To address this, several software tools have been developed, including Carbontracker [20] and CodeCarbon [21], which estimate carbon emissions, and EnergyMeter [6], which estimates energy consumption of virtualized environments or bare metal instances. Zeus [22] offers a library for measuring and optimizing energy consumption of ML training workloads by adjusting parameters such as power limits and batch size. Recently, ML.ENERGY [23] presents an LLM energy efficiency leaderboard to collect energy per response for different LLMs and tasks, based on measurements collected with Zeus. In addition, recent works have explored the performance and energy characteristics of LLM workloads. Frameworks such as DynamoLLM [24], LLMCO2 [25], throttll’em [26], and Patel et al. [27] analyze and aim to optimize energy or carbon footprint for specific GPUs (A100, H100) and models. Fernandez et al. introduce a modeling approach to estimate the energy efficiency of LLM inference, testing it on three GPUs and seven LLMs using PyTorch and vLLM [28]. Similarly, workload-based analyses on multiGPU systems [29, 26, 30] focus on optimization for specific hardware. Our dataset provides empirical data across a much wider hardware space to complement and enable the evaluation of these modeling approaches. 

**Energy-aware datasets for LLM inference.** Table 1 highlights a key gap in the literature: there is currently no open dataset or benchmark that enables systematic characterization of LLM inference energy across a diverse set of models, heterogeneous GPUs, and batch and server scenarios. Most existing studies are limited in scope, covering only a unique or narrow range of GPUs [31, 32, 23, 33], are restricted to batch scenarios, or in the case of MLPerf Power [17], they do not report GPU-level energy measurements, instead providing only aggregate system-level metrics. 

No publicly available dataset spans a diverse set of both models and heterogeneous GPU architectures under both batch and server scenarios. This gap is consequential for the deployment of heterogeneous system for LLM inference. Without cross-LLM-GPU empirical data, system operators cannot make informed decisions about GPU selection, workload placement, or energy-performance trade-offs in heterogeneous LLM deployments. _Watt Counts_ addresses this directly: it is the first large-scale, open dataset and benchmark covering 50 LLMs across 10 GPUs spanning five NVIDIA architectures, in both batch and server scenarios, fully reproducible without physical power meters. It provides the empirical foundation that the field lacks for datadriven, hardware-aware deployment of LLMs at scale. 

# **3. Benchmarking Methodology** 

This section describes the design of our benchmark and its use for the construction of the Watt Counts dataset. We first present the research questions that motivate the 

Page 2 of 13 

: _Preprint submitted to Elsevier_ 



**Table 1** 

Comparison of energy-aware LLM inference datasets 

|**Dataset**|**# GPUs**|**# LLMs**|**LLM-GPU comb.**|**Energy Metrics**|**Workload**<br>**Type/Scenario**|**Open Data**|**Engine**|
|---|---|---|---|---|---|---|---|
|LLM Energy Consumption Dataset [31]|3|6|18|✓|Batch|✓|ollama|
|Azure LLM Inf. Traces [24]|1|6|6|✗<sup>1</sup>|Server|✓|vLLM|
|AI Energy Score [33]|1|121|121|✓|Batch|✓|PyTorch,<br>vLLM,<br>Optimum<sup>2</sup>|
|ML.ENERGY [23]|2|27|54|✓|Batch|✓|vLLM|
|MLPerf Power [17]|11<sup>3</sup>|3|33|✓<sup>4</sup>|Server,<br>Batch,<br>Single/Multiple<br>Stream|✓|PyTorch, ONNX|
|J. Fernandez et al. [28]|3|7|21|✓|Batch|✗|PyTorch, vLLM|
|**Watt Counts (ours)**|**10**|**50**|**372**|✓|Server, Batch|✓|vLLM|



> 1Although the dataset does not publish per-run energy measurements, the paper reports aggregated energy statistics. 

> 2Although these inference engines are listed as supported, the published data does not specify which inference engine was used; it is therefore assumed that the default PyTorch engine was employed. 

> 3We consider only GPUs used for LLM workloads; devices such as the Cloud AI 100 Pro and Jetson AGX Orin are excluded. Including these devices would increase the total number of evaluated devices by MLPerf to 24. 

> 4MLPerf Power reports energy consumption at the system level only; GPU-level energy measurements are not provided. 

benchmark design, dataset construction and analysis (Section 3.1). Next, we introduce the design of the benchmark, including the defined inference scenarios, the energy and performance metrics collected, and its implementation to ensure the experiments are executed in a reproducible manner (Section 3.2). 

## **3.1. Research Questions** 

To systematically examine how energy efficiency varies with respect to GPU characteristics, LLM properties, and workload scenarios, we define the following research questions: 

- **RQ1: GPU Architecture Efficiency Across LLMs** . Is a single GPU consistently the most energy-efficient across all LLMs, or do efficiency rankings vary? Which GPU architectural characteristics (e.g., memory capacity, memory bandwidth, FLOPS, TDP) most strongly correlate with energy efficiency for LLM inference? 

- **RQ2: LLM Efficiency Across GPUs** . What model properties (such as size, model family, etc.) impact energy efficiency significantly during inference across GPUs? 

- **RQ3: Deployment Scenario Sensitivity** . How does the deployment scenario (batch vs. server load) influence the energy efficiency of LLMs on a given GPU? Are the same LLM–GPU pairs optimal across server and batch? 

These questions guide the benchmark design, its use to construct the Watt Counts dataset, and the subsequent analysis of LLM energy efficiency. 

## **3.2. Benchmark Design** 

The benchmark defines LLM inference for QuestionAnswering (QA) as the application under test, a common use case for LLMs. This application is used in two predefined 

scenarios (batch and server), and during execution, a metrics collector captures energy and performance measurements. The benchmark design is guided by three principles: reproducibility, modularity, and practical relevance. To support community adoption and extension of our benchmark and dataset, both are open-source and publicly available. 

The benchmark is implemented in Python and structured in four main components as shown on Figure 1. The _orchestrator_ reads a user-provided config.yaml specifying the models (via their Hugging Face hub handle), deployment scenarios, number of iterations, and request arrival rates, and launches the System Under Test (SUT.) The SUT is a vLLM inference engine [34] operating in one of two modes: _batch_ , which uses the synchronous engine to process all prompts in a single submission, and _server_ , which uses the asynchronous engine paired with a _server load generator_ that dispatches prompts following a Poisson arrival process. Throughout execution, the _metrics collector_ records GPU metrics using EnergyMeter, CPU and memory metrics with PSUtil, and collects LLM-level performance indicators including TTFT and end-to-end latency from the vLLM metrics. Additionally, before each run, detailed information about the underlying hardware, including CPU, GPU, memory, and disk specifications are collected. This metadata ensures reproducibility and allows the analysis of how hardware characteristics influence energy and performance metrics. 

vLLM was selected as it is often at the top of latency, throughput, and energy performance studies when compared to other inference engines such as SGLang or TensorRTLLM [28, 35] and is considered one of the most complete and widely used general purpose inference engines [36]. Its main optimization is PagedAttention, which enables a fine-grained GPU memory allocation and reuse to maximize throughput and minimize memory usage. To support setups from single-node deployments to large-scale evaluations on HPC clusters, we provide bash scripts for orchestrating 

Page 3 of 13 

: _Preprint submitted to Elsevier_ 



benchmark runs on bare-metal and cloud instances, as well as SLURM-managed clusters. 

To improve measurement accuracy and reproducibility, we control sources of non-determinism by fixing relevant parameters (i.e., temperature= 0, top_k= 0, top_p= 1, maximum generation length= 256, repetition_penalty= 1) and random seeds (SamplingParams, Python’s random, numpy’s random, all Pytorch’s seeds), acknowledging that full determinism is not achievable in practice unless using specialized libraries, due to GPU floating-point non-determinism and dynamic kernel scheduling [37]. We utilize default settings for vLLM including its GPU memory utilization of 90%, except for GPUs with less than 20GB where we set it to 80% to avoid out of memory errors that occur otherwise. The maximum context length is set to 1024 and the maximum generation length to 256 tokens to reflect typical QA interaction lengths while keeping total benchmarking time tractable across 370+ LLM-GPU configurations. 

### **_3.2.1. Deployment Scenarios_** 

The benchmark evaluates LLM inference for two representative deployment scenarios, based on MLPerf Inference [38]: 

1. _Batch Processing (Offline Processing)._ This scenario represents the use of LLMs in high-throughput scientific and data analysis workflows, where a large number of queries are processed in bulk. For example, LLMs are extensively applied to Information Extraction (IE), where documents are processed in a batch to extract relevant structured information [39]. In this scenario, 1,000 queries are submitted simultaneously to the vLLM engine in synchronous mode. Prompts are drawn from the SQuAD v1.1 dataset [40], which contains crowd-sourced questions based on Wikipedia articles and is commonly used to evaluate QA systems. Energy is measured and attributed only to the execution of the finite workload, i.e., idle periods before or after batch execution are outside the scope of this scenario, as it models a bounded computational task rather than a continuously running service. 

2. _Server Mode (Online Processing)._ This scenario simulates real-world LLM applications, where the model is deployed as a service and must handle several requests concurrently. The load in the server is variable as it depends on the number of concurrent users and queries the application receives. In this scenario, requests using prompts from the SQuAD v1.1 dataset are sent to the asynchronous vLLM engine, which handles concurrent requests without blocking, processing requests as they arrive following a Poisson arrival process (as in MLPerf [18]), with rates _𝜆_ = {0 _._ 017 _,_ 0 _._ 3} corresponding to low-load and high-load conditions, respectively. 

Each scenario includes an initial warmup phase: for the batch scenario, this is equal to running inference for one 

batch, for server, the warmup runs the scenario for one minute. This warmup is followed by five independent iterations per configuration (batch, low-load server, and highload server). After the warmup and between iterations, we introduce a cold-down period in which the system remains idle to stabilize hardware conditions: experiments start once the GPU power draw remains within a 3 W range for at least 30 seconds – empirically determined to reliably indicate that residual heat from the previous iteration no longer influences power measurements – and the GPU temperature falls below 65 °C, ensuring that thermal throttling, which normally occurs from 75 °C, does not influence the results. In addition, we enforce a 5 minute timeout per cold-down period to avoid blocking the benchmark in cases in which the temperature does not drop from its defined threshold. In the benchmark, each server iteration runs for five minutes, a duration empirically shown to yield energy measurements within <2% deviation (95% confidence interval) when compared to onehour runs, while substantially reducing total benchmarking time. 

### **_3.2.2. Energy Metering_** 

We leverage EnergyMeter [6] to measure the energy consumption of the GPU through pyNVML, which obtains the GPU power draw using the NVIDIA Management Library (NVML) and integrates it over execution time to obtain energy consumption in Joules. NVML was chosen because it is natively supported by NVIDIA GPUs in both bare metal and virtualized environments, enabling consistent and reproducible measurements across heterogeneous NVIDIA GPUs and deployment settings without external instrumentation. This significantly increases the portability of the benchmark, enabling the addition of more data to our dataset in the future. While NVML does not provide circuit-level power measurements, it allows to separate the GPU energy from other components. Moreover, prior work has shown its measurements deviations from external power meters are below 5% [41], which is sufficient for comparing performance across models and GPUs, the main goal of our benchmark. 

GPU power draw is sampled with NVML at board level at 10 Hz during each experiment, reflecting the complete device under exclusive use; each measured device corresponds to a single physical GPU with its own framebuffer memory (in the case of NVL devices, 94 GB for the H100 NVL and 141 GB for the H200 NVL, MultiGPU Board: No in both cases). Sampling frequencies ranging from 1 Hz to 200 Hz were evaluated empirically and 10 Hz was selected as it was observed to provide the best trade-off between measurement overhead and accuracy, with an integrated energy error of <1% when compared to 200 Hz sampling and an execution time overhead of <0.5%. 

We focus on GPU energy consumption, as it has been shown to be the dominant energy consumer in LLM inference. To validate this assumption, we ran a subset of our experiments on a representative bare metal instance (Intel(R) Core(TM) i9-9900K @ 3.60GHz, 32 GB RAM, 

Page 4 of 13 

: _Preprint submitted to Elsevier_ 



a single NVIDIA GeForce RTX 4090) using EnergyMeter to collect energy metrics for the GPU via NVML, and the CPU and DRAM via RAPL. In batch and high load server scenarios, the GPU consumed 91.2-92.5% of the total energy consumed by the three components, while in the low load server scenario it represented 78.7% and the CPU 16.6%, due to the GPU being idle for large periods of time. In all scenarios, the GPU remained the main energy consumer, exhibiting a 0.998 correlation with the total measured energy consumption. 

### **_3.2.3. Energy Metrics_** 

The total energy consumed by the GPU during LLM inference is given by: _𝐸_ = _𝑃_ ⋅ _𝑡_ where _𝑃_ denotes the average power draw during inference, and _𝑡_ is the measurement period. However, total energy consumption is often an impractical measurement for comparisons across LLM deployments, as it depends on specific workload factors including input size and execution duration, which motivates the use of normalized energy efficiency metrics. 

In **Batch scenarios** , the total energy consumption is determined by the interplay of power draw and execution time. While higher throughput, i.e., the number of tokens generated per unit of time, reduces execution time, lower power draw can also reduce total energy even at reduced throughput [42], making _energy per token_ the appropriate metric as it jointly captures both dimensions. 

In **Server scenarios** , the system must remain available to handle incoming requests, so total execution time is largely decoupled from the system’s processing speed. Moreover, the system might spend long periods idling, waiting for incoming requests. As a result, attributing total energy consumption to generated tokens or individual requests requires assumptions about request arrival rates and idle power consumption, making energy per token highly deploymentdependent in this context. We therefore utilize _mean power draw_ , as it captures idle and active periods and enables consistent comparisons of LLM-GPU deployments in server scenarios. 

### **_3.2.4. Additional Performance Metrics_** 

In addition to energy metrics, we collect a set of systemlevel and LLM-level performance metrics at a 10 Hz frequency. System metrics include GPU utilization, GPU memory usage, GPU temperature, GPU clock frequency, CPU utilization, and main memory usage. LLM-level metrics include number of input tokens (per request and in total), number of tokens generated (per request and in total), the total duration of the experiment, Time-To-First-Token (TTFT), queuing time and end-to-end response time. These measurements enable the computation of additional metrics and statistics, including throughput and aggregated performance indicators such as response-time percentiles, useful for balancing energy consumption with service-level constraints, such as response time or throughput requirements. 

Since the benchmark is designed to characterize energy and performance independently of task-specific outputs, accuracy evaluation is intentionally excluded from scope; the systems output correctness does not affect power draw, energy consumption, latency nor throughput measurements. Practitioners are referred to the respective model documentation for task-specific accuracy benchmarks. 

# **4. Dataset Construction** 

We leverage our benchmark to create a comprehensive dataset that captures energy consumption patterns for a wide range of LLM-GPU pairings. 

We select 50 open-source, popular models from the Hugging Face Hub, covering a wide range of architectures, sizes, and vendors. Because our objective is to characterize LLM energy and performance across GPUs, we limit model size to 30B parameters (requiring approximately 60GB in 16-bit precision) to ensure that each model fits on at least two of the evaluated GPUs. The current version of the dataset focuses on single-GPU setups to thoroughly evaluate model power behavior across diverse GPU architectures, avoiding additional noise and variability introduced by factors such as communication overhead, tensor-parallel and pipelineparallel synchronization costs, memory sharding strategies, and interconnect bandwidth limitations. 

For hardware, we select a wide range of NVIDIA GPUs, spanning five different architectures and including serverclass and consumer-grade devices (Table 2), enabling the analysis of architectural design trade-offs that affect energy efficiency. The selected GPUs vary significantly in memory capacity (16GB–141GB) and thermal design power (TDP) (70W–600W), reflecting the heterogeneity of realworld LLM deployment environments. 

# **5. Dataset Analysis and Insights** 

This section presents a systematic analysis based on the Watt Counts dataset described in Section 4, comprising 5,000+ experiments across 50 LLMs and 10 GPUs in batch, low-load server, and high-load server scenarios. Each configuration was repeated five independent times following the benchmark methodology described in Section 3. 

## **5.1. RQ1: GPU Architecture Efficiency Across LLMs** 

_Is a single GPU consistently the most energy-efficient across all LLMs, or do efficiency rankings vary? Which GPU architectural characteristics (e.g., memory capacity, memory bandwidth, FLOPS, TDP) most strongly correlate with energy efficiency for LLM inference?_ 

We measure the energy per token each model achieves across GPUs in the batch scenario, ranking the GPUs from the most efficient, i.e., the one that achieves the lowest energy per token for the model, to the least efficient. In particular, we analyze batch inference data because this scenario minimizes idle periods (idle periods average 2% in the dataset), providing a cleaner signal for correlating 

Page 5 of 13 

: _Preprint submitted to Elsevier_ 



**Table 2** 

NVIDIA GPUs used in the dataset and their main technical characteristics: architecture, profile (consumer or enterprise), memory, TDPs, tensor FP16 TFLOPS without sparsity, memory bandwidth, L2 cache size, TFLOPS (without sparsity) per Watt, and release year. 

|GPU|Architecture|Profile|Memory<br>(GB)|TDP (W)|TFLOPS<br>(FP16)|Mem BW<br>(GB/s)|L2 Cache<br>(MB)|TFLOPS<br>per Watt|Year|
|---|---|---|---|---|---|---|---|---|---|
|Tesla V100 SXM2|Volta|Enterprise|32|250|125|898|6|0.5|2018|
|Tesla T4|Turing|Enterprise|16|70|65|320|4|0.93|2018|
|A100 SXM4|Ampere|Enterprise|40|400|312|1555|40|0.78|2020|
|GeForce RTX 3090|Ampere|Consumer|24|350|71|936|6|0.20|2020|
|A30 PCIe|Ampere|Enterprise|24|165|165|933|24|1.00|2021|
|GeForce RTX 4090|Ada Lovelace|Consumer|24|450|165|1010|72|0.37|2022|
|L40S|Ada Lovelace|Enterprise|48|300|362|864|48|1.21|2022|
|L4|Ada Lovelace|Enterprise|24|72|121|300|48|1.68|2023|
|H100 NVL|Hopper|Enterprise|94|400|835|3940|50|2.39|2023|
|H200 NVL|Hopper|Enterprise|141|700|835|4800|50|1.19|2024|



**Table 3** 

GPU ranking positions of energy-per-token achieved in the batch scenario. Only GPUs ranked 3rd or above for at least one model are shown. 

|GPU|1st|2nd|3rd|Total models|
|---|---|---|---|---|
|A100 SXM4 40 GB|0|0|13|42|
|A30 PCIe|0|1|13|34|
|GeForce RTX 4090|0|1|1|34|
|H100 NVL 94 GB|45|4|1|50|
|H200 NVL|2|43|2|50|
|L4|3|1|11|33|
|L40S|0|0|2|43|



GPU architectural characteristics with energy efficiency. The impact of workload scenario (batch vs. server) is examined separately in RQ3. We summarize the rankings for all models in Table 3, together with the total number of models executed on each GPU – determined by memory capacity constraints and compatibility limitation on older devices (Tesla V100, T4.) 

Across all models, the H100 achieves the lowest energy per token in 90% of cases. The H200 ranks second for most models and only achieves first place for two of the largest models in our set (‘google/gemma-3-27bit‘, ‘nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16‘.) In contrast, the L4 achieves the first place for three models (‘Qwen/Qwen2.5-0.5B-Instruct‘, ‘openai-community/gpt2‘, ‘tiiuae/Falcon3-Mamba-7B-Instruct‘,) two of which are among the smallest. This suggests that energy efficiency depends on interactions between GPU characteristics and model size. 

To analyze this interaction, we group the models in the dataset into four categories based on their hardware compatibility observed during benchmarking (FP16 inference, fixed context length). Specifically, we define the following size categories: 

- Small: models that fit on all evaluated GPUs, comprising models between 0.124B and 6B parameters. 

- Medium: models that fit on all GPUs with more than 16GB of memory (i.e., excluding the T4 in our setup). This includes models with 6B-10.7B parameters. 

- Large: models that fit on GPUs with more than 32GB (i.e., excluding the T4, A30, RTX 3090, RTX 4090, and L4), including models with 12B-20B parameters. 

- Xlarge: models that fit on GPUs with more than 48GB of memory (i.e., H100 and H200), in our setup this includes models with 20B-27B parameters. 

Models are assigned to the smallest compatible category based on whether they successfully loaded without out-ofmemory errors. We compute the mean energy per generated token for each GPU and category as shown in Figure 2. To ensure fair comparisons, each category includes only dense models that ran on all GPUs in that category. 

The H100 achieves the lowest mean energy per token across all categories, followed by the H200. Interestingly, for small and medium models the A30 achieves third place, tied with the L4 for small models, both surpassing more powerful GPUs in terms of TFLOPS and memory bandwidth. Their advantage is explained by their low TDPs, which compensate for their low throughput. 

For throughput, the H200 achieves the highest mean in all categories (Figure 3), with the H100 achieving a significantly lower throughput (38%-58% lower than the H200’s.) This advantage likely stems from the H200’s higher memory bandwidth, a 25% higher than the one of H100. A similar behavior is observed in the L40S and the A100, while the L40S has a 16% higher TFLOPS performance, the A100 achieves a higher throughput in all categories thanks to its 80% higher memory bandwidth. This is consistent with previous work that has identified memory bandwidth as a primary bottleneck in LLM inference [43]. 

These results show that TFLOPS performance does not imply higher throughput, and a higher throughput does not necessarily translate to lower energy per token. For instance, deploying an Xlarge model in an H100 can save 20% of energy compared to the H200, which would be selected if 

Page 6 of 13 

: _Preprint submitted to Elsevier_ 



we were to optimize the throughput. Moreover, a commonly used energy efficiency proxy, TFLOPS per Watt, is a poor predictor at the system level across full inference workloads: as shown on Table 2 and Figure 2, GPUs with higher TFLOPS/W do not consistently achieve lower energy per token. 

**Answering RQ1:** _While no single GPU is universally the most energy-efficient across all LLMs, the H100 dominates in 90% of models in the batch scenario._ Exceptions arise for smaller and very large models, suggesting that GPU efficiency rankings are largely consistent but sensitive to model size. GPUs with lower TDP and moderate performance, such as the L4 and the A30, can achieve a lower energy per token than more powerful GPUs for running inference with small models, while large models take advantage of powerful GPUs, offsetting their higher power draw with high throughput. Overall, energy efficiency depends on the specific LLM-GPU combination, with model properties interacting with GPU architecture properties; mainly TDP, memory bandwidth, and TFLOPS. These findings demonstrate that energy-efficient LLM inference depends on hardwareaware model placement, motivating further analysis of how model architectural properties affect energy efficiency. 

## **5.2. RQ2: LLM Efficiency Across GPUs** 

_What model properties (such as size, model family, etc.) impact energy efficiency significantly during inference across GPUs?_ 

RQ2 examines how LLM architectural properties affect energy efficiency across GPUs. As in RQ1, we analyze batch inference data in RQ2, as it provides a cleaner signal for correlating model architectural properties with energy efficiency. RQ1 has already identified model size as a factor that affects energy efficiency and throughput across GPUs (Figures 2, 3). When looking at each LLM-GPU combination, energy per token strongly varies spanning nearly three orders of magnitude (from 0 _._ 003 _𝐽_ to 1 _𝐽_ ). 

Figure 4 shows the mean energy per token against the number of _active parameters_ for each LLM–GPU pair across all evaluated configurations. We use active rather than total parameters to ensure comparability between dense and mixture-of-experts (MoE) architectures, as energy consumption during inference is driven by the parameters involved in the operations per token rather than the full model capacity. For dense models, active and total parameters are equivalent. For MoE models, active parameters correspond to the number of parameters engaged per forward pass – i.e., the parameters of the selected experts plus the shared nonMoE layers – as reported in the respective model cards or original publications. The full list of models with their total and active parameter counts is provided in Table 11. 

Previous works [6, 19] have observed a sublinear relationship in the increase of energy with respect to active parameters, so we fit a log–log regression line to capture this relationship in Figure 4, with the grey area showing the 95% prediction interval of the fitted log–log regression. The remaining outliers arise from implementation-specific (e.g., 

overflow behavior or reliance on ad hoc library versions) or hardware-related factors (limited support in the Tesla V100 and T4 for modern fused kernels leads to higher energy consumption.) More details are provided in the appendix. 

We further analyze the effect of model size and architectural features on energy with two mixed-effects models: Model A quantifies the effect of model size on energy, and Model B adds architectural features to isolate their independent contributions beyond size. 

_Model A._ To quantify the effect of size on energy consumption, we use a mixed-effects model with a random intercept per GPU, regressing log( _𝐸𝑖,𝑗_ ) on log( _𝑃𝑖_ ) while controlling for model type (i.e., model family architecture.) The results reveal a strong and highly significant scaling relationship, where the coefficient for log( _𝑃𝑖_ ) is 0 _._ 237 ( _𝑝<_ 0 _._ 001), i.e., a 10× increase in active parameters approximately increases 1 _._ 7× the energy per token. 

_Model B._ Furthermore, Figure 4 shows that many models of similar size exhibit substantially different energy per token even on the same GPU, suggesting that architectural properties beyond parameter count influence energy consumption. To quantify their effects while accounting for systematic differences across GPUs, we use another mixedeffects model. Based on prior work [6, 16], we focus on a set of architectural properties expected to have a first-order impact on energy usage: (i) _key-value head count_ , i.e., the number of heads in the key-value cache of the attention mechanism, (ii) _attention head count_ , i.e., the number of parallel attention heads, (iii) _hidden size_ , i.e., the width of the model’s internal representations, and (iv) _number of layers_ , i.e., the model depth. 

However, architectural features such as depth, width, and attention structure are strongly correlated with model size, so directly including them alongside log( _𝑃𝑖_ ) would introduce multicollinearity. To isolate the effect of architectural properties independently of size, we residualize each numerical property with respect to log( _𝑃𝑖_ ) before running the mixedeffects model. Concretely, for each architectural feature _𝑍𝑖_ , we estimate a linear regression of the form 

### _𝑍𝑖_ = _𝛼_ 0 + _𝛼_ 1 log( _𝑃𝑖_ ) + _𝑟𝑖,_ 

and use the residual _𝑟𝑖_ – representing the variation in the architectural feature not explained by size – in the final model. The resulting mixed-effects model has the following form: 



$$
log(𝐸𝑖,𝑗) = 𝛽0 + 𝛽1 log(𝑃𝑖) + 𝜸⊤𝐑𝑖+ 𝑢𝑗+ 𝜖𝑖,𝑗 (1)
$$

where 

_𝐸𝑖,𝑗_ is the energy per token of model _𝑖_ on GPU _𝑗_ , _𝛽_ 0 is the baseline log-energy per token, _𝛽_ 1 captures the Empirical Energy Scaling Law, **𝐑** _𝑖_ =<sup><u>[</u></sup> _𝐿𝑖_<sup>⟂</sup> _𝐻𝑖_<sup>⟂</sup> _𝐴_<sup>⟂</sup> _𝑖 𝐾𝑖_<sup>⟂</sup> _𝑇𝑖_ <u>]</u> _,_ are residualized 

Page 7 of 13 

: _Preprint submitted to Elsevier_ 



architectural features orthogonal to log( _𝑃𝑖_ ) _, 𝑇𝑖_ is the model type (categorical), 

_𝑢𝑗_ is the random intercept for GPU _𝑗_ , 

_𝜖𝑖,𝑗_ is the residual error. 

In addition, numerical architectural features are standardized to zero mean and unit variance before model fitting, to allow the fixed-effect coefficients to be interpreted as relative importance for energy efficiency. 

The results confirm that size is the greatest contributor, with coefficient 0 _._ 282 ( _𝑝<_ 0 _._ 001), and show that several architectural characteristics significantly affect energy consumption beyond size: 

- **Key-value head count** has the strongest independent association with higher energy ( _𝛽_ = 0 _._ 121, _𝑝<_ 0 _._ 001), so configurations with more key-value heads are systematically more energy-intensive. 

- **Attention head count** also increases energy consumption ( _𝛽_ = 0 _._ 070, _𝑝_ = 0 _._ 004). 

- **Hidden size** shows a negative conditional effect ( _𝛽_ = −0 _._ 098, _𝑝_ = 0 _._ 026), suggesting that, for a fixed parameter count, models that are wider may be relatively more energy-efficient, potentially reflecting improved hardware utilization. 

- **Number of layers** shows a weaker, marginally significant positive association ( _𝛽_ = 0 _._ 028, _𝑝_ = 0 _._ 062). 

Finally, the estimated random-effect variance (Group Var = 0.072, reported in Table 10) confirms that GPUspecific factors – such as GPU architecture, memory bandwidth, TFLOPs – introduce non-negligible baseline differences, justifying the use of mixed-effects modeling to account for GPU-specific baselines. 

**Answering RQ2** , our analysis reveals two key insights. First, **model size is the dominant factor that influences** energy consumption, following a sublinear scaling relationship (exponents 0.237 and 0.282, both with _𝑝<_ 0 _._ 001). Second, **model architecture design introduces deviations** around this baseline. Key-value and attention head counts increase energy consumption while wider hidden dimensions reduce it, showing that parameter allocation across the model architecture materially influences inference energy efficiency. 

This reinforces our claim that LLM inference energy efficiency is fundamentally a systems-level problem: both model architectural properties and GPU specifications contribute to energy cost. Consequently, optimizing inference efficiency cannot rely on model design alone, but requires software–hardware co-design and hardware-aware model placement to achieve meaningful energy savings. 

## **5.3. RQ3: Deployment Scenario Sensitivity** 

_How does the deployment scenario (batch vs. server load) influence the energy efficiency of LLMs on a given GPU? Are the same LLM–GPU pairs optimal across scenarios?_ 

So far, our analysis has focused on batch scenarios. However, LLMs are often used in server deployment scenarios, where resources are continuously awaiting incoming requests, producing distinct energy consumption patterns. Server scenarios introduce idle periods driven by the stochastic arrival of requests. In our data, mean idle time rises from 2% in the batch scenario to 53% (high load) and 96% (low load) in server scenarios. Consequently, as noted in Section 3.2.2, we evaluate energy efficiency using mean power draw, as it captures consumption in active and idle periods. 

Figure 5 shows power draw across GPUs, model sizes and scenarios, including TDP and idle power. TDP values are sourced from manufacturer documentation, while idle power is estimated as the mean power draw recorded by NVML prior to each inference experiment, averaged across all runs for each GPU. Mean power draw approaches the maximum in the batch scenario, while converging to idle power in server high- and low-load, especially for smaller models. This pattern highlights the importance of idle power draw in underutilized systems, which unfortunately is rarely disclosed by manufacturers, reporting TDP instead. As Figure 5 shows, while TDP is a useful proxy for mean power draw comparisons in batch scenarios, it is a poor proxy in server scenarios. 

Table 4 shows the energy efficiency rankings across GPUs in the server scenario, based on mean power draw. High load rankings favor the L4 (lowest mean power draw for 19 models) and the T4 (14); low load favors the T4 (14 models) and the A30 (11); in both cases the H100 ranks third. Idle and maximum power drive these results: the L4, T4, and A30 have very low idle and max power draws (L4: 17W, 73W, T4: 12W, 82W, A30: 27W, 171W). The H100 with a relatively high idle power (59W) and midrange max power draw (416W) benefits from its throughput advantage – processing requests quickly reduces active time and therefore mean power draw – outperforming other GPUs such as the L40S with a lower max power draw (357W) but lower throughput. This shows that throughput still influences energy efficiency in server scenarios, even though its effect is secondary to idle and max power. 

In real-world settings, response time requirements are critical. Figure 6 plots mean power draw versus the 95th percentile of Time-to-First-Token (TTFT) as measured by vLLM, defined as the time from the arrival of the request until the first token is generated (equal to the sum of queue and prefill time), for all models in Watt Counts, a common metric for evaluating response time in server scenarios. The T4 and L4 GPUs achieve the lowest power draw for most models but their high TTFTs limit their use in real-word applications. Conversely, the H200 consistently achieves the lowest TTFTs but with the highest mean power draw. In addition, high load scenarios require more energy and exhibit higher TTFTs than low load scenarios across all GPUs. A practical approach for energy efficiency is to define a response time requirement and select the GPU with the lowest mean power draw that satisfies it, given the model 

Page 8 of 13 

: _Preprint submitted to Elsevier_ 



**Table 4** 

GPU ranking positions based on energy efficiency under server inference. 

|GPU|1st|2nd|3rd|
|---|---|---|---|
|**Server High Load**||||
|A100 SXM4 40 GB|6|1|7|
|A30 PCIe|1|19|14|
|GeForce RTX 4090|0|0|1|
|H100 NVL 94 GB|9|6|4|
|H200 NVL|0|7|5|
|L4|19|13|1|
|L40S|0|1|1|
|Tesla T4|14|1|1|
|Tesla V100|1|2|9|
|**Server Low Load**||||
|A100 SXM4 40 GB|8|0|0|
|A30 PCIe|11|12|9|
|GeForce RTX 3090|0|0|1|
|GeForce RTX 4090|5|12|15|
|H100 NVL 94 GB|8|5|3|
|H200 NVL|0|8|1|
|L4|4|10|8|
|L40S|0|0|5|
|Tesla T4|14|0|1|
|Tesla V100|0|3|0|



and expected request load. For illustration purposes, we could define this TTFT threshold of 70 _𝑚𝑠_ based on humancomputer interaction research, which identifies 69 _𝑚𝑠_ as the upper limit for a system response to be perceived as instantaneous by a user after tapping a button [44]. 

Tables 5 and 6 exhibit the mean power and 95th percentile of TTFT across model size categories and GPUs for the low load and high load server scenarios, respectively. By selecting the GPU with the lowest mean power draw and response time requirement (TTFT _<_ 70 _𝑚𝑠_ ), it is possible to save significant energy depending on the model size and scenario. For instance, for deploying a medium-sized model such as Llama 3.1 8B, it is possible to save up to 70% of energy by using an A30 compared to the H200, the highestthroughput GPU in our study, or 60% when compared to the current industry standard, the H100 in the low load scenario. Another interesting case emerges for high load scenarios and large models, where the older-generation A100 achieves 10% energy savings over the newer H100, suggesting that newer GPU generations do not always translate to better energy efficiency across all deployment scenarios. Our results show that energy-optimal GPU selection is highly deployment-specific, offering system operators actionable trade-offs between performance and energy. 

**Answering RQ3:** _the deployment scenario strongly influences the energy efficiency and GPU rankings. In the server scenario, idle and max power of the GPU are main considerations while throughput becomes secondary for energy consumption, though it remains critical for meeting response time requirements._ Moreover, TDP is a poor proxy for 

**Table 5** 

Tail latency (TTFT p95) per experiment, averaged per model, then averaged per size category. Scenario: Server Low Load. The GPU with the lowest mean power per category that satisfies the requirement of a TTFT _<_ 70 _𝑚𝑠_ is in bold text. 

|GPU|TTFT|p95 (ms)|Mean Power (W)|
|---|---|---|---|
|**Small**||||
|A100 SXM4 40 GB|12||67.4|
|A30 PCIe|33||32.3|
|GeForce RTX 3090|57||52.3|
|**GeForce RTX 4090**|**21**||**31.9**|
|H100 NVL 94 GB|17||90.8|
|H200 NVL|08||120.3|
|L4|54||33.1|
|L40S|17||113.6|
|Tesla T4|95||29.6|
|Tesla V100|28||66.2|
|**Medium**||||
|A100 SXM4 40 GB|26||70.7|
|**A30 PCIe**|**60**||**38.3**|
|GeForce RTX 3090|88||65.2|
|GeForce RTX 4090|39||41.1|
|H100 NVL 94 GB|29||94.7|
|H200 NVL|13||125.1|
|L4|132||38.8|
|L40S|42||123.5|
|Tesla V100|49||72.8|
|**Large**||||
|**A100 SXM4 40 GB**|**31**||**71.1**|
|H100 NVL 94 GB|24||96.6|
|H200 NVL|13||124.7|
|L40S|53||123.1|
|**Xlarge**||||
|**H100 NVL 94 GB**|**27**||**101.7**|
|H200 NVL|21||131.5|



power draw in server scenarios and idle power, an essential characteristic for assessing the energy efficiency of GPUs in this scenario, is rarely reported by manufacturers. 

# **6. Discussion & Limitations** 

**Single-GPU scope.** Our current work is limited to LLMs deployed in single-GPU configurations, common in edge, on-prem, and cost- or energy-constrained inference systems. While this scope allows us to thoroughly characterize GPUlevel power dynamics across heterogeneous architectures in isolation – avoiding confounding effects of communication overhead, tensor-parallel synchronization, and interconnect bandwidth variability – future work should extend this analysis to multi-GPU deployments, where additional system-level effects may alter the energy efficiency rankings observed here. Nonetheless, the single-GPU insights are directly applicable to the growing class of edge-cloud and resource-constrained deployments. 

Page 9 of 13 

: _Preprint submitted to Elsevier_ 



**Table 6** 

Tail latency (TTFT p95) per experiment, averaged per model, then averaged per size category. Scenario: Server high load. The GPU with the lowest mean power per category that satisfies the requirement of a TTFT _<_ 70 _𝑚𝑠_ is in bold text. 

|GPU|TTFT p95 (ms)|Mean Power (W)|
|---|---|---|
|**Small**|||
|A100 SXM4 40 GB|17|112.7|
|A30 PCIe|31|72.5|
|GeForce RTX 3090|25|198.0|
|GeForce RTX 4090|23|121.1|
|H100 NVL 94 GB|18|119.9|
|H200 NVL|09|151.7|
|**L4**|**67**|**55.7**|
|L40S|25|166.6|
|Tesla T4|125|54.2|
|Tesla V100|41|119.9|
|**Medium**|||
|A100 SXM4 40 GB|48|172.9|
|A30 PCIe|81|121.5|
|GeForce RTX 3090|73|275.8|
|GeForce RTX 4090|56|208.8|
|**H100 NVL 94 GB**|**39**|**165.9**|
|H200 NVL|21|212.3|
|L4|203|68.7|
|L40S|74|237.6|
|Tesla V100|100|171.3|
|**Large**|||
|**A100 SXM4 40 GB**|**59**|**179.0**|
|H100 NVL 94 GB|29|200.5|
|H200 NVL|22|226.0|
|L40S|103|236.4|
|**Xlarge**|||
|**H100 NVL 94 GB**|**49**|**275.3**|
|H200 NVL|40|336.7|



**Precision and quantization.** All evaluated models use 16-bit weights to control the dimensionality of the experimental space. Quantized models (e.g., INT8, INT4) are increasingly common in practical deployments and are known to alleviate total memory usage, memory bandwidth bottlenecks, and improve throughput, which could shift the energy efficiency ranks observed here, in particular for GPUs where memory bandwidth is their primary constraint. However, we expect the qualitative insights to generalize (e.g., total energy being the result of both GPU and model properties, the importance of memory bandwidth, the effect of model architecture properties on energy consumption, the role of idle power in server scenarios), the specific GPU rankings and model architectural properties coefficients may differ for quantized workloads. Extending Watt Counts to quantized models is a natural and important direction for future work, and the open benchmark is designed to support exactly this kind of community contribution. 

**Inference engine.** We adopt vLLM, a state-of-theart, widely adopted inference engine that has been shown to achieve leading efficiency in scalable settings [36, 35, 28]. This choice ensures that observed energy–performance trade-offs primarily reflect architectural and workload characteristics rather than implementation inefficiencies. However, results may differ with other engines such as TensorRTLLM or SGLang, particularly for specific GPU-model combinations where engine-specific optimizations may have outsized effects. 

**Energy versus accuracy trade-offs.** Task accuracy and fairness are out of scope in this work, as our goal is comparative characterization of energy consumption across hardware and deployment scenarios rather than model evaluation. However, we recognize that optimizing deployments solely for energy may introduce trade-offs with output quality. These considerations should be incorporated in future multiobjective deployment frameworks that jointly optimize energy, latency, and accuracy. 

**Host consumption and cooling.** Our measurements focus on GPU energy consumption, which we validate as the dominant component (78.7–92.5%) and the one with the highest Pearson correlation against the total energy consumed across scenarios (0.995-1.0) . Total host consumption, including CPU, DRAM, storage, and cooling overhead, is not captured. In real-world data center deployments, cooling can add 30–50% to total energy consumption depending on the power usage effectiveness (PUE) of the facility, and should be accounted for in full system-level carbon analyses. 

**Carbon footprint of this work.** The dataset creation and experiments were run in data centers powered from the grid in Switzerland, Spain, and Czechia, consuming approximately 180 kWh with estimated emissions of 17 kgCO2. In particular, most experiments were run on servers sourced with a 99% share of renewable energies. We believe that the potential energy savings enabled by reducing redundant experimentation across the community, and by enabling more informed hardware selection and deployment decisions, significantly outweigh this one-time footprint. 

# **7. Conclusion** 

This work presents Watt Counts, a large-scale, open, energy-aware dataset and an empirical study of energy efficiency in LLM inference across 50 models and 10 heterogeneous GPUs in batch and server scenarios. Our benchmark and dataset provide foundational resources for hardwareaware deployment decisions in heterogeneous LLM inference systems, while our findings yield insights for hardwareaware placement and more general energy-efficient, sustainable LLM inference. 

First, **GPU energy efficiency rankings are not universal – they vary significantly with model size and deployment scenario** . The H100 dominates in batch scenarios for most models, but low power GPUs such as the L4 and A30 outperform it for small models, and server scenarios alter rankings entirely. This highlights the importance of 

Page 10 of 13 

: _Preprint submitted to Elsevier_ 



hardware-aware model placement. Second, **while model size is the primary factor in energy consumption, architectural properties such as depth, hidden size, and key–value head configuration have significant effects** . As a consequence, models with similar sizes significantly vary in their energy consumption. Third, **we show that TFLOPS per Watt is a poor proxy for energy efficiency at the system level across full inference workloads** , as one of the main bottlenecks in LLM inference is memory bandwidth. 

Fourth, within our evaluation scope, **the dominant factors driving energy efficiency differ fundamentally between batch (offline) and server (online) deployments** . In batch scenarios, where energy is attributed to a finite workload, throughput and TDP are the main factors. In server scenarios, where the system runs continuously, idle and max power dominate. However, these are rarely disclosed by manufacturers, who report TDP instead. This has direct implications for hardware selection in production LLM deployments. Finally, **energy efficiency must be jointly considered with service-level requirements** , such as response time, as GPUs with low power draw might suffer from high response times, limiting their use in practical applications, underscoring the need for multi-objective evaluation. 

We release the Watt Counts dataset to support future work on heterogeneous deployments, scheduling, and benchmarking for scalable and sustainable LLM inference. 

# **CRediT authorship contribution statement** 

**Mauricio Fadel Argerich:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, Writing. **Jonathan Fürst:** Methodology, Software, Writing. **Marta Patiño-Martínez:** Conceptualization, Supervision, Writing. 

# **8. Declaration of competing interest** 

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

# **9. Use of Generative AI** 

During the preparation of this manuscript, the authors used generative AI tools, including Claude (Anthropic) and ChatGPT (OpenAI), to assist with language refinement and clarity. All AI-generated suggestions were carefully reviewed and edited by the authors, who take full responsibility for the accuracy, integrity, and originality of the published work. 

# **10. Research data availability** 

The Watt Counts dataset and benchmark will be publicly released on Github with an MIT License upon acceptance and are submitted as supplementary material. 

# **References** 

- [1] Y. Jernite, S. Luccioni, Compute and competition in ai: Different flops for different folks, hugging Face Community Article (Feb. 2026). URL https://huggingface.co/blog/sasha/compute-competition 

- [2] J. You, How much energy does chatgpt use?, Gradient Updates (Epoch AI newsletter), accessed: 2025-07-31 (Feb. 2025). URL https://epoch.ai/gradient-updates/ how-much-energy-does-chatgpt-use 

- [3] G. Yang, C. He, J. Guo, J. Wu, Y. Ding, A. Liu, H. Qin, P. Ji, X. Liu, Llmcbench: Benchmarking large language model compression for efficient deployment, Advances in Neural Information Processing Systems 37 (2024) 87532–87544. 

- [4] V. Burian, A. Stalla-Bourdillon, The increasing energy demand of artificial intelligence and its impact on commodity prices, Economic Bulletin, _Focus_ Box 2/2025, European Central Bank, accessed: 2025-07-31 (May 2025). 

   - URL https://www.ecb.europa.eu/press/economic-bulletin/focus/ 2025/html/ecb.ebbox202502_03~8eba688e29.en.html 

- [5] D. Xue, Strategies for mitigating the global energy and carbon impact of artificial intelligence, United Nations SDGs (2023) 2023–05. 

- [6] M. F. Argerich, M. Patiño-Martínez, Measuring and improving the energy efficiency of large language models inference, IEEE Access (2024). 

- [7] G. Theodorou, S. Karagiorgou, C. Kotronis, On energy-aware and verifiable benchmarking of big data processing targeting ai pipelines, in: 2024 IEEE International Conference on Big Data (BigData), IEEE, 2024, pp. 3788–3798. 

- [8] V. Liu, Y. Yin, Green ai: exploring carbon footprints, mitigation strategies, and trade offs in large language model training, Discover Artificial Intelligence 4 (1) (2024) 49. 

- [9] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So, M. Texier, J. Dean, Carbon emissions and large neural network training, arXiv preprint arXiv:2104.10350 (2021). 

- [10] J. McDonald, B. Li, N. Frey, D. Tiwari, V. Gadepally, S. Samsi, Great power, great responsibility: Recommendations for reducing energy for training language models, arXiv preprint arXiv:2205.09646 (2022). 

- [11] S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones, W. Bergeron, J. Kepner, D. Tiwari, V. Gadepally, From words to watts: Benchmarking the energy costs of large language model inference, in: 2023 IEEE High Performance Extreme Computing Conference (HPEC), IEEE, 2023, pp. 1–9. 

- [12] G. Bai, Z. Chai, C. Ling, S. Wang, J. Lu, N. Zhang, T. Shi, Z. Yu, M. Zhu, Y. Zhang, et al., Beyond efficiency: A systematic survey of resource-efficient large language models, arXiv preprint arXiv:2401.00625 (2024). 

- [13] J. Morrison, C. Na, J. Fernandez, T. Dettmers, E. Strubell, J. Dodge, Holistically evaluating the environmental impact of creating language models, arXiv preprint arXiv:2503.05804 (2025). 

- [14] C.-J. Wu, R. Raghavendra, U. Gupta, B. Acun, N. Ardalani, K. Maeng, G. Chang, F. Aga, J. Huang, C. Bai, et al., Sustainable ai: Environmental implications, challenges and opportunities, Proceedings of Machine Learning and Systems 4 (2022) 795–813. 

- [15] N. Jegham, M. Abdelatti, C. Y. Koh, L. Elmoubarki, A. Hendawi, How hungry is ai? benchmarking energy, water, and carbon footprint of llm inference, arXiv preprint arXiv:2505.09598 (2025). 

- [16] S. Kim, C. Hooper, T. Wattanawong, M. Kang, R. Yan, H. Genc, G. Dinh, Q. Huang, K. Keutzer, M. W. Mahoney, et al., Full stack optimization of transformer inference: a survey, arXiv preprint arXiv:2302.14017 (2023). 

- [17] A. Tschand, A. T. R. Rajan, S. Idgunji, A. Ghosh, J. Holleman, C. Kiraly, P. Ambalkar, R. Borkar, R. Chukka, T. Cockrell, et al., Mlperf power: Benchmarking the energy efficiency of machine learning systems from _𝜇_ watts to mwatts for sustainable ai, in: 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), IEEE, 2025, pp. 1201–1216. 

- [18] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, G. Schmuelling, C.-J. Wu, B. Anderson, M. Breughe, M. Charlebois, W. Chou, R. Chukka, C. Coleman, S. Davis, P. Deng, G. Diamos, J. Duke, D. Fick, J. S. 

Page 11 of 13 

: _Preprint submitted to Elsevier_ 



Gardner, I. Hubara, S. Idgunji, T. B. Jablin, J. Jiao, T. S. John, P. Kanwar, D. Lee, J. Liao, A. Lokhmotov, F. Massa, P. Meng, P. Micikevicius, C. Osborne, G. Pekhimenko, A. T. R. Rajan, D. Sequeira, A. Sirasao, F. Sun, H. Tang, M. Thomson, F. Wei, E. Wu, L. Xu, K. Yamada, B. Yu, G. Yuan, A. Zhong, P. Zhang, Y. Zhou, Mlperf inference benchmark (2019). arXiv:1911.02549. 

- [19] C. Niu, W. Zhang, J. Li, Y. Zhao, T. Wang, X. Wang, Y. Chen, Tokenpowerbench: Benchmarking the power consumption of llm inference, arXiv preprint arXiv:2512.03024 (2025). 

- [20] L. F. W. Anthony, B. Kanding, R. Selvan, Carbontracker: Tracking and predicting the carbon footprint of training deep learning models, arXiv preprint arXiv:2007.03051 (2020). 

- [21] V. Schmidt, K. Goyal, A. Joshi, B. Feld, L. Conell, N. Laskaris, D. Blank, J. Wilson, S. Friedler, S. Luccioni, Codecarbon: estimate and track carbon emissions from machine learning computing (2021), DOI: https://doi. org/10.5281/zenodo 4658424 (2021). 

- [22] J. You, J.-W. Chung, M. Chowdhury, Zeus: Understanding and optimizing GPU energy consumption of DNN training, in: USENIX NSDI, 2023. 

- [23] J.-W. Chung, J. J. Ma, R. Wu, J. Liu, O. J. Kweon, Y. Xia, Z. Wu, M. Chowdhury, The ML.ENERGY benchmark: Toward automated inference energy measurement and optimization, in: NeurIPS Datasets and Benchmarks, 2025. 

- [24] J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, E. Choukse, Dynamollm: Designing llm inference clusters for performance and energy efficiency, in: 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), IEEE, 2025, pp. 1348–1362. 

- [25] Z. Fu, F. Chen, S. Zhou, H. Li, L. Jiang, Llmco2: Advancing accurate carbon footprint prediction for llm inferences, ACM SIGENERGY Energy Informatics Review 5 (2) (2025) 63–68. 

- [26] A. K. Kakolyris, D. Masouros, P. Vavaroutsos, S. Xydis, D. Soudris, throttll’em: Predictive gpu throttling for energy efficient llm inference serving, in: 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), IEEE, 2025, pp. 1363–1378. 

- [27] P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, R. Bianchini, Characterizing power management opportunities for llms in the cloud, in: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 2024, pp. 207–222. 

- [28] J. Fernandez, C. Na, V. Tiwari, Y. Bisk, S. Luccioni, E. Strubell, Energy considerations of large language model inference and efficiency optimizations, in: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 32556–32569. 

- [29] G. Wilkins, S. Keshav, R. Mortier, Offline energy-optimal llm serving: Workload-based energy models for llm inference on heterogeneous systems, ACM SIGENERGY Energy Informatics Review 4 (5) (2024) 113–119. 

- [30] A. Agrawal, N. Kedia, J. Mohan, A. Panwar, N. Kwatra, B. S. Gulavani, R. Ramjee, A. Tumanov, Vidur: A large-scale simulation framework for llm inference, Proceedings of Machine Learning and Systems 6 (2024) 351–366. 

- [31] E. J. Husom, A. Goknil, L. K. Shar, S. Sen, The price of prompting: Profiling energy use in large language models inference, arXiv preprint arXiv:2407.16893 (2024). 

- [32] C. Guo, S. Wang, R. Xie, J. Song, Estimating energy consumption of neural networks with joint structure–device encoding, Sustainable Computing: Informatics and Systems 45 (2025) 101062. 

- [33] S. Luccioni, B. Gamazaychikov, E. Strubell, S. Hooker, Y. Jernite, M. Mitchell, S. Chamberlin, Ai energy score leaderboard - december 2025, https://huggingface.co/spaces/AIEnergyScore/Leaderboard (2025). 

- [34] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, I. Stoica, Efficient memory management for large language model serving with pagedattention, in: Proceedings of the 29th symposium on operating systems principles, 2023, pp. 611–626. 

- [35] L. Stuhlmann, M. F. Argerich, J. Fürst, Bench360: Benchmarking local llm inference from 360 {∖deg}, arXiv preprint arXiv:2511.16682 

(2025). 

- [36] S. Park, S. Jeon, C. Lee, S. Jeon, B.-S. Kim, J. Lee, A survey on inference engines for large language models: Perspectives on optimization and efficiency, arXiv preprint arXiv:2505.01658 (2025). 

- [37] R. Gond, A. K. Kamath, A. Basu, R. Ramjee, A. Panwar, Llm42: Enabling determinism in llm inference with verified speculation, arXiv preprint arXiv:2601.17768 (2026). 

- [38] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, G. Schmuelling, C.-J. Wu, B. Anderson, M. Breughe, M. Charlebois, W. Chou, et al., Mlperf inference benchmark, in: 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), IEEE, 2020, pp. 446– 459. 

- [39] G. Colakoglu, G. Solmaz, J. Fürst, Problem solved? information extraction design space for layout-rich documents using LLMs, in: C. Christodoulopoulos, T. Chakraborty, C. Rose, V. Peng (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, Association for Computational Linguistics, Suzhou, China, 2025, pp. 17908–17927. doi:10.18653/v1/2025.findings-emnlp.973. URL https://aclanthology.org/2025.findings-emnlp.973/ 

- [40] P. Rajpurkar, J. Zhang, K. Lopyrev, P. Liang, Squad: 100,000+ questions for machine comprehension of text, arXiv preprint arXiv:1606.05250 (2016). 

- [41] Z. Yang, K. Adamek, W. Armour, Accurate and convenient energy measurements for gpus: A detailed study of nvidia gpu’s built-in power sensor, in: SC24: International Conference for High Performance Computing, Networking, Storage and Analysis, IEEE, 2024, pp. 1–17. 

- [42] P. J. Maliakel, S. Ilager, I. Brandic, Investigating energy efficiency and performance trade-offs in llm inference across tasks and dvfs settings, arXiv preprint arXiv:2501.08219 (2025). 

- [43] P. G. Recasens, F. Agullo, Y. Zhu, C. Wang, E. K. Lee, O. Tardieu, J. Torres, J. L. Berral, Mind the memory gap: Unveiling gpu bottlenecks in large-batch llm inference, arXiv preprint arXiv:2503.08311 (2025). 

- [44] J. Deber, R. Jota, C. Forlines, D. Wigdor, How much faster is fast enough? user perception of latency & latency improvements in direct and indirect touch, in: Proceedings of the 33rd annual acm conference on human factors in computing systems, 2015, pp. 1827–1836. 

- [45] S. Kim, D. Kim, C. Park, W. Lee, W. Song, Y. Kim, H. Kim, Y. Kim, H. Lee, J. Kim, et al., Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling, in: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), 2024, pp. 23–35. 

# **A. Full List of Evaluated Large Language Models** 

We list the 50 models evaluated in our experiments in Table 11, classifying them by their model category as discussed in Section 5.1. 

# **B. Validation of GPU as the Primary Energy Consumer** 

We utilize EnergyMeter to measure the energy consumption of CPU and memory DRAM with Intel RAPL and the energy consumption of the GPU with NVML on a bare metal instance (Intel(R) Core(TM) i9-9900K @ 3.60GHz, 32 GB RAM, a single NVIDIA GeForce RTX 4090), calculating the share of energy represented by the CPU, DRAM, and GPU over the total as well as measuring their empirical correlation in our experiments. Table 7 shows the results. 

Page 12 of 13 

: _Preprint submitted to Elsevier_ 



**Table 7** 

Mean energy share and Pearson correlation with total system energy consumption per component and deployment scenario in the bare metal instance with RAPL access. 

|**Scenario**|**Component**|**Share (%)**|**Corr. with Total**|
|---|---|---|---|
|Batch|GPU|92.5|1.000|
||CPU|7.0|0.989|
||DRAM|0.5|0.989|
|Server high|GPU|91.2|1.000|
||CPU|7.4|0.995|
||DRAM|1.4|0.127|
|Server Low|GPU|78.7|0.995|
||CPU|16.6|0.575|
||DRAM|4.6|0.351|



GPU consumes 92.5-78.7% of the total energy consumption across all scenarios, with the highest correlation in all cases. 

# **C. Outlier Analysis: Energy per Token vs. Model Size** 

This section details the outliers identified in the log-log regression of energy per token against model size discussed in Section 5.2. Outliers exhibit clear structural patterns and are shown in Table 8. Most outliers come from models running on the Tesla V100 and Tesla T4, older GPUs and possess limited memory bandwidth and reduced support for modern kernels leading to disproportionately high energy per token. 

In addition, two models emerge as outliers on modern GPUs apart from the V100 and T4. SmolLM2-1.7B-Instruct appears as an outlier on the L40S, RTX 3090, T4, and Tesla V100, consuming disproportionately high energy relative to its 1.7B parameter count. We hypothesize this is related to architecture-specific kernel inefficiencies on these devices, as the model performs within expectations on newer Ada Lovelace and Hopper GPUs. SOLAR-10.7B-Instruct-v1.0 is another case of higher energy consumption than expected by our log-log model, appearing as an outlier on the RTX 3090, RTX 4090, and A30. This model uses a depth-upscaling architecture that duplicates specific transformer layers [45], increasing the depth of the original model (e.g., Mistral 7B) and its energy consumption as discussed in section 5.2. In addition, we hypothesize the limited memory capacity of these GPUs (24GB) combined with the model size (10.7B parameters estimated to use 21GB of RAM for its weights only), cause excessive memory pressure and degraded throughput that increases energy per token. 

**Table 8** 

Log-log energy per token vs model size outliers 

|GPU|Model|Params (B)|Mean Energy/Token|
|---|---|---|---|
|Tesla V100 SXM2 32 GB<br>|allenai/OLMoE-1B-7B-0924<br>|1.000000<br>|0.256803<br>|
|Tesla T4|microsoft/Phi-tiny-MoE-instruct|1.100000|0.135182|
|Tesla V100 SXM2 32 GB|microsoft/Phi-tiny-MoE-instruct|1.100000|0.140642|
|L40S|HuggingFaceTB/SmolLM2-1.7B-Instruct|1.700000|0.113926|
|GeForce RTX 3090|HuggingFaceTB/SmolLM2-1.7B-Instruct|1.700000|0.143895|
|Tesla T4|HuggingFaceTB/SmolLM2-1.7B-Instruct|1.700000|0.129636|
|Tesla V100 SXM2 32 GB|HuggingFaceTB/SmolLM2-1.7B-Instruct|1.700000|0.176601|
|Tesla T4|microsoft/phi-2|2.700000|0.151607|
|Tesla V100 SXM2 32 GB|ibm-granite/granite-3.1-3b-a800m-instruct<br>|3.000000|0.172620|
|Tesla T4|ibm-granite/granite-3.1-3b-a800m-instruct|3.000000|0.190443|
|Tesla T4|EleutherAI/gpt-j-6b|6.050000|0.305580|
|GeForce RTX 3090|upstage/SOLAR-10.7B-Instruct-v1.0<br>|10.700000|0.971666|
|GeForce RTX 4090|upstage/SOLAR-10.7B-Instruct-v1.0|10.700000|0.490263|
|A30 PCIe|upstage/SOLAR-10.7B-Instruct-v1.0<br>|10.700000|0.378546|
|Tesla V100 SXM2 32 GB|meta-llama/Llama-2-13b-chat-hf<br>|13.000000|0.630607|
|Tesla V100 SXM2 32 GB<br>|microsoft/Phi-3-medium-4k-instruct<br>|14.000000|0.546969|
|Tesla V100 SXM2 32 GB|allenai/OLMo-2-1124-13B-Instruct|14.000000|0.986962|



#### **Table 9** 

Mixed-effects linear regression results for log energy per token (Model A: log parameters + model type). GPU is modeled as a random intercept. Coefficients are reported with standard errors, _𝑧_ -statistics, _𝑝_ -values, and 95% confidence intervals. 

|**Variable**|**Coef.**|**Std.Err.**|**z**|**P**_>_|**z**||**[0.025**|**0.975]**|
|---|---|---|---|---|---|---|
|Intercept|-0.919|0.123|-7.47|0.000|-1.160|-0.678|
|log_params|0.237|0.014|16.93|0.000|0.210|0.264|
|C(model_type)[T.gemma2]|-0.495|0.109|-4.54|0.000|-0.709|-0.281|
|C(model_type)[T.gpt_oss]|-0.219|0.154|-1.42|0.155|-0.521|0.083|
|C(model_type)[T.granite]|-0.409|0.098|-4.17|0.000|-0.601|-0.217|
|C(model_type)[T.granitemoe]|-0.443|0.105|-4.22|0.000|-0.649|-0.237|
|C(model_type)[T.internlm2]|-0.377|0.104|-3.63|0.000|-0.581|-0.173|
|C(model_type)[T.llama]|-0.383|0.092|-4.16|0.000|-0.563|-0.203|
|C(model_type)[T.mistral]|-0.441|0.094|-4.69|0.000|-0.625|-0.257|
|C(model_type)[T.nemotron]|-0.512|0.106|-4.83|0.000|-0.720|-0.304|
|C(model_type)[T.nemotron_h]|0.174|0.154|1.13|0.258|-0.128|0.476|
|C(model_type)[T.olmo2]|-0.167|0.103|-1.62|0.106|-0.369|0.035|
|C(model_type)[T.olmoe]|-0.082|0.108|-0.76|0.448|-0.294|0.130|
|C(model_type)[T.phi]|-0.221|0.105|-2.10|0.036|-0.427|-0.015|
|C(model_type)[T.phi3]|-0.296|0.099|-2.99|0.003|-0.490|-0.102|
|C(model_type)[T.phimoe]|-0.162|0.106|-1.53|0.126|-0.370|0.046|
|C(model_type)[T.qwen2]|-0.598|0.092|-6.50|0.000|-0.778|-0.418|
|C(model_type)[T.qwen2_moe]|-0.520|0.128|-4.06|0.000|-0.771|-0.269|
|C(model_type)[T.qwen3]|-0.438|0.128|-3.42|0.001|-0.689|-0.187|
|C(model_type)[T.solar]|-0.332|0.157|-2.12|0.034|-0.640|-0.024|
|Group Var|0.071||||||



_Model details:_ 335 observations; 10 GPU groups; REML estimation; log-likelihood = 55.81; model converged. 

token. The data points used in each model correspond to the mean computed over five repeated runs for each experimental configuration (i.e., each LLM-GPU-scenario) to reduce measurement noise. 

Table 9 shows the results of mixed-effects model A, which focuses on the effect of active parameters on energy consumption and Table 10 shows the results of model B, which includes the residuals of hidden size, number of layers, number of attention heads, and number of key value heads. 

# **D. Full Mixed-Effects Model Results for LLM Properties and Energy Per Token** 

In this section we include the full results for the mixedmodels presented in Section 5.2, used for understanding the effect of LLM architectural properties on energy per 

Page 13 of 13 

: _Preprint submitted to Elsevier_ 



#### **Table 10** 

Mixed-effects linear regression results for log energy per token (Model B: residualized architectural features). GPU is modeled as a random intercept. Coefficients are reported with standard errors, _𝑧_ -statistics, _𝑝_ -values, and 95% confidence intervals. 

|**Variable**|**Coef.**|**Std.Err.**|**z**|**P**_>_|**z**||**[0.025**|**0.975]**|
|---|---|---|---|---|---|---|
|Intercept|-0.948|0.117|-8.10|0.000|-1.177|-0.719|
|log_params|0.282|0.013|21.69|0.000|0.257|0.307|
|hidden_size_resid|-0.098|0.044|-2.23|0.026|-0.184|-0.012|
|num_layers_resid|0.028|0.015|1.87|0.062|-0.001|0.057|
|num_attention_heads_resid|0.070|0.024|2.92|0.004|0.023|0.117|
|num_key_value_heads_resid|0.121|0.014|8.64|0.000|0.094|0.148|
|C(model_type)[T.gemma2]|-0.270|0.093|-2.90|0.004|-0.452|-0.088|
|C(model_type)[T.gpt_oss]|-0.413|0.177|-2.33|0.020|-0.760|-0.066|
|C(model_type)[T.granite]|-0.421|0.092|-4.58|0.000|-0.601|-0.241|
|C(model_type)[T.granitemoe]|-0.477|0.094|-5.07|0.000|-0.661|-0.293|
|C(model_type)[T.internlm2]|-0.313|0.100|-3.13|0.002|-0.509|-0.117|
|C(model_type)[T.llama]|-0.382|0.083|-4.60|0.000|-0.545|-0.219|
|C(model_type)[T.mistral]|-0.375|0.089|-4.21|0.000|-0.549|-0.201|
|C(model_type)[T.nemotron]|-0.431|0.092|-4.69|0.000|-0.611|-0.251|
|C(model_type)[T.nemotron_h]|0.167|0.139|1.20|0.230|-0.105|0.439|
|C(model_type)[T.olmo2]|-0.408|0.096|-4.25|0.000|-0.596|-0.220|
|C(model_type)[T.olmoe]|0.014|0.101|0.14|0.890|-0.184|0.212|
|C(model_type)[T.phi]|-0.497|0.095|-5.23|0.000|-0.683|-0.311|
|C(model_type)[T.phi3]|-0.419|0.092|-4.55|0.000|-0.599|-0.239|
|C(model_type)[T.phimoe]|0.179|0.150|1.19|0.234|-0.115|0.473|
|C(model_type)[T.qwen2]|-0.448|0.080|-5.60|0.000|-0.605|-0.291|
|C(model_type)[T.qwen2_moe]|-0.591|0.132|-4.48|0.000|-0.850|-0.332|
|C(model_type)[T.qwen3]|-0.397|0.118|-3.36|0.001|-0.628|-0.166|
|C(model_type)[T.solar]|-0.427|0.141|-3.03|0.002|-0.703|-0.151|
|Group Var|0.072||||||



_Model details:_ 335 observations; 10 GPU groups; REML 

estimation; log-likelihood = 104.02; model converged. 

![Figure](assets/figure_0001_page_0014.svg)![Figure](assets/figure_0002_page_0014.svg)**Figure 2:** Mean energy per token for model size categories across GPUs for selected models. Error bars show 95% confidence intervals. Missing bars indicate that the GPU could not run any models in that size category due to memory or compatibility limitations. 

**Figure 1:** Benchmark design overview. The orchestrator launches the System Under Test (SUT) based on a userprovided configuration. The SUT runs in batch mode (vLLM synchronous engine) or server mode (vLLM asynchronous engine with a Poisson-based load generator). The metrics collector aggregates energy, system, and LLM-level performance measurements throughout execution. 

Page 14 of 13 

: _Preprint submitted to Elsevier_ 



![Figure](assets/figure_0003_page_0015.svg)![Figure](assets/figure_0004_page_0015.svg)**Figure 4:** The gray area shows the 95% prediction interval of a log–log regression, log( _𝐸_ ) = _𝛼_ log(size) + _𝛽_ , fitted over all data points; data points outside this envelope are considered energy outliers. 

**Figure 3:** Mean throughput for model size categories across GPUs for selected models. Error bars show 95% confidence intervals. Missing bars indicate that the GPU could not run any models in that size category due to memory or compatibility limitations. 

Page 15 of 13 

: _Preprint submitted to Elsevier_ 



![Figure](assets/figure_0005_page_0016.svg)![Figure](assets/figure_0006_page_0016.svg)**Figure 6:** Scatter plot showing mean GPU power draw versus the 95th percentile of Time-to-First-Token (TTFT), per model and scenario. Each data point is a (model, GPU, scenario) configuration, averaged over five iterations. 

**Figure 5:** GPU power draw across model sizes and scenarios. Bars show the minimum, mean, and maximum power draw for different model size categories in order (S/M/L/XL) across GPUs under batch, server low, and server high load scenarios. TDP and measured idle power are reported in the last row for reference. 

Page 16 of 13 

: _Preprint submitted to Elsevier_ 



#### **Table 11** 

Full list of the 50 LLMs evaluated in this study. Models are grouped by size category (based on minimum GPU memory required) and sorted by total parameter count within each group. MoE models are listed separately as size categories are based on GPU memory fitting of dense models. “GPUs” indicates the number of distinct GPUs each model was evaluated on. 

|**Model**|**Family**|**Arch.**|**Total Params (B)**|**Active Params (B)**|**GPUs**|
|---|---|---|---|---|---|
|**Small models**||||||
|openai-community/gpt2|gpt2<br>|Dense<br>|0.124<br>|0.124<br>|10<br>|
|Qwen/Qwen2.5-0.5B-Instruct|qwen2<br>|Dense<br>|0.490<br>|0.490<br>|10<br>|
|nvidia/AceMath-1.5B-Instruct|qwen2|Dense|1.500|1.500|10|
|Qwen/Qwen2.5-1.5B-Instruct|qwen2<br>|Dense<br>|1.540<br>|1.540<br>|10<br>|
|HuggingFaceTB/SmolLM2-1.7B-Instruct|llama<br>|Dense<br>|1.700<br>|1.700<br>|10<br>|
|ibm-granite/granite-3.0-2b-instruct|granite|Dense|2.000|2.000|10|
|google/gemma-2-2b-it|gemma2<br>|Dense<br>|2.610<br>|2.610<br>|8<br>|
|microsoft/phi-2|phi<br>|Dense<br>|2.700<br>|2.700<br>|10<br>|
|ibm-granite/granite-3.1-3b-a800m-instruct|granitemoe|Dense|3.000|3.000|10|
|tiiuae/Falcon3-3B-Instruct|llama|Dense|3.000|3.000|10|
|Qwen/Qwen2.5-3B-Instruct|qwen2|Dense|3.090|3.090|10|
|meta-llama/Llama-3.2-3B-Instruct|llama|Dense|3.210|3.210|10|
|microsoft/Phi-3-mini-4k-instruct|phi3|Dense|3.800|3.800|10|
|nvidia/Nemotron-Mini-4B-Instruct|nemotron|Dense|4.000|4.000|10|
|nvidia/Llama-3.1-Minitron-4B-Width-Base|llama<br>|Dense<br>|5.000<br>|5.000<br>|10<br>|
|EleutherAI/gpt-j-6b|gptj|Dense|6.050|6.050|10|
|**Medium models**||||||
|01-ai/Yi-1.5-6B-Chat|llama|Dense|6.000|6.000|9|
|mlabonne/NeuralBeagle14-7B|mistral<br>|Dense<br>|7.000<br>|7.000<br>|9<br>|
|berkeley-nest/Starling-LM-7B-alpha|mistral<br>|Dense<br>|7.000|7.000|9|
|mlabonne/AlphaMonarch-7B|mistral<br>|Dense<br>|7.000<br>|7.000<br>|9<br>|
|nvidia/AceMath-7B-Instruct|qwen2|Dense|7.000|7.000|9|
|allenai/OLMo-2-1124-7B-Instruct|olmo2|Dense|7.000|7.000|9|
|deepseek-ai/deepseek-llm-7b-chat|llama|Dense|7.000|7.000|9|
|internlm/internlm2-7b|internlm2|Dense|7.000|7.000|9|
|tiiuae/Falcon3-Mamba-7B-Instruct|falcon_mamba|Dense|7.000|7.000|9|
|mistralai/Mistral-7B-Instruct-v0.3<br>|mistral<br>|Dense<br>|7.250<br>|7.250<br>|9<br>|
|Qwen/Qwen2.5-7B-Instruct|qwen2|Dense|7.620|7.620|9|
|ibm-granite/granite-3.0-8b-instruct|granite|Dense|8.000|8.000|9|
|nvidia/Mistral-NeMo-Minitron-8B-Instruct|mistral|Dense|8.000|8.000|9|
|meta-llama/Llama-3.1-8B-Instruct|llama<br>|Dense|8.030|8.030|9|
|01-ai/Yi-1.5-9B-Chat|llama|Dense|9.000|9.000|9|
|upstage/SOLAR-10.7B-Instruct-v1.0|llama|Dense|10.700|10.700|8|
|**Large models**||||||
|google/gemma-3-12b-it|gemma3|Dense|12.000|12.000|4|
|meta-llama/Llama-2-13b-chat-hf<br>allenai/OLMo-2-1124-13B-Instruct|llama<br>olmo2|Dense<br>Dense|13.000<br>14.000|13.000<br>14.000|5<br>5|
|Qwen/Qwen3-14B|qwen3|Dense|14.000|14.000|4|
|microsoft/Phi-3-medium-4k-instruct|phi3|Dense|14.000|14.000|5|
|Qwen/Qwen1.5-MoE-A2.7B-Chat|qwen2_moe|Dense|14.300|14.300|4|
|microsoft/phi-4|phi3|Dense|14.700|14.700|4|
|internlm/internlm2_5-20b-chat|internlm2|Dense|20.000|20.000|3|
|**XLarge models**||||||
|EleutherAI/gpt-neox-20b|gpt_neox|Dense|20.000|20.000|2|
|upstage/solar-pro-preview-instruct|solar<br>|Dense<br>|22.100|22.100|2|
|mistralai/Mistral-Small-24B-Instruct-2501<br>mistralai/Mistral-Small-Instruct-2409|mistral<br>mistral|Dense<br>Dense|24.000<br>24.000|24.000<br>24.000|2<br>2|
|google/gemma-3-27b-it|gemma3|Dense|27.000|27.000|2|
|**MoE models**||||||
|microsoft/Phi-tiny-MoE-instruct|phimoe|MoE|3.800|1.100|10|
|allenai/OLMoE-1B-7B-0924|olmoe|MoE|7.000|1.000|9|
|deepseek-ai/deepseek-moe-16b-base|deepseek|MoE|16.000|2.800|4|
|unsloth/gpt-oss-20b-BF16|gptoss|MoE|21.000|3.600|2|
|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|_<br>nemotron_h|MoE|30.000|3.500|2|



Page 17 of 13 

: _Preprint submitted to Elsevier_ 

