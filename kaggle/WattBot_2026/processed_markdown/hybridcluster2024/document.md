# **Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads** 

Grant Wilkins Srinivasan Keshav Richard Mortier gfw27@cam.ac.uk sk818@cam.ac.uk rmm1002@cam.ac.uk University of Cambridge University of Cambridge University of Cambridge Cambridge, UK Cambridge, UK Cambridge, UK 

## **ABSTRACT** 

Both the training and use of Large Language Models (LLMs) require large amounts of energy. Their increasing popularity, therefore, raises critical concerns regarding the energy efficiency and sustainability of data centers that host them. This paper addresses the challenge of reducing energy consumption in data centers running LLMs. We propose a _hybrid_ data center model that uses a cost-based scheduling framework to dynamically allocate LLM tasks across hardware accelerators that differ in their energy efficiencies and computational capabilities. Specifically, our workload-aware strategy determines whether tasks are processed on energy-efficient processors or high-performance GPUs based on the number of input and output tokens in a query. Our analysis of a representative LLM dataset, finds that this hybrid strategy can reduce CPU+GPU energy consumption by 7.5% compared to a workload-unaware baseline. 

## **KEYWORDS** 

Large-Scale Data, LLMs, Energy Consumption, Green Computing 

### **ACM Reference Format:** 

Grant Wilkins, Srinivasan Keshav, and Richard Mortier. 2024. Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads. In _Proceedings of 12th International Workshop on The EverEvolving Data Centers (E2DC)._ ACM, New York, NY, USA, 8 pages. https: //doi.org/XXXXXXX.XXXXXXX 

## **1 INTRODUCTION** 

Large Language Models (LLMs) such as OpenAI’s GPT-4 [24] and Google’s PaLM [4] have become emblematic of the AI revolution, driving significant advancements not only in natural language understanding, generation, and translation but also in summarizing and contextualizing large volumes of textual data. Characterized by their extensive scale and depth, their deployment demands substantial computational resources and hence poses significant challenges in terms of energy consumption and operational efficiency [38]. The increasing application of LLMs across diverse sectors further 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _E2DC, June 3, 2024, Singapore, SG_ 

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM 

https://doi.org/XXXXXXX.XXXXXXX 

compounds these challenges, because datacenters, which are responsible for a considerable portion of global electricity consumption, must balance performance targets for LLM tasks running on heterogeneous hardware with the need for energy efficiency [7, 21]. Reducing the energy efficiency of LLMs thus emerges as both a technical challenge and an environmental imperative [22]. 

Traditional data center designs often struggle to best exploit the capabilities of heterogeneous hardware-based LLMs, particularly when trying to minimize energy consumption without sacrificing output quality and latency [6]. However, this challenge also presents an opportunity to innovate in datacenter architecture and management. We show that by rethinking how GPU resources are allocated and managed, there is potential to significantly reduce the energy footprint of LLM deployments while maintaining or even enhancing computational performance. 

We find that a dynamic task-scheduling model that assigns LLM tasks to GPUs based on the resulting energy efficiency can reduce overall energy. Moreover, implementing a workload-aware system for input and output token processing can further reduce energy usage. Thus, a hybrid datacenter task allocation model, which allocates different tasks to different hardware accelerators based on their system demands, can reduce the overall energy consumption of LLM inference compared to a workload-unaware baseline. 

Our contributions are as follows: 

- (1) We analyze the energy consumption and runtime of several 7B-parameter LLMs’ across various hardware configurations. 

- (2) We propose and evaluate a workload-aware scheduler for LLMs that optimizes energy efficiency based on the size of input and output token loads, demonstrating a 7.5% decrease in energy consumption over non-workload-aware baselines. 

- (3) We release a comprehensive dataset and benchmark suite for evaluating the energy efficiency of LLM inference, enabling researchers and practitioners to assess the impact of their design choices. 

Through these contributions, we hope to support more sustainable and cost-effective AI inference deployments. 

The remainder of this paper is as follows: Section 2 provides background information on LLM inference and energy consumption in AI systems. Section 3 formulates the problem and introduces our cost function. Section 4 details the methods used for benchmarking LLM inference on diverse systems. Section 5 presents the performance results of LLM inference across multiple hardware configurations. Section 6 proposes and evaluates our energy-optimal hybrid data center design. Finally, Section 7 discusses related works, and Section 8 summarizes the conclusions of the paper. 



Grant Wilkins, Srinivasan Keshav, and Richard Mortier 

E2DC, June 3, 2024, Singapore, SG 

## **2 BACKGROUND** 

## **2.1 Inference Using Large Language Models** 

Transformer-based neural network architectures have led to impressive gains in the performance of LLMs for language understanding and generation [5]. LLMs such as OpenAI’s GPT-4 [24] and Google’s Gemini [32] have demonstrated human-level proficiency on many language benchmarks while requiring billions of parameters and massive datasets for training. The inference phase of LLMs involves utilizing a trained model to make predictions based on new, unseen data. Unlike the training phase, which is typically a one-time, compute-intensive process that occurs offline, inference is an ongoing, real-time process that directly impacts end-user experiences [7]. This phase is critical as it represents the point at which AI capabilities become accessible to users. 

Inference in LLMs can be computationally expensive due to several factors: **(1) Model Size:** The sheer size of these models, often billions of parameters, necessitates significant computational power to process each query [38]. **(2) Latency Expectations:** Many applications based on LLMs, such as digital assistants, automated writing aids, and real-time translators, require low-latency responses [35]. **(3) Scalability:** The ability to scale inference operations to accommodate varying user demands without degradation in response times is crucial. 

## **2.2 Energy Consumption in AI Systems** 

Recent reports have found that the computational requirements for state-of-the-art AI entail massive energy consumption and carbon emissions [7, 21, 26, 29, 38]. The energy intensity of AI systems can be broadly divided into the energy required for training versus inference after models are deployed [13]. Training complex models on massive datasets is an energy-intensive process, with estimates finding that training GPT-3 required 1,287 megawatt-hours of energy [26]. LLMs can also have huge emissions depending on deployment scale and hardware efficiency [29]. For example, over a year of use, inference by LLMs on cloud infrastructure can consume over 25× more energy than training a model [7]. Optimizing software and hardware specifically for AI workloads is thus essential [3]. 

## **2.3 Heterogeneous Systems for Efficient Computing** 

Modern systems demonstrate a complex interplay between scale, architecture, workload behavior and efficiency objectives. The architecture of compute nodes can significantly impact the energy efficiency and processing capabilities of large-scale computing systems [18]. Conventional server architectures based on multicore CPUs face energy proportionality and scalability limitations for modern data-intensive workloads [20]. Several researchers have explored heterogeneous server configurations to improve energy efficiency [12, 15, 16, 19]. Distributed solutions can translate to lower energy efficiency, as communication overheads dominate [9]. Still, specialized clusters like NVIDIA’s DGX show 4x better performance per watt over conventional servers [30]. 

## **3 PROBLEM FORMULATION** 

To model the operational demands of a hybrid, heterogeneous datacenter hosting LLMs, we define a cost function to reflect the workload distribution across different systems. We define a cost function _𝑈_ ( _𝑚,𝑛,𝑠_ ) that accounts for both energy consumption and runtime: 



$$
𝑈(𝑚,𝑛,𝑠) = 𝜆𝐸(𝑚,𝑛,𝑠) + (1 -𝜆)𝑅(𝑚,𝑛,𝑠), (1)
$$

where _𝑚_ and _𝑛_ denote the number of input and output tokens, respectively. _𝜆_ ∈[0 _,_ 1] is a tunable parameter that balances the weight of energy efficiency versus speed. _𝐸_ ( _𝑚,𝑛,𝑠_ ) is the energy consumed by system _𝑠_ to process _𝑚_ input tokens and generate _𝑛_ output tokens, measured in joules. _𝑅_ ( _𝑚,𝑛,𝑠_ ) is the time required to process these tokens on system _𝑠_ , measured in seconds. 

Our objective is to minimize the total cost across all tasks and systems: 



$$
min {𝑄𝑠}𝑠\in{}{}𝑆 \sum{}{}︁ 𝑠\in{}{}𝑆 \sum{}{}︁ (𝑚,𝑛)\in{}{}𝑄𝑠 𝑈(𝑚,𝑛,𝑠) (2)
$$



$$
s.t. � 𝑠\in{}{}𝑆 𝑄𝑠= 𝑄 (3)
$$



$$
∀𝑠: 𝑄𝑠\cap{}{}𝑄𝑠′ = \emptyset{}{}for 𝑠\neq{}{}𝑠′ (4)
$$

where _𝑆_ is the set of all systems, _𝑄_ is the total set of queries, _𝑄𝑠_ is the subset of queries assigned to system _𝑠_ . 

This model ensures that each query is processed exactly once, optimizing for energy efficiency or quick response times, depending on the operational needs, as parameterized by _𝜆_ . We note, however, that certain systems may be better suited to specific tasks, based on the workload characteristics, such as the need for rapid response times. Adjustments in _𝜆_ allow the datacenter to shift its focus between minimizing energy consumption and reducing runtime as operational priorities change. 

## **4 METHODS** 

Here, we describe the methods and tools we use to benchmark LLM inference. In all cases, we use Huggingface’s Accelerate [11] to standardize hardware optimization for inference across all platforms. T his library takes advantage of the available accelerator resources and shards models accordingly to minimize intermediate communication and maximize the distributed capabilities for computation across the devices. 

## **4.1 Model Selection** 

Our study employs three open-source LLMs for their capabilities and ability to run on diverse hardware efficiently: (1) Falcon [2], (2) Llama-2 [33], and (3) Mistral (7B parameters) [17]. These models were selected to represent a spectrum of architectures and training corpora. We subject each model to a series of standardized NLP tasks to evaluate their energy consumption during inference. 

_4.1.1 Falcon._ The Falcon (7B) [2] model utilizes multi-query attention, significantly reducing memory requirements and increasing processing speed. The model’s training on the bilingual RefinedWeb dataset enhances its applicability across diverse linguistic contexts. 

_4.1.2 Llama-2._ We select Llama-2 (7B) for its optimization in dialogue tasks and its improvements in safety and helpfulness. The 



Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads 

E2DC, June 3, 2024, Singapore, SG 

model’s unique pretraining methodologies and advanced architectural features, such as grouped-query attention, make it an ideal candidate for analyzing energy efficiency in complex language tasks. 

_4.1.3 Mistral._ We include Mistral (7B) [17] for its grouped-query attention and sliding window attention mechanisms, contributing to fast and efficient inference. Its superior performance in various benchmarks, especially in reasoning, mathematics, and code generation, makes it an essential model for our analysis. 

## **4.2 Energy Profiling of Diverse Systems** 

Depending on the platform, we profile each system’s energy consumption during inference using customized setups that capture runtime and energy or power metrics. Here, we describe how we monitor the energy usage of NVIDIA GPUs, Apple Silicon CPU/GPU, Intel CPUs, and AMD CPUs. 

_4.2.1 NVIDIA GPUs._ We use PyJoules [27], a Python-based energy measurement library, to quantify the energy consumption associated with inference on NVIDIA GPUs. PyJoules provides an interface to NVML [23], providing a software-defined energy usage assessment for targeted NVIDIA devices. This tool offers real-time energy consumption of GPUs for a given tracked process, which is a critical component of our analysis given the GPU-heavy computation involved in LLM inference. 

_4.2.2 Apple Silicon CPU/GPU._ No standard energy measurement tools are available for profiling energy and power usage for Apple Silicon through an API like PyJoules or RAPL. Therefore, we employ a daemon-based approach to poll macOS’ powermetrics utility, providing a detailed view of the energy usage during model inference. To capture the energy consumption of the M1 GPU, we execute the powermetrics command through a Python subprocess. This command returns the percentage of the CPU power each CPU top process uses and the total CPU and GPU power consumption in 200ms intervals. This interval was chosen after testing to find the finest granularity measurement without incurring a significant CPU overhead for the I/O of buffering the large powermetrics output into memory. 

The energy monitoring is conducted concurrently with the LLM inference. A separate thread is dedicated to running the powermetrics command, ensuring real-time data collection. Post-inference, the collected data is processed to extract the recorded power data and then find the energy consumption through integration over the runtime. The GPU energy consumption, _𝐸𝑇𝑜𝑡𝑎𝑙,𝐺𝑃𝑈 ,_ is straightforward to calculate for each recorded power value, _𝑃𝐺𝑃𝑈,𝑖,_ at each timestep Δ _𝑡𝑖 ._ 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝐺𝑃𝑈= \sum{}{}︁ 𝑖 𝑃𝐺𝑃𝑈,𝑖Δ𝑡𝑖. (5)
$$

The CPU power draw data is less clear, as many processes run on the CPU. However, an "energy impact factor" through powermetrics allows us to infer how much power our Python inference process uses. Therefore, we calculate the CPU energy, _𝐸𝑇𝑜𝑡𝑎𝑙,𝐶𝑃𝑈 ,_ by multiplying _𝑃𝐶𝑃𝑈,𝑖_ by the "energy impact factor," which we denote as _𝛼𝑖,_ at each timestep: 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝐶𝑃𝑈= \sum{}{}︁ 𝑖 (𝛼𝑖𝑃𝐶𝑃𝑈,𝑖)Δ𝑡𝑖. (6)
$$

_4.2.3 Intel CPUs._ For Intel CPUs, we leverage PyJoules, a Pythonbased energy measurement library similar to our approach for NVIDIA GPUs. This tool supports RAPL (Running Average Power Limit) interfaces, enabling us to obtain fine-grained energy consumption data [36]. We focus on two primary RAPL domains: Package 0 and Package 1, which correspond to the entire CPU package’s energy consumption, including all cores in the package. 

PyJoules allows us to capture the energy usage of these domains in real time, enabling us to profile the energy consumption specifically during model inference tasks. To account for base energy consumption unrelated to our inference process, we conduct a preanalysis phase to measure the CPU’s average idle power draw. This idle measurement is then subtracted from the total energy consumption during inference to accurately determine the net energy expenditure attributable to the inference process. 

We instrument our code to query the RAPL readings at the start and end of the inference task, calculating the energy consumption as follows: 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝐶𝑃𝑈= \sum{}{}︁ 𝑖 �� 𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒-0,𝑖-𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒-0,𝐼𝑑𝑙𝑒 � + � 𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒-1,𝑖-𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒-1,𝐼𝑑𝑙𝑒 �� Δ𝑡𝑖, (7)
$$

where _𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒_ −0 _,𝑖_ and _𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒_ −1 _,𝑖,_ represent the power draw from Package 0 and Package 1, respectively, and _𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒_ −0 _,𝐼𝑑𝑙𝑒_ and _𝑃𝑃𝑎𝑐𝑘𝑎𝑔𝑒_ −1 _,𝐼𝑑𝑙𝑒_ represent the average idle power draw of the CPU packages, respectively. 

_4.2.4 AMD CPUs._ We adopt a different strategy for AMD CPUs due to the absence of a Python API. Instead, we utilize AMD _𝜇_ Prof’s timechart feature, which provides detailed power draw metrics for every core on the chip at fine-grained intervals. By polling AMD _𝜇_ Prof at 100ms intervals, we can capture the power draw of each physical core throughout the model inference process. 

To ensure we accurately attribute the energy consumption to our inference task, we monitor the CPU core residency through psutil. This information allows us to identify and record the specific cores actively engaged in the inference process at each time step. The total energy consumption for the inference task is then calculated by summing the power usage across all active cores and summing over the product of the power usage and time of inference, as follows: 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝐶𝑃𝑈= \sum{}{}︁ 𝑐𝑜𝑟𝑒 �\sum{}{}︁ 𝑖 𝑃𝑐𝑜𝑟𝑒,𝑖Δ𝑡𝑖 � (8)
$$

where _𝑃𝑐𝑜𝑟𝑒,𝑖_ represents the power draw of an individual core at each time step, _𝑖_ . 

## **5 LLM INFERENCE PERFORMANCE ON DIVERSE CLUSTERS** 

## **5.1 Hardware and Software Versions** 

The systems we profile are shown in Table 1. We consider these systems as they demonstrate three prominent CPU manufactures and different generations of GPUs. We utilize PyTorch v2.0.1, Torchvision v0.15.2, Numpy v1.26.0, Huggingface v0.20.2, and Accelerate v0.26.1. 



Grant Wilkins, Srinivasan Keshav, and Richard Mortier 

E2DC, June 3, 2024, Singapore, SG 

|System Name|CPU|GPU(s) per Node|DRAM per Node|VRAM per GPU|
|---|---|---|---|---|
|Macbook Pro|10-core M1 Pro|14-core M1 Pro|32GB|-|
|Swing AMD+A100|2×64-core AMD EPYC 7742|8×NVIDIA A100|1TB|40GB|
|Palmetto Intel+V100|40-Core Intel Xeon 6148G|2×NVIDIA V100|376GB|16GB|



**Table 1: Our System Configurations** 

We note that the M1-Pro results only include the Llama-2 (7B) and Mistral (7B) results, as Falcon (7B) generally did not complete tasks in less than two orders of magnitude greater runtime. 

## **5.2 Experimental Strategy** 

To comprehensively evaluate the performance of different system configurations across various models, we conducted a series of controlled experiments. We systematically varied the number of input and output tokens to measure their effects on runtime and energy consumption under two main experimental conditions. In each experiment we do not allow for key-value caches to be re-used to ensure our testing environment is standardized. 

_5.2.1 Vary Input Tokens._ For the first experimental condition, we executed inference requests with increasing input token sizes, ranging from 8 to 2048 tokens, while maintaining a fixed output token size of 32. This setup allowed us to isolate the impact of input size on the system’s performance and energy efficiency. 

_5.2.2 Vary Output Tokens._ In the second set of experiments, we varied the output token limit from 8 to 4096 tokens, keeping the input token size constant at 32. This approach helped us understand how increasing output demands affect the runtime and energy consumption of the systems tested. 

_5.2.3 Randomization and Stopping Criteria._ Each experiment was conducted in a randomized order to mitigate any potential bias introduced by the sequence of tests. To ensure the reliability of our results, we adhered to strict criteria for statistical confidence. Each configuration was tested repeatedly until either of two conditions was met: (1) The measured runtime had to be within 0.5 seconds of the actual mean runtime with 95% confidence. (2) A maximum of 25 trials were conducted for each setting if the first condition could not be met. 

## **5.3 Input Token Analysis** 

Here, we present the impacts on runtime, energy consumption per token, and throughput for LLMs across different hardware configurations while varying the number of input tokens. We perform these experiments using the suite of systems outlined in Table 1 with the models outlined in Section 4.1. In our experiments on the Palmetto Intel+V100 system, the V100 GPU had an out-of-memory error beyond 1024 output tokens for Falcon (7B). 

Our runtime measurements show a significant increase as input tokens grow. As depicted in Figure 1(a), all systems exhibit a nonlinear escalation in runtime with increasing token counts, with the M1-Pro system showing the most significant magnitude. This trend highlights the computational burden imposed by larger input 

sizes, particularly on smaller systems that are not as well designed to handle extensive workloads. 

For all systems, we notice that throughput follows a "roofline model" with increasing input tokens [37]. Figure 1(b) illustrates these dynamics, indicating an increase in throughput for all systems until a certain point where inference becomes bound by compute and not by the overhead of the software, as described by roofline performance models [37]. 

Energy efficiency varies markedly across different systems. The M1-Pro demonstrates consistently low energy consumption per token, particularly for smaller input sizes, as shown in Figure 1(c). This efficiency reflects the M1-Pro’s design optimization for low-power operations. In contrast, the Swing AMD+A100, while capable of handling more significant token inputs more efficiently, consumed more energy per token for small workloads yet became more energy efficient at larger input token sizes, underscoring a trade-off between workload size and energy efficiency. 

## **5.4 Output Token Analysis** 

Here we examine the performance trends associated with increasing the number of output tokens for our LLMs and systems of interest, specifically focusing on runtime, energy consumption per token, and throughput. In our experiments, the M1-Pro also could not generate more than 512 output tokens without significant runtime penalties. For the Palmetto Intel+V100 system, the V100 GPU had an OOM error beyond 1024 output tokens for Falcon (7B) and for all models beyond 2048 tokens. 

Runtime significantly increases with the number of output tokens across all systems. As illustrated in Figure 2(a), the escalation in runtime is pronounced, particularly as the output token count reaches higher magnitudes. This increase is indicative of the substantial computational effort required by LLMs to generate successive tokens. 

In Figure 2(b), we observe a decrease in throughput across all systems as the number of output tokens increases. This trend highlights the inherent computational complexity involved in generating larger sequences of tokens in LLM tasks. As the output token count grows, the system must process each additional token, recalculating the context and updating internal model states [34]. This not only increases the total computation per query but also leads to a greater accumulation of processing time per token, which consequently lowers the overall throughput. 

Energy consumption per token also shows an increasing trend as the number of output tokens grows. Displayed in Figure 2(c), this trend underscores the energy-intensive nature of producing larger outputs. Systems such as the M1-Pro, while generally more energy-efficient, begin to consume more energy per token as output 



Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads 

E2DC, June 3, 2024, Singapore, SG 

![Figure](assets/figure_0001_page_0005.svg)**Figure 1: Performance of Various Systems and Models for Processing Variable Input Tokens–Due to the low variance in the data, error bars are too small to be visible.** 

demands increase, reflecting the intensive processing involved in output generation. 

## **5.5 Comparing the Input and Output Analyses** 

When comparing Figure 1(a) and Figure 2(a), we observe that increases in the number of output tokens result in a more considerable increase in runtime than increases in input tokens. The computational complexity of processing input tokens primarily involves encoding the input context, which occurs once per input sequence and follows a more linear computational trajectory. In contrast, generating output tokens is inherently more complex and iterative. Each new output token requires the model to run through all its layers to predict the next token based on an ever-expanding context, which includes both the initial input and all previously generated tokens [34]. This ongoing computation involves recalculating attention across an increasing number of tokens, updating hidden states, and generating a probability distribution over the vocabulary for each new token. Consequently, as the number of output tokens grows, the computational load increases significantly, leading to more significant runtime increases than processing input tokens. 

The impacts on runtime also translate to the throughput, depicted in Figure 1(b) and Figure 2(b). There is a noticeable decline in throughput as output tokens increase, more so than input tokens. The decrease in throughput for output tokens is primarily due to the heightened computational requirements for generating subsequent tokens, where each token’s generation slows down as the sequence lengthens. Furthermore, the energy per token also increases as output tokens grow, as shown in our analysis. The energy required to generate each output token becomes significant due to longer passes through the transformer network. We contrast this with the energy consumption when processing input tokens, which, despite increasing, does so at a less steep rate. 

## **6 ENERGY-OPTIMAL HYBRID DATACENTER FOR LLM INFERENCE** 

Considering the performance results we collect from LLM inference across multiple systems, we notice that there is an energy-optimal way to construct a hybrid datacenter with a combination of M1 Pro’s and A100s. The intuition behind this is that the energy expended per token for the M1 Pro is lower than that of the A100 up to a 

certain point in the number of input and output tokens as seen in Figures 1(c) and 2(c). However, the energy efficiency characteristics are different when varying the number of input and output tokens, and therefore, we will proceed with separate analyses. 

## **6.1 Number of Input Tokens Analysis** 

Suppose we have a hybrid data center with M1-Pros and A100s. Then, we have some workload for an LLM, a set of queries with some outputs. In such a configuration, we implement a scheduling heuristic based on a cutoff threshold, _𝑇𝑖𝑛,_ for input token length. This heuristic dictates that queries with _𝑛_ ≤ _𝑇𝑖𝑛_ tokens are processed on M1 Pro systems, which we have shown have good energy efficiency with handling smaller computational loads. Conversely, queries with _𝑛 > 𝑇𝑖𝑛_ tokens leverage the greater computational ability of A100 GPUs, which offer greater energy-per-token advantages for larger tasks despite their higher power usage. We point out that this is the same method mentioned in the problem formulation in Eqn. 2, where our queries _𝑄_ are partitioned into _𝑄𝑀_ 1 and _𝑄𝐴_ 100 strictly on input and output size. 

To find an optimal threshold _𝑇𝑖𝑛_ empirically, we analyze the token distribution in prompts from the Alpaca [31] dataset, a benchmark dataset frequently used in model fine-tuning. This dataset comprises 52K prompts, offering a diverse range of lengths akin to a typical workload in systems like GPT-4 [24]. The distribution of input tokens, visualized in our analysis (see Fig. 3(a)), serves as a proxy for understanding the variegated nature of LLM workloads. 

The energy component of our cost function, split over the token threshold, is as follows: 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝑖𝑛= 𝑇𝑖𝑛 \sum{}{}︁ 𝑚=1 𝑚𝑓𝑖𝑛(𝑚)𝐸𝑀1,𝑖𝑛(𝑚) + 𝑀 \sum{}{}︁ 𝑚=𝑇𝑖𝑛+1 𝑚𝑓𝑖𝑛(𝑚)𝐸𝐴100,𝑖𝑛(𝑚), (9)
$$

where _𝐸𝑇𝑜𝑡𝑎𝑙,𝑖𝑛_ represents the total energy consumption for a given dataset of input lengths _𝑚_ with corresponding frequencies _𝑓𝑖𝑛_ ( _𝑚_ ) _,_ and _𝐸𝑀_ 1 _,𝑖𝑛_ ( _𝑚_ ) and _𝐸𝐴_ 100 _,𝑖𝑛_ ( _𝑚_ ) denote the mean energy per token for varying the input token size for the M1-Pro and A100 systems, respectively. Utilizing this model with our dataset enables the approximation of total energy consumption for various threshold settings, offering insights into the energy dynamics of hybrid datacenter operation. In Figure 4, we show the energy and runtime 



Grant Wilkins, Srinivasan Keshav, and Richard Mortier 

E2DC, June 3, 2024, Singapore, SG 

![Figure](assets/figure_0002_page_0006.svg)**Figure 2: Performance of Various Systems and Models for Processing Variable Output Tokens–Missing data points in M1-Pro and Palmetto Intel+V100 are due to CUDA out of memory errors. Due to the low variance in the data, error bars are too small to be visible.** 

![Figure](assets/figure_0003_page_0006.svg)**Figure 3: Distribution of Token Counts for Alpaca [31]** 

simulation results of performing inference for the input token sizes from the Alpaca dataset. 

Our findings indicate that a threshold of 32 tokens strikes an optimal balance, significantly reducing energy consumption by relegating the inference of shorter queries to the more energyefficient M1 Pro systems. This policy not only capitalizes on the inherent energy efficiency of the M1 Pro for smaller tasks but also reserves the computational might of the A100 for queries that necessitate its robust capabilities. However, it’s important to note that this energy optimization comes at the cost of increased runtime. 

## **6.2 Number of Output Tokens Analysis** 

We want to use the same scheduling heuristic and performance model to determine a threshold _𝑇𝑜𝑢𝑡_ for the number of output tokens. Except this time, we have different frequencies _𝑓𝑜𝑢𝑡_ ( _𝑛_ ) for the _𝑛_ output tokens and different mean energy per token for varying the output token size, _𝐸𝑀_ 1 _,𝑜𝑢𝑡_ ( _𝑛_ ) and _𝐸𝐴_ 100 _,𝑜𝑢𝑡_ ( _𝑛_ ) _._ We also utilize the distribution of the number of output tokens in the Alpaca dataset (see Fig. 3(b)). We revise our performance model as follows: 



$$
𝐸𝑇𝑜𝑡𝑎𝑙,𝑜𝑢𝑡= 𝑇𝑜𝑢𝑡 \sum{}{}︁ 𝑛=1 𝑛𝑓𝑜𝑢𝑡(𝑛)𝐸𝑀1,𝑜𝑢𝑡(𝑛) + 𝑁 \sum{}{}︁ 𝑛=𝑇𝑜𝑢𝑡+1 𝑛𝑓𝑜𝑢𝑡(𝑛)𝐸𝐴100,𝑜𝑢𝑡(𝑛). (10)
$$

![Figure](assets/figure_0004_page_0006.svg)**Figure 4: Performance of Hybrid Datacenter for Input Tokens Processing Alpaca–Dashed line shows the value for using only one kind of hardware for inference** 

As the M1 Pro could only generate up to 512 tokens of a response, we only test _𝑇𝑜𝑢𝑡_ up until this point. In Figure 5, we show the energy and runtime simulation results of performing inference for the input token sizes from the Alpaca dataset. 

Fig. 5(b) and Fig. 2(c) assess the energy consumption and runtime implications of various threshold settings for output generation. Our findings suggest that although higher thresholds may leverage the M1 Pro’s energy efficiency for smaller outputs, there is an optimal point at 32 output tokens that minimizes energy consumption. 



Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads 

E2DC, June 3, 2024, Singapore, SG 

![Figure](assets/figure_0005_page_0007.svg)**Figure 5: Performance of Hybrid Datacenter for Output Tokens Processing Alpaca – Dashed line shows the value for using only one kind of hardware for inference** 

## **6.3 Balancing Energy Efficiency and Runtime Performance** 

Our analysis of both input and output token processing within a hybrid, heterogeneous datacenter framework has led to the identification that with certain thresholds at _𝑇𝑖𝑛𝑝𝑢𝑡_ = 32 and _𝑇𝑜𝑢𝑡𝑝𝑢𝑡_ = 32, we can strategically allocate tasks to M1 Pro systems or A100 GPUs based on token count, optimizing for energy efficiency. 

Shifting the token distribution leverages the M1 Pro’s superior energy efficiency for input and output tasks up to the threshold, beyond which we utilize the A100’s computational power. This policy saves energy as smaller-token tasks are handled by the more efficient M1 Pro for outputs up to the threshold. However, this energy optimization comes at the expense of increased runtime, which is particularly noticeable in output token generation where the M1 Pro, despite its efficiency, does not match the A100’s speed. 

The energy-runtime trade-off presents a favorable scenario for applications that have low runtime sensitivity. For instance, batch processing of LLM tasks, such as overnight data analyses or nontime-critical computations, can benefit significantly from this energyefficient configuration. Similarly, free or not directly monetized services, where the cost of computation impacts operational sustainability, stand to gain from minimizing energy expenditures even at the cost of longer processing times. 

This approach also opens discussions on Quality of Service (QoS) for LLMs, an area that still needs to be explored [1, 35]. Traditional QoS metrics often prioritize speed and reliability, but energy efficiency may also become a critical QoS dimension for LLM applications, particularly in energy-constrained or cost-sensitive scenarios. 

## **7 RELATED WORK** 

## **7.1 Hybrid and Energy Efficient Heterogeneous Data Centers** 

Recent studies in optimizing data center architectures for deep learning have highlighted the necessity of energy-efficient scheduling and task allocation across diverse hardware. Gu et al. [10] explore GPU clusters’ energy-efficient scheduling, revealing substantial improvements in power utilization without considering diverse GPU types for different task requirements. This work highlights a gap in understanding how various GPU configurations could enhance energy efficiency further. Similarly, Patel et al. [25] demonstrate the benefits of hybrid computing environments, emphasizing FPGA over GPU diversity. This focus leaves room to explore the specific impacts of different GPU classes in such settings. 

In the realm of LLMs, Zhao et al. [39] introduce strategies like phase-aware partitioning and adaptive quantization in heterogeneous clusters but do not integrate energy considerations into their analysis, which is crucial for understanding the real-world applicability of these models in power-sensitive environments. On the other hand, Radovanović et al. [28] and Chien et al. [7] discuss broader aspects of carbon-aware computing and reducing the carbon impact of AI inference, respectively. These works emphasize the importance of node/device-level energy metrics, often overlooked in typical LLM deployment strategies, thus underscoring the need for detailed energy consumption profiling across different models and hardware types. 

## **7.2 LLM Inference as a Service** 

Further focusing on energy consumption, Hu et al. [14] analyze deep learning workloads in GPU datacenters, offering insights into energy conservation strategies through workload scheduling. This research aligns with our objectives by confirming the critical role of scheduling in reducing energy footprints. Anderson et al. [3] propose carbon-aware datacenter software that could complement physical hardware adjustments by making energy and carbon metrics visible to application developers, encouraging more energyefficient coding practices. 

Addressing service quality, Wang et al. [35] study the efficiency and reliability of LLM serving, highlighting the challenges of maintaining high-quality service while managing computational loads effectively. This perspective is pertinent as it underscores the tradeoff between performance and energy efficiency, which is central to our study. Lastly, Desislavov et al. [8] provide a timely examination of trends in AI inference energy consumption, arguing that while performance has increased dramatically, energy consumption has not escalated at the same pace, thanks to hardware optimizations and algorithmic innovations. This outlook is necessary as it suggests the potential for further optimizations in LLM inference tasks, which are typically energy-intensive. 

## **8 CONCLUSIONS** 

By carefully analyzing the energy and runtime of heterogeneous compute hardware to host LLMs, we show that a hybrid, heterogeneous datacenter and a cost-based scheduling framework can allocate LLM tasks to accelerators that are best suited to run them, 



Grant Wilkins, Srinivasan Keshav, and Richard Mortier 

E2DC, June 3, 2024, Singapore, SG 

in terms of energy efficiency and computational performance. This decision is based simply on the size of input and output tokens, making the decision process easy to integrate into existing workloads. Our cost function allows us to balance energy consumption against runtime , providing a method to quantitatively assess and manage trade-offs in real-time. By demonstrating that data-driven approaches can mitigate the energy impact of serving LLM inference, we hope to pave the way for more energy-efficient and environmentally friendly technology infrastructures. 

   - _Data Management on New Hardware_ (New York, NY, USA, 2010), DaMoN ’10, Association for Computing Machinery, p. 47–55. 

- [19] Liu, W., Yan, Y., Sun, Y., Mao, H., Cheng, M., Wang, P., and Ding, Z. Online job scheduling scheme for low-carbon data center operation: An information and energy nexus perspective. _Applied Energy 338_ (2023), 120918. 

- [20] Lo, D., Cheng, L., Govindaraju, R., Ranganathan, P., and Kozyrakis, C. Heracles: Improving resource efficiency at scale. In _2015 ACM/IEEE 42nd Annual International Symposium on Computer Architecture (ISCA)_ (2015), pp. 450–462. 

- [21] Luccioni, A. S., Viguier, S., and Ligozat, A.-L. Estimating the carbon footprint of bloom, a 176b parameter language model, 2022. 

- [22] Mytton, D., and Ashtine, M. Sources of data center energy estimates: A comprehensive review. _Joule 6_ , 9 (2022), 2032–2056. 

- [23] NVIDIA. Nvidia-nvml. https://docs.nvidia.com/deploy/nvml-api/index.html, Accessed 2024. Available online. 

## **ACKNOWLEDGMENT** 

We gratefully acknowledge the computing resources provided on Swing and Palmetto, both high-performance computing clusters operated by the Laboratory Computing Resource Center at Argonne National Laboratory and Clemson University, respectively. During this work GW was supported by a Churchill Scholarship. 

## **REFERENCES** 

- [1] Agarwal, M., Qureshi, A., Nikhil Sardana, L. L., Quevedo, J., and Khudia, D. LLM Inference Performance Engineering: Best Practices, 2023. 

- [2] Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Alhammadi, M., Daniele, M., Heslow, D., Launay, J., Malartic, Q., Noune, B., Pannier, B., and Penedo, G. The falcon series of language models: Towards open frontier models. 

- [3] Anderson, T., Belay, A., Chowdhury, M., Cidon, A., and Zhang, I. Treehouse: A case for carbon-aware datacenter software. _SIGENERGY Energy Inform. Rev. 3_ , 3 (oct 2023), 64–70. 

- [4] Anil, R., Dai, A. M., Firat, O., and et al. Palm 2 technical report, 2023. 

- [5] Bommasani, R., Hudson, D. A., Adeli, E., and et al. On the opportunities and risks of foundation models, 2022. 

- [6] Chen, L., Ahmed, N. K., Dutta, A., Bhattacharjee, A., Yu, S., Mahmud, Q. I., Abebe, W., Phan, H., Sarkar, A., Butler, B., Hasabnis, N., Oren, G., Vo, V. A., Munoz, J. P., Willke, T. L., Mattson, T., and Jannesari, A. The landscape and challenges of hpc research and llms, 2024. 

- [7] Chien, A. A., Lin, L., Nguyen, H., Rao, V., Sharma, T., and Wijayawardana, R. Reducing the carbon impact of generative ai inference (today and in 2035). In _Proceedings of the 2nd Workshop on Sustainable Computer Systems_ (New York, NY, USA, 2023), HotCarbon ’23, Association for Computing Machinery. 

- [8] Desislavov, R., Martínez-Plumed, F., and Hernández-Orallo, J. Trends in ai inference energy consumption: Beyond the performance-vs-parameter laws of deep learning. _Sustainable Computing: Informatics and Systems 38_ (2023), 100857. 

- [9] Georgiou, Y., Glesser, D., and Trystram, D. Adaptive resource and job management for limited power consumption. In _Proceedings of the 2015 IEEE International Parallel and Distributed Processing Symposium Workshop_ (USA, 2015), IPDPSW ’15, IEEE Computer Society, p. 863–870. 

- [10] Gu, D., Xie, X., Huang, G., Jin, X., and Liu, X. Energy-efficient gpu clusters scheduling for deep learning, 2023. 

- [11] Gugger, S., Debut, L., Wolf, T., Schmid, P., Mueller, Z., Mangrulkar, S., Sun, M., and Bossan, B. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022. 

   - [24] OpenAI, :, Achiam, J., Adler, S., Agarwal, S., and et al. Gpt-4 technical report, 2023. 

   - [25] Patel, P., Lim, K., Jhunjhunwalla, K., Martinez, A., Demoulin, M., Nelson, J., Zhang, I., and Anderson, T. Hybrid computing for interactive datacenter applications, 2023. 

   - [26] Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., and Dean, J. Carbon emissions and large neural network training, 2021. 

   - [27] powerapi ng. Pyjoules: Python-based energy measurement library for various domains including nvidia gpus. https://github.com/powerapi-ng/pyJoules, 2024. Accessed: 2024-01-10. 

   - [28] Radovanović, A., Koningstein, R., Schneider, I., Chen, B., Duarte, A., Roy, B., Xiao, D., Haridasan, M., Hung, P., Care, N., et al. Carbon-aware computing for datacenters. _IEEE Transactions on Power Systems 38_ , 2 (2022), 1270–1280. 

   - [29] Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergeron, W., Kepner, J., Tiwari, D., and Gadepally, V. From words to watts: Benchmarking the energy costs of large language model inference, 2023. 

   - [30] Špeťko, M., Vysocký, O., Jansík, B., and Říha, L. Dgx-a100 face to face dgx-2— performance, power and thermal behavior evaluation. _Energies 14_ , 2 (2021). 

   - [31] Taori, R., Gulrajani, I., Zhang, T., and et al. Stanford alpaca: An instruction following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2024. Accessed: 2024-01-15. 

   - [32] Team, G. G. Gemini: A family of highly capable multimodal models, 2024. [33] Touvron, H., Martin, L., Stone, K., and et al. Llama 2: Open foundation and fine-tuned chat models, 2023. 

   - [34] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ (Red Hook, NY, USA, 2017), NIPS’17, Curran Associates Inc., p. 6000–6010. 

   - [35] Wang, Y., Chen, Y., Li, Z., Tang, Z., Guo, R., Wang, X., Wang, Q., Zhou, A. C., and Chu, X. Towards efficient and reliable llm serving: A real-world workload study, 2024. 

   - [36] Weaver, V. M., Johnson, M., Kasichayanula, K., Ralph, J., Luszczek, P., Terpstra, D., and Moore, S. Measuring energy and power with papi. In _2012 41st International Conference on Parallel Processing Workshops_ (2012), pp. 262–268. 

   - [37] Williams, S., Waterman, A., and Patterson, D. Roofline: an insightful visual performance model for multicore architectures. _Commun. ACM 52_ , 4 (apr 2009), 65–76. 

   - [38] Wu, C.-J., Raghavendra, R., Gupta, U., and et al. Sustainable ai: Environmental implications, challenges and opportunities. _Proceedings of Machine Learning and Systems 4_ (2022), 795–813. 

   - [39] Zhao, J., Wan, B., Peng, Y., Lin, H., and Wu, C. Llm-pq: Serving llm on heterogeneous clusters with phase-aware partition and adaptive quantization. _arXiv preprint arXiv:2403.01136_ (2024). 

- [12] He, Y., and Elnikety, S. Position paper: Embracing heterogeneity—improving energy efficiency for interactive services on heterogeneous data center hardware. In _Workshops at the Twenty-Fifth AAAI Conference on Artificial Intelligence_ (2011). 

- [13] Henderson, P., Hu, J., Romoff, J., Brunskill, E., Jurafsky, D., and Pineau, J. Towards the systematic reporting of the energy and carbon footprints of machine learning. _J. Mach. Learn. Res. 21_ , 1 (jan 2020). 

- [14] Hu, Q., Sun, P., Yan, S., Wen, Y., and Zhang, T. Characterization and prediction of deep learning workloads in large-scale gpu datacenters. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ (New York, NY, USA, 2021), SC ’21, Association for Computing Machinery. 

- [15] Hu, X., Li, P., and Sun, Y. Minimizing energy cost for green data center by exploring heterogeneous energy resource. _Journal of Modern Power Systems and Clean Energy 9_ , 1 (2021), 148–159. 

- [16] Hussain, M., Wei, L.-F., Lakhan, A., Wali, S., Ali, S., and Hussain, A. Energy and performance-efficient task scheduling in heterogeneous virtualized cloud computing. _Sustainable Computing: Informatics and Systems 30_ (2021), 100517. 

- [17] Jiang, A. Q., Sablayrolles, A., Mensch, A., and et al. Mistral 7b, 2023. 

- [18] Lang, W., Patel, J. M., and Shankar, S. Wimpy node clusters: what about non-wimpy workloads? In _Proceedings of the Sixth International Workshop on_ 

