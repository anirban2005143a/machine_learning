# **The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization** 

**Jae-Won Chung Jiachen Liu Jeff J. Ma Ruofan Wu Oh Jun Kweon Yuxuan Xia Zhiyu Wu Mosharaf Chowdhury** University of Michigan 

## **Abstract** 

As the adoption of Generative AI in real-world services grow explosively, _energy_ has emerged as a critical bottleneck resource. However, energy remains a metric that is often overlooked, under-explored, or poorly understood in the context of building ML systems. We present the ML.ENERGY Benchmark, a benchmark suite and tool for measuring inference energy consumption under realistic service environments, and the corresponding ML.ENERGY Leaderboard, which have served as a valuable resource for those hoping to understand and optimize the energy consumption of their generative AI services. In this paper, we explain four key design principles for benchmarking ML energy we have acquired over time, and then describe how they are implemented in the ML.ENERGY Benchmark. We then highlight results from the latest iteration of the benchmark, including energy measurements of 40 widely used model architectures across 6 different tasks, case studies of how ML design choices impact energy consumption, and how automated optimization recommendations can lead to significant (sometimes more than 40%) energy savings without changing what is being computed by the model. The ML.ENERGY Benchmark is open-source and can be easily extended to various customized models and application scenarios. 

## **1 Introduction** 

Generative AI models have rapidly transitioned from research prototypes to real-world services such as ChatGPT [48], Character AI [4], Sora [49], and Midjourney [45]. However, exponential growth rarely continues without facing scaling bottlenecks; currently for generative AI, one of the most crucial bottlenecks is the _energy bottleneck_ [3,13,14,43,44,46]. That is, even with fleets of latest GPUs and exploding demand for ML compute, getting access to the energy necessary to power these systems is becoming increasingly costly, slow, and sometimes impossible. This particularly impacts serving real-world services as ML inference reportedly accounts for 80–90% of the total compute demand [10,27,50,51]. Left unaddressed, the energy bottleneck will not only hinder AI research and development progress [26], but also lead to energy being either squeezed out of existing electricity grids and impacting availability and price [3], or generated from power sources like fossil fuels [29]. 

However, despite its growing importance, energy remains a secondary consideration compared to traditional optimization objectives like time and accuracy [57]. How much energy does a model consume during inference? What is the right way for energy measurement and accounting during execution, let alone optimization? To bridge this gap, we launched the ML.ENERGY Leaderboard,<sup>1</sup> the first inference energy leaderboard for modern generative AI models to the best of our knowledge. The Leaderboard has been gradually expanding in multiple dimensions to now include (1) 40 different generative AI model architectures across a wide range of tasks – including LLM [64] chat and 

> 1 `https://ml.energy/leaderboard` 

Preprint. Under review. 



![Figure](assets/figure_0001_page_0002.svg)Figure 1: Overview of the benchmarking and optimization flow of the ML.ENERGY Benchmark. 

coding, VLM [38] visual chat, and text-to-image, text-to-video, and image-to-video generation using Diffusion models [12,20] – and (2) more up-to-date hardware and software stacks following rapid advancements in each area. 

In this paper, we share the design principles we have established over time (Section 2) and present the ML.ENERGY Benchmark that embodies these principles (Section 3). It provides two key functionalities as a tool: 

- **Extensible Benchmark** : It provides an easily extensible benchmark suite and a comprehensive set of tools for measuring the inference energy consumption of generative AI models for various tasks under _realistic_ deployment environments. 

- **Automated Optimization** : Based on energy measurement results, it provides automated energy optimization recommendations for generative AI model deployment. 

Finally, we highlight notable results from the most recent iteration of the ML.ENERGY Leaderboard, shedding light on (1) how much energy consumption varies across different generative AI models and tasks, (2) the complex trade-offs that involve energy, time, and model architecture design, and (3) the energy savings opportunity unlocked by automated optimization (Section 4). 

The ML.ENERGY Benchmark is open-source on GitHub,<sup>2</sup> and the ML.ENERGY Leaderboard allows everyone to browse full results from the ML.ENERGY Benchmark. The benchmark is designed to be easily extensible, allowing users to benchmark their models and compare them to others without the heavy lifting of building their own benchmarking dataset, runtime, and analysis infrastructure. 

## **2 Design Principles** 

The design of the ML.ENERGY Benchmark is guided by four core principles. Our overarching goal is to create a benchmark that is representative of real-world generative AI service deployments, and to produce energy measurement results that are accurate, reusable, and ultimately actionable. 

### **2.1 Generalizability and Portability** 

**Goal.** Every computer system is configured with different hardware and software components, and measurements from a particular system will never truly represent those from another system. For instance, systems can be configured with different CPU and DRAM models, and running different 

> 2 `https://github.com/ml-energy/leaderboard` 

2 



Linux kernel versions with different daemons running in the background. Further, not all users have physical access to the target system hardware, a common case for cloud-based environments. Still, we wanted (1) the benchmark to run seamlessly on a wide variety of systems, and (2) measurement results to provide generalizable insights and recommendations across a wide range of systems. 

**Our approach.** 

We focus on software-based GPU energy measurement for the following reasons: 

- GPUs are the dominant worker and energy consumer in a system running ML services, accounting for 50–70% of the total provisioned power in the datacenter [50]. 

- Compared to other hardware components, GPU models are more standardized across different systems [11], making measurements useful across systems that use the same GPU. 

- GPUs allow accurate software-based energy measurement [1,2,9,70], allowing measurement tools to be portable across systems without requiring physical hardware access or modification. 

### **2.2 Representing Real-World Deployments** 

**Goal.** Benchmarking results often inform real-world deployment optimizations, are used to plan future energy usage, affect the design of new hardware and software systems, and serve as base numbers for long term projections that affect policymaking. Therefore, it is crucial that our measurements represent those from real-world deployments as closely as possible. 

**Our approach.** To obtain realistic measurements, we adhere to the following principles: 

- We adopt production-grade software and hardware (e.g., vLLM [34] on NVIDIA H100 GPUs) and run them with generation request workloads that are representative of real-world use cases. 

- During our measurement, we directly run or closely mimic the state of a well-utilized serving system during long term deployment. This allows us to capture the _steady state_ energy consumption of the service while using a fixed-size benchmarking dataset. 

### **2.3 Energy Measurement at the Right Granularity** 

**Goal.** Energy can be measured at different computation granularities. For instance, for LLM text generation, energy can be reported for the end-to-end benchmarking run, for each generated response, or for each token generated. Our goal is to measure and report energy consumption at a granularity that is neither too coarse (as it only provides limited insight into the runtime behavior of the service) nor too fine (as it may miss important higher-level insights relevant to the service). 

**Our approach.** Also aligned with our goal of representing real-world deployments (Section 2.2), our approach is to mainly report energy consumption at the granularity of a single, whole generation response to a request (e.g., entire chat response, image, video). This is because any work less than the full response (e.g., per token) is not considered a complete request, and may ignore model- and task-specific characteristics. For instance, for LLM text generation, different models exhibit different _verbosity_ (i.e., given the same prompt, different models respond with varying number of tokens), and different tasks have vastly different output token length distributions (e.g., chat vs. code generation), all of which we want to capture in our measurements. 

### **2.4 Actionable Measurement Results** 

**Goal.** While energy measurements are useful in themselves, they are even more useful when they lead to actionable insights and recommendations. For instance, how much is the potential energy savings of your model without sacrificing accuracy or latency? If your service allows inference to run 10% slower, what is the energy-optimal configuration, and how much is the potential energy savings? 

**Our approach.** The ML.ENERGY Benchmark allows users to provide computation latency constraints specific to their application scenario (e.g., LLM average Time Per Output Token), and will automatically recommend (1) the _energy-optimal_ configuration that meets the latency constraints, and (2) the expected amount of energy savings. Due to the generalizability of our measurements (Section 2.1), these recommendations generalize to optimizing a wide range of systems. 

3 



![Figure](assets/figure_0002_page_0004.svg)Figure 2: LLM inference server and per-request energy accounting. The steady state is defined as the period when batch size is saturated at the server’s maximum configured batch size, and measurements during the steady state represent that of a well-utilized serving system during long-term deployment. 

## **3 The ML.ENERGY Benchmark** 

The ML.ENERGY Benchmark is a comprehensive tool for measuring and optimizing the inference energy consumption of generative AI models, built upon our core design principles (Section 2). Here, we describe the overall flow of the ML.ENERGY Benchmark (Section 3.1), which includes service-aware energy measurement and accounting (Section 3.2) and automated optimization recommendations (Section 3.3). Finally, we describe extension points of the ML.ENERGY Benchmark that allows users to easily benchmark their customized application scenarios (Section 3.4). 

### **3.1 Benchmark Flow** 

Figure 1 provides an overview of the usage flow of the ML.ENERGY Benchmark. 1 First, the generative model to benchmark and the request dataset (set of inputs) to use are selected, alongside with the set of configurations to sweep (e.g., GPU model, parallelism configuration, maximum batch size). 2 Then the ML.ENERGY Benchmark runs configurations independently on designated hardware, and measures the time and energy consumption of each configuration using Zeus [2], a library that provides programmatic energy measurement (Section 3.2). 3 After benchmarking is complete, users can specify a latency target based on their application requirements. 4 Given that, the ML.ENERGY Benchmark constructs the time–energy Pareto frontier, and recommends the energy-optimal configuration while satisfying the latency target (Section 3.3). 

### **3.2 Energy Measurement and Service-Aware Energy Accounting** 

Our goal is to provide per-request energy measurements (Section 2.3) that are representative of real-world deployments (Section 2.2). However, a realistic serving system batches together the generation of multiple requests (e.g., iteration-level batching [71] for LLM text generation), making the energy consumption of a single request dependent on all other requests being processed at the same time. Therefore, we implement measurement and energy accounting methods that capture the batching behavior of different types of models. 

**Diffusion models.** We begin with the relatively more straightforward case of diffusion models, which are used for text-to-image, text-to-video, and image-to-video generation. Diffusion models are typically batched as a whole, meaning that the energy consumption of a single request is: 



$$
Energyrequest = Energybatch B (1)
$$

where the batch consists of _B_ image or video generation requests. 

**LLM text generation.** Request-level energy accounting is less straightforward for LLM inference, because iteration-level batching [71] is an essential optimization in any realistic LLM serving system [34]. Figure 2 shows how requests are served by a serving system implementing iterationlevel batching and how the ML.ENERGY Benchmark performs energy accounting. Because the 

4 



beginning and end of each request are often not aligned with each other, finding each request’s individual energy consumption is non-trivial. For this, we first submit all requests in the request dataset, and as the system runs, identify the _steady state_ as the time period where the batch size is saturated at the server’s maximum configured batch size. This steady state is designed to closely approximate the state of a serving system when it is well-utilized during long-term deployment. Particularly, when the system is ramping up initially with a full queue or ramping down at the end with an empty queue, the server runs with a smaller batch size and does not exhibit the same energy amortization benefits as the steady state. With this, we can derive the average per-request energy consumption with: 



$$
Energyrequest = Energysteady Tokenssteady \times{}{} 1 N � i Tokensrequest,i. (2)
$$

In essence, we compute the average energy consumption per token during the steady state and multiply it by the average number of output tokens to derive the average per-request energy consumption. 

### **3.3 Automated Optimization Recommendation** 

Our goal is to provide actionable insights beyond just energy measurements (Section 2.4) by recommending energy-optimal configurations for a given model and task. Central to the optimization recommendation is the construction of the _Pareto frontier_ of energy vs. time, which is a collection of configurations where there is no other configuration that leads to both lower energy and lower time. Then, the energy-optimal configuration is selected based on user-specified latency constraints. 

Latency constraints inherently depend on the user’s or application’s needs. For example, for image generation with Diffusion models, computation results are useful only when the full image is generated, so latency constraints would be specified in terms of the time to generate the whole image. On the other hand, for LLM text generation for chat, output tokens are _streamed_ to users (either in written text or synthesized speech) as they are generated, so as long as the average time per output token is at least as fast as the users’ reading or listening speed, user experience will not be affected [39]. However, for LLM text generation for coding, where code is likely only useful when it is fully generated, latency constraints would be specified in terms of the time to generate the whole snippet, similar to the case of image generation. Given the latency constraints, the time–energy Pareto frontier is used to suggest the minimum-energy configuration that satisfies the latency constraint. 

### **3.4 Extending the Benchmark** 

The ML.ENERGY Benchmark is designed to be easily extensible, allowing users to benchmark their own models or customized application scenarios. 

**Model.** The ML.ENERGY Benchmark already supports various popular architectures like Llama [64], LLaVA [38], Stable Diffusion [20], and Stable Video Diffusion [12] (See Appendix A for a full list). Models that are fine-tuned based on already-supported models work as is. Models with different architectures are also supported as is as long as they are supported by the underlying runtime, like vLLM [34], which supports arbitrary LLMs provided by Hugging Face Transformers. 

**Request dataset.** For each task (e.g., LLM text generation for chat), the ML.ENERGY Benchmark provides a default request dataset that contains a set of inputs representative of real-world usage (See Appendix A for a full list). Users can also provide their own request dataset, which is then used to invoke the runtime and measure energy consumption. 

**Configuration space.** The ML.ENERGY Benchmark provides a default set of configurations specific to tasks. For instance, for LLM text generation, it supports maximum batch sizes and parallelism configuration (e.g., tensor and pipeline parallelism). For diffusion models, it supports not only batch size, but also changing the number of denoising steps, as it has a non-trivial impact on time, energy, and output quality. Users can customize the range of values swept for each configuration, and also provide new configurations (e.g., GPU power limit [1, 70]) as long as they implement the corresponding configuration interface in the top-level routine. More configuration dimensions and finer grained sweeps will lead to longer benchmarking time, but will also push the Pareto frontier towards the lower left corner of the time–energy space, leading to the discovery of more energy-efficient configurations. 

5 



![Figure](assets/figure_0003_page_0006.svg)Figure 3: Per-request energy consumption across various generative AI models. Black and orange represents text and vision modalities, respectively. Solid bars are energy measurements, whereas dimmed bars behind each solid bar are estimations based on the GPU’s TDP, with numbers showing the ratio of overestimation. Note the log scale Y-axis. 

**Metrics.** Energy consumption (Joules) reported by the ML.ENERGY Benchmark is a fundamental quantity that can be used to derive other useful metrics. • **Average power draw (Watts)** : Average power draw can be calculated by dividing total energy consumption by the time taken to run the benchmark. This is useful for power provisioning. 

- **Monetary cost ($)** : Monetary cost can be calculated by multiplying energy consumption by the cost of electricity in the region and time frame in which the benchmark was run. 

- **Operational carbon emissions (gCO** 2 **e)** : This quantity estimates the greenhouse gas emissions associated with the electricity consumed. It can be calculated by multiplying energy consumption by the carbon intensity of the particular region and time frame in which the benchmark was run. 

## **4 Results Highlight** 

In this section, we highlight notable results from the ML.ENERGY Benchmark; the full set of results is available on the ML.ENERGY Leaderboard.<sup>3</sup> The most recent iteration of the benchmark and leaderboard presents energy measurements across 40 models and 6 tasks (See Appendix A for a full list). We ran the benchmark on NVIDIA A100 (40 GB) and H100 (80 GB) GPUs, each using AWS p4d.24xlarge and p5.48xlarge instances, respectively, and used vLLM [34] and Diffusers [68] as the inference runtime. In the following, we first present energy measurement results and discuss implications (Section 4.1), and then provide deeper understanding by showing how model architecture choices affect their energy consumption (Section 4.2). Then, we present the energy savings opportunities from our automated optimization recommendations (Section 4.3). 

### **4.1 Energy Measurements** 

**Significant variation in energy consumption.** The solid bars in Figure 3 (A100 GPUs in Figure 3a and H100 in Figure 3b) show the per-request energy consumption of various generative AI models across different tasks. First, energy consumption varies widely across models. In particular, Diffusion models generally consume energy that is on par with larger LLMs (e.g., Mistral Large (123B)). This is mainly because Diffusion models (1) draw higher power in general (more in Section 4.2) and (2) 

> 3 `https://ml.energy/leaderboard` 

6 



![Figure](assets/figure_0004_page_0007.svg)Figure 4: Phi-3 Mini and Small [21] benchmarked with the chat task on one NVIDIA A100 GPU. 

Figure 5: Power consumption of Llama 3.1 70B and Stable Diffusion 3 Medium models. 

cannot perform as many concurrent generations compared to LLMs due to their long latency in real services, preventing them from amortizing energy consumption across many generations. 

**Importance of measuring.** The dimmed bars behind each solid bar in Figure 3 show the estimated energy consumption based on the GPU’s Thermal Design Power (TDP) instead of measuring the real GPU power consumption, which is a common practice [6,7,23,35,42,65]. Estimations using TDP are nearly always an overestimation since it is rare for a GPU – or any computing device – to draw its maximum power at every moment in time. In fact, such an estimation can lead to a worst-case overestimation of energy consumption by a factor of 4.1 (CodeGemma 2B on H100 GPUs). Inaccuracies may be overlooked when they influence downstream decisions and projections, leading to misleading conclusions. Therefore, it is crucial to aim for more accurate measurements. 

### **4.2 Energy Implications of ML Design Decisions** 

ML decisions reflected in model architectures and trained models impact energy consumption. For the interest of space, we defer system parameter implications on energy consumption to Appendix B. 

**LLM response verbosity and energy.** In Figure 3, we can see that energy consumption varies even among LLMs of similar sizes. This is because different LLMs generate responses of different _length_ even when given the same prompt. Such differences in _verbosity_ can be non-trivial; for instance, Mistral Large’s responses were on average 36% longer than that of Mixtral 8 _×_ 7B. As the number of output tokens equals the number of forward passes through the model, longer responses leads to a proportional increase in energy consumption. As humans are known to prefer longer responses [74], this potentially introduces a trade-off between energy consumption and user satisfaction. 

**Memory consumption of operations and energy amortization.** Generally, models with more parameters consume more energy, but this is not always the case. Figure 4 highlights the case of Phi-3 Mini (3.8B) and Small (7B) [21]. Even though Small has nearly twice the parameters, the left plot shows that the larger Small model can consume less energy than Mini as batch size grows. This happens because Mini uses Multi-Head Attention (MA) [67], whereas Small uses Grouped Query Attention (GQA) [8]. Due to this, Mini’s KV cache uses 3 _×_ more memory than Small, which prevents it from scaling to larger batch sizes and amortizing energy consumption across more generations. 

7 



![Figure](assets/figure_0005_page_0008.svg)Figure 6: Energy consumption of SDXL [52] and SDXL Turbo [5] on one NVIDIA A100 GPU. 

Figure 7: Time–energy Pareto frontiers constructed by the ML.ENERGY Benchmark. 

**Compute-intensity of operations and power draw.** Figure 5 shows the power consumption of Llama 3.1 70B [64] and Stable Diffusion 3 Medium [20] on A100 and H100 GPUs. It can be seen that the LLM’s power consumption is much lower than what the GPUs can draw at maximum, whereas the Diffusion model’s power consumption is close to the maximum. This is because LLM decoding is characterized by _low compute-intensity_ , meaning that the number of arithmetic operations (e.g., multiplication and addition) per byte of memory loaded is low [33, 50]. This leads to the GPU’s computation throughput being bottlenecked by VRAM bandwidth and results in the GPU’s computation units being underutilized, leading to low power draw. Appendix C dives deeper into power consumption with measurements for all models and GPU power breakdowns over time. 

**Inference-time parameters and energy.** Figure 6 shows the energy consumption of Stable Diffusion XL (SDXL) [52] and SDXL Turbo [5]. On the left, while SDXL and SDXL Turbo have identical model sizes and architectures, their energy consumption is significantly different. This is because SDXL Turbo is tuned to generate smaller resolution images (512 _×_ 512) than SDXL (1024 _×_ 1024), which leads to different latent sizes and amounts of computation. On the right, it can be seen that the number of denoising steps linearly increases energy consumption, as one denoising step requires one forward pass through the model. While simple in isolation, these inference-time parameters lead to non-trivial design tradeoffs at the application-level. For instance, increasing the number of denoising steps may improve final image quality, but beyond some point, it may be virtually indistinguishable to human users. Also, generating images in lower resolution and then upscaling them with a separate super-resolution model (e.g., DAT [16]) may consume less energy end-to-end. 

### **4.3 Automated Energy Optimization Recommendation** 

Figure 7 shows the time–energy Pareto frontier constructed by the ML.ENERGY Benchmark measurement results for Llama 3.1 8B and Stable Diffusion 2.1. In general, the Pareto frontier is convex, meaning that by sacrificing some latency, one can achieve significant energy savings. 

Conversational AI services like LLM-based chatbots achieve interactivity by streaming tokens to users either in written text or synthesized speech, making Time Per Output Token (TPOT) an important 

8 



performance metric that impacts user experience [39]. In this context, a chatbot provider can target an average TPOT of 100 ms (equivalent to 10 tokens per second or about 7.5 words per second [47]), which is sufficient for most reading or listening speeds. This will land on the Pareto frontier at the point where average TPOT is 77 ms, reducing energy consumption per generation by 44% compared to the configuration that simply minimizes latency. 

Here, we note that for Llama 3.1 8B [64], the Pareto frontier is a mixture of configurations from both A100 and H100 GPUs. This is because LLM decoding does not fully exert the GPU’s compute units and are rather bound by memory, so going from A100 to H100 GPUs neither provides significantly higher performance nor significantly increases power draw (See Appendix C for details). These two – power and time – multiplied, energy consumption is comparable across the two GPUs. 

On the other hand, for Stable Diffusion 2.1 [55], the Pareto frontier is dominated by configurations on the H100 GPU. Diffusion models consume power close to the GPU’s TDP (See Appendix C for details), which increases power draw significantly when going from A100 to H100. However, since computation latency was reduced even more, configurations on H100 Pareto-dominate those on A100. If an application has a generation latency target of, for instance, 5 seconds, the energy-optimal configuration will lie on the Pareto frontier with latency 3.63 seconds, which is 21% less energy than the configuration that minimizes latency. 

## **5 Related Work** 

**ML energy measurement.** The Hugging Face LLM-Perf leaderboard [28] is specific to LLMs and reports the _per-token_ energy consumption of LLM text generation, which fails to capture the verbosity and task-specific output token length distribution difference of LLMs (Section 2.3). MLPerf Power [66] provides measurements for ML training and inference, but crucially, requires direct access to the system under test to physically install the power analyzer, which significantly limits who can run the benchmarks (Section 2.1). Furthermore, it benchmarks at most a few model architectures for each task (sometimes only one), failing to provide insights on how ML design choices impact energy consumption. The Hugging Face AI Energy Score leaderboard [22] provides measurement data for broader AI tasks. However, it fixes the inference batch size to 1 for all models, failing to reflect how services are deployed in the real world (Section 2.2). We emphasize that none of these efforts are fundamentally wrong; they are rather designed for different purposes, complementing the ML.ENERGY Benchmark. The ML.ENERGY Benchmark is the first inference energy benchmark for modern generative AI models, and empowers users to not only measure but also optimize the energy consumption of their models. See Appendix D for more details. 

**ML energy optimization.** The ML.ENERGY Benchmark provides automated energy optimization recommendations based on energy measurements (Section 3.3). There are several other efforts that also provided automated energy optimizations – while preserving mathematical equivalence and/or model quality – for ML training and inference. Zeus [70], EnvPipe [17], and Perseus [18] optimizes the energy consumption of ML training by adjusting GPU-level and training job-level configurations, either statically after profiling or dynamically during training. _µ_ -Serve [54] and DynamoLLM [59] are also similar, but optimize energy consumption for ML inference clusters. Optimization recommendations by the ML.ENERGY Benchmark are complementary to the techniques proposed by these works. Further, our results support the need for automated _cross-layer_ energy optimizations that span all model, software, and hardware layers [19], as opposed to efforts siloed within a single layer. 

## **6 Conclusion** 

In this work, we described the ML.ENERGY Benchmark, a comprehensive energy benchmark for generative AI models that not only provides realistic energy measurements, but also automatically suggests energy-optimal configurations based on user- and app-specific performance constraints. Measurement results show that energy consumption is a metric that is impacted by design choices across the whole AI stack, including application, model, software, and hardware, demonstrating the importance of automated _cross-layer_ energy optimizations instead of siloed optimizations within a single layer. We are confident that the ML.ENERGY Benchmark will democratize the art of measuring, understanding, and optimizing ML energy consumption for the community. 

9 



## **Acknowledgments and Disclosure of Funding** 

We would like to thank Yunseok Jang for his helpful comments and suggestions on the paper. This work is in part supported by NSF grants CNS-2104243, CNS-2106184, and CCF-2327011, grants from VMware and the Mozilla Foundation, and gifts from Salesforce and Google. Jae-Won Chung is additionally supported by the Kwanjeong Educational Foundation. 

## **References** 

- [1] NVIDIA Management Library (NVML). `https://developer.nvidia.com/ nvidia-management-library-nvml` . 

- [2] Zeus: Deep learning energy measurement and optimization. `https://github.com/ ml-energy/zeus` . 

- [3] International Energy Agency. Electricity 2025, 2025. 

- [4] Character AI. Character ai. `https://character.ai` , 2023. 

- [5] Stability AI. Introducing SDXL turbo: A real-time text-to-image generation model, 2023. 

- [6] AI@Meta. Llama 3 model card. 2024. 

- [7] AI@Meta. Llama 4 model card. 2024. 

- [8] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023. 

- [9] Yehia Arafa, Ammar ElWazir, Abdelrahman ElKanishy, Youssef Aly, Ayatelrahman Elsayed, Abdel-Hameed Badawy, Gopinath Chennupati, Stephan Eidenbenz, and Nandakishore Santhi. Verified instruction-level energy consumption measurement for NVIDIA GPUs. _Proceedings of the 17th ACM International Conference on Computing Frontiers_ , 2020. 

- [10] Jeff Bar. Amazon EC2 update – inf1 instances with AWS inferentia chips for high performance cost-effective inferencing, 2019. 

- [11] Nathan Beniach and Air Street Capital. State of AI report compute index. `https://www. stateof.ai/compute` . 

- [12] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. _arXiv preprint arXiv:2311.15127_ , 2023. 

- [13] CBRE. Global data center trends 2023. `https://www.cbre.com/insights/reports/ global-data-center-trends-2023` , 2023. 

- [14] CBRE. Global data center trends 2024. `https://www.cbre.com/insights/reports/ global-data-center-trends-2024` , 2024. 

- [15] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. _Advances in Neural Information Processing Systems Datasets and Benchmarks_ , 2024. 

- [16] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xiaokang Yang, and Fisher Yu. Dual aggregation transformer for image super-resolution. _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2023. 

- [17] Sangjin Choi, Inhoe Koo, Jeongseob Ahn, Myeongjae Jeon, and Youngjin Kwon. EnvPipe: Performance-preserving DNN training framework for saving energy. _Proceedings of the 2023 USENIX Annual Technical Conference_ , 2023. 

- [18] Jae-Won Chung, Yile Gu, Insu Jang, Luoxi Meng, Nikhil Bansal, and Mosharaf Chowdhury. Reducing energy bloat in large model training. _Proceedings of the 30th ACM Symposium on Operating Systems Principles_ , 2024. 

10 



- [19] Jae-Won Chung, Nishil Talati, and Mosharaf Chowdhury. Toward cross-layer energy optimizations in AI systems. _DOE ASCR Energy-Efficient Computing for Science Workshop_ , 2024. 

- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In _ICML_ , 2024. 

- [21] Marah Abdin et al. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_ , 2024. 

- [22] Hugging Face. Ai energy score. `https://huggingface.github.io/AIEnergyScore` , 2025. 

- [23] Meta GenAI. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint_ , 2023. 

- [24] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning. In _ICLR_ , 2024. 

- [25] Yatharth Gupta, Vishnu V. Jaddipal, Harish Prabhala, Sayak Paul, and Patrick Von Platen. Progressive knowledge distillation of stable diffusion xl using layer level loss. _arXiv preprint arXiv:2401.02677_ , 2024. 

- [26] The White House. Fact sheet: President donald j. trump establishes the national energy dominance council, 2025. 

- [27] HPCwire. AWS to offer NVIDIA’s T4 GPUs for AI inferencing, 2019. 

- [28] Régis Pierrard Ilyas Moutawwakil. LLM-Perf leaderboard. `https://huggingface.co/ spaces/optimum/llm-perf-leaderboard` , 2023. 

- [29] Independent Institute. AI boom power surge: Plants revived, fossil fuels reconsidered, 2025. 

- [30] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- [31] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- [32] Heehoon Kim, Junyeol Ryu, and Jaejin Lee. TCCL: Discovering better communication paths for PCIe GPU clusters. _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ , 2024. 

- [33] Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, Minwoo Kang, Ruohan Yan, Hasan Genc, Grace Dinh, Qijing Huang, Kurt Keutzer, Michael W. Mahoney, Sophia Shao, and Amir Gholami. Full stack optimization of transformer inference. _Architecture and System Support for Transformer Models_ , 2023. 

- [34] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. _Proceedings of the 29th Symposium on Operating Systems Principles_ , 2023. 

- [35] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. _arXiv preprint_ , 2019. 

- [36] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _CVPR_ , 2024. 

- [37] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, ocr, and world knowledge, 2024. 

11 



- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in Neural Information Processing Systems_ , 2023. 

- [39] Jiachen Liu, Jae-Won Chung, Zhiyu Wu, Fan Lai, Myungjin Lee, and Mosharaf Chowdhury. Andes: Defining and enhancing quality-of-experience in LLM-based text streaming services. _arXiv preprint arXiv:2404.16283_ , 2024. 

- [40] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In _NeurIPS_ , 2023. 

- [41] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. _arXiv preprint arXiv:2402.19173_ , 2024. 

- [42] Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language model. _Journal of Machine Learning Research_ , 2024. 

- [43] McKinsey & Company. Investing in the rising data center economy. `https: //www.mckinsey.com/industries/technology-media-and-telecommunications/ our-insights/investing-in-the-rising-data-center-economy` , 2023. 

- [44] McKinsey & Company. How data centers and the energy sector can sate AI’s hunger for power. `https://www.mckinsey.com/industries/private-capital/our-insights/ how-data-centers-and-the-energy-sector-can-sate-ais-hunger-for-power` , 2024. 

- [45] Midjourney. Midjourney. `https://midjourney.com` , 2022. 

- [46] Sebastian Moss. Meta’s mark zuckerberg says energy constraints are holding back AI data center buildout, 2024. 

- [47] OpenAI. What are tokens and how to count them? `https://help.openai.com/en/ articles/4936856-what-are-tokens-and-how-to-count-them` . 

- [48] OpenAI. ChatGPT. `https://chatgpt.com` , 2022. 

- [49] OpenAI. Sora. `https://openai.com/index/sora` , 2024. 

- [50] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Brijesh Warrier, Nithish Mahalingam, and Ricardo Bianchini. Characterizing power management opportunities for llms in the cloud. _ASPLOS_ , 2024. 

- [51] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. _arXiv preprint_ , 2021. 

- [52] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In _ICLR_ , 2024. 

- [53] PromptHero. OpenJourney v4, 2023. 

- [54] Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Tamer Ba¸sar, and Ravishankar K. Iyer. Power-aware deep learning model serving with u-Serve. In _ATC_ , 2024. 

- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 

- [56] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ , 2024. 

- [57] Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green AI. _Commun. ACM_ , 63(12):54–63, 2020. 

12 



- [58] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. _arXiv preprint_ , 2019. 

- [59] Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, and Esha Choukse. DynamoLLM: Designing llm inference clusters for performance and energy efficiency. In _HPCA_ , 2025. 

- [60] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. _arXiv preprint arXiv:2405.09818_ , 2024. 

- [61] CodeGemma Team, Heri Zhao, Jeffrey Hui, Joshua Howland, Nam Nguyen, Siqi Zuo, Andrea Hu, Christopher A. Choquette-Choo, Jingyue Shen, Joe Kelley, Kshitij Bansal, Luke Vilnis, Mateo Wirth, Paul Michel, Peter Choy, Pratik Joshi, Ravin Kumar, Sarmad Hashmi, Shubham Agrawal, Zhitao Gong, Jane Fine, Tris Warkentin, Ale Jakse Hartman, Bin Ni, Kathy Korevec, Kelly Schaefer, and Scott Huffman. CodeGemma: Open code models based on gemma. _arXiv preprint arXiv:2406.11409_ , 2024. 

- [62] Gemma Team. Gemma 2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_ , 2024. 

- [63] ShareGPT Team. ShareGPT. `https://sharegpt.com/` . 

- [64] Llama team at Meta. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- [65] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. _arXiv preprint_ , 2023. 

- [66] Arya Tschand, Arun Tejusve Raghunath Rajan, Sachin Idgunji, Anirban Ghosh, Jeremy Holleman, Csaba Kiraly, Pawan Ambalkar, Ritika Borkar, Ramesh Chukka, Trevor Cockrell, Oliver Curtis, Grigori Fursin, Miro Hodak, Hiwot Kassa, Anton Lokhmotov, Dejan Miskovic, Yuechao Pan, Manu Prasad Manmathan, Liz Raymond, Tom St. John, Arjun Suresh, Rowan Taubitz, Sean Zhan, Scott Wasson, David Kanter, and Vijay Janapa Reddi. MLPerf power: Benchmarking the energy efficiency of machine learning systems from uWatts to MWatts for sustainable ai. In _HPCA_ , 2025. 

- [67] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 

- [68] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, Steven Liu, William Berman, Yiyi Xu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. `https://github.com/huggingface/ diffusers` . 

- [69] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope text-to-video technical report. _arXiv preprint arXiv:2308.06571_ , 2023. 

- [70] Jie You, Jae-Won Chung, and Mosharaf Chowdhury. Zeus: Understanding and optimizing GPU energy consumption of DNN training. _NSDI_ , 2023. 

- [71] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for Transformer-Based generative models. In _OSDI_ , 2022. 

- [72] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. _Transactions on Machine Learning Research_ , 2022. 

- [73] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2VGen-XL: High-quality image-to-video synthesis via cascaded diffusion models. _arXiv preprint arXiv:2311.04145_ , 2023. 

- [74] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In _NeurIPS_ , 2023. 

13 



Table 1: Model type, task, and default request dataset used in the ML.ENERGY Benchmark. 

|**Model architecture**|**Task**|**Request dataset**|
|---|---|---|
|Large Language Model|Chat<br>Code|ShareGPT [63]<br>EvalPlus [40]|
|Vision Language Model|Visual chat|LLaVA instruction dataset [38]|
||Text-to-image|PartiPrompts [72]|
|Diffusion Model|Text-to-video|Captions in ShareGPT4Video [15]|
||Image-to-video|Captions and first frames in ShareGPT4Video [15]|



Table 2: Model architectures supported by the ML.ENERGY Benchmark for each task. 

|**Task**|**Model Architectures**|
|---|---|
|Chat|Gemma 2 2B/9B/27B [62], Llama 3.1 8B/70B/405B [64],<br>Phi 3 Mini/Small/Medium [21], Mistral 7B/Nemo/Large [30],<br>Mixtral 8x7B/8x22B [31]|
|Code|CodeLlama 7B/13B/34B/70B [56], StarCoder 2 3B/7B/15B [41],<br>CodeGemma 2B/7B [61]|
|Visual chat|LLaVA 1.5 7B/13B [36], LLaVA NeXT 8B [37], Phi 3 Vision [21],<br>Chameleon 7B/30B [60]|
|Text-to-image|Stable Diffusion 2.1/XL/XL Turbo/3 Medium [5,20,52,55],<br>OpenJourney 4 [53], SSD 1B [25]|
|Text-to-video|ModelScope T2V [69], AnimateDiff [24]|
|Image-to-video|I2VGen XL [73], Stable Video Diffusion and Stable Video Diffusion XT [12]|



## **A Tasks, Model Architectures, and Default Request Datasets** 

Tables 1 and 2 list the model architectures and tasks supported by current iteration of the ML.ENERGY Benchmark, along with the default request datasets for each task. We note that models that were fine-tuned based on the supported models are also supported as is, and the benchmark is designed to be extensible (Section 3.4). 

The ML.ENERGY Benchmark cannot avoid being outdated given the rapid pace of development in the generative AI field. As such, we have been updating the benchmark (and the accompanying Leaderboard) with new tasks, models, datasets, hardware, runtimes, and more, and we intend to continue doing so as long as resources allow. 

## **B Energy Implication of System Parameters** 

This section discusses the energy implication of different system-level configurations. System-level configurations are those that do not change _what_ is computed but rather _how_ it is computed by the underlying software system. 

### **B.1 Request Preemption Mechanism** 

Even with the model and inference parameters fixed, the software system used to serve inference requests, which determines how model computations are executed on a given hardware, significantly impacts energy consumption. As a concrete example, we will examine the effect of “preemption mechanism,” a configuration parameter for LLM inference servers. When a server is overloaded with more requests than its capacity, it needs to temporarily remove (or, preempt) some requests from the system and then later bring them back (or, restore). For LLM inference, there are two widely-used mechanisms for preemption: Recomputation and Swapping [34]. Recomputation simply drops all temporary request data or state on preemption and recomputes everything from scratch on restoration. On the other hand, Swapping moves the request state to the CPU’s memory, and then returns it to the GPU on restoration. The best preemption mechanism depends on the computing hardware and software configuration and the LLM being served. 

14 



![Figure](assets/figure_0006_page_0015.svg)Figure 8: Energy consumption per generation while varying the maximum batch size for Mistral Nemo (12B). The LLM inference server’s preemption mechanism is compared. 

Figure 9: Energy consumption per generation while varying batch size for Llama 3.1 8B. The number of NVIDIA A100 GPUs used to run the same model is scaled up. 

Figure 8, we compare the energy consumption per generation of the two preemption mechanisms with the Mistral Nemo (12B) model by intentionally overloading the server with a high maximum batch size configuration and causing preemption. It can be seen that when the server is overloaded, Swapping consistently consumes less energy. This is because Recomputation performs extra computation when restoring requests whereas Swapping copies data without running computation, and the energy consumption of computation is larger than memory operations (this will be further examined in the next section). Furthermore, as the server gets more and more overloaded, energy consumption generally increases. This is because with higher overload, more preemptions – and thus more recomputation or data movement – occur. Since preemptions do not directly contribute to the completion of the request, the extra energy consumption from preemptions increases the average energy consumption of completing each request. 

### **B.2 Tensor Parallelism Scaling** 

We investigate the impact of communication overhead to energy consumption. This is important as modern large models frequently do not fit within the memory capacity of a single GPU. This requires multiple GPUs to execute inference for a single model, and GPUs must constantly communicate with each other to do so [58]. 

In order to ablate the effect of communication, we employ the same Llama 3.1 8B model and vary the number of GPUs used (Figure 9). Because the amount of computation executed is the same regardless of the number of GPUs, energy consumption should ideally be constant. Indeed, energy consumption barely changes when scaling from one GPU (no communication) to two, but when scaling further, energy consumption significantly increases. This is because, while the amount of computation decreases for each GPU, additional communication time between the GPUs offsets the reduction in computation time. Since communication time increases with the number of GPUs, using too many GPUs can lead to slowdowns in executing the same amount of computation and increase energy consumption. 

15 



![Figure](assets/figure_0007_page_0016.svg)Figure 10: Power consumption of various models on A100 and H100 GPUs. 

From this scaling experiment, we can observe that the energy impact of communication overhead can be large. This impact will be even more pronounced in hardware environments without sufficient or state-of-the-art networking infrastructure, which is common in real world settings due to its cost [32]. 

## **C Power Consumption Analysis** 

Figure 10 shows the power consumption of various models on A100 and H100 GPUs. Figure 11 further shows the ratio of a model’s power consumption to the maximum GPU power draw across all models. Generally, LLMs and VLMs consume significantly less power than the GPU’s TDP because LLM decoding, the dominant operation for LLM serving, is memory-intensive and does not fully utilize the GPU’s compute resources. VLMs show slightly higher power consumption than LLMs due to its additional modality encoder, which is compute-intensive. Diffusion models, on the other hand, consume nearly the maximum power of the GPU when batch size is not small. This is because Diffusion models are significantly more compute-intensive compared to LLM decoding. 

Figure 12 shows the GPU power draw breakdown over time on one NVIDIA H100 GPU. The GPU’s power is measured (1) in whole and (2) only for the VRAM while the ML.ENERGY Benchmark is running. First, for Llama 3.1 8B, the timeline shows the effect of the two phases in LLM text generation: Prefill and Decode. Prefill happens once at the beginning of a request to digest the input 

16 



![Figure](assets/figure_0008_page_0017.svg)Figure 11: Ratio of power consumption to maximum GPU power draw across various models. 

![Figure](assets/figure_0009_page_0017.svg)(b) Stable Video Diffusion XT 

Figure 12: GPU power draw breakdown over time on one NVIDIA H100 GPU. “Entire GPU” and “Only VRAM” (memory) were measured, and the two were subtracted to derive “Entire GPU excluding VRAM.” 

prompt, which is then followed by hundreds to thousands of Decode phases, each of which generates one output token. Importantly, Prefill has high compute-intensity (and also high power draw) because it needs to digest the whole input prompt, whereas Decode has low compute-intensity (and low power draw) as it does not entail very much computation. With this, we can first understand the initial spike in power draw – when the benchmark begins, the server begins admitting new requests, creating a short period where numerous Prefills are executed back-to-back, leading to high power draw. After the initial spike, power draw repeats a periodic fluctuation. This is because, before each Prefill or Decode, the server must make numerous control decisions, including determining which requests are now finished and which ones should run next. Since these decisions are executed by the CPU, this 

17 



creates a periodic time gap where the GPU is not running any computation. This GPU idle time leads to the periodic drop in GPU power draw. 

On the other hand, Stable Video Diffusion XT shows a different power draw pattern. Diffusion models generally have three phases: Encode, Denoise, and Decode. The Encode phase digests the input prompt and passes it to the Denoise phase, which iteratively removes noise from a random vector. Finally, the Decode phase transforms the denoised vector into the final image or video. 

From the timeline, especially Denoise and Decode can be clearly distinguished. Denoise is the most compute-intensive and consumes power close to the GPU’s TDP. For each batch, there are 25 local peaks that hit the GPU’s TDP, each of which corresponds to one denoising step in Denoise. During Decode, power draw generally decreases, with each local power peak corresponding to the two large layers in the decoding module. On the other hand, VRAM power draw increases during Decode because it allocates a large chunk of memory and performs writes in order to generate the final video. Finally, as the final generated video is copied from the GPU’s memory to the CPU’s, the GPU does not run any computation, resulting in a steep drop in power draw. 

From the power breakdown, we can observe that memory operations indeed draw significantly less power compared to computation, and thus computations with low compute-intensity should indeed draw less power. Furthermore, we can observe that the power draw and energy consumption of a specific hardware (GPU in this case) is not a function of just itself and the computations that it runs. Rather, software and hardware components that are integrated in the same system stack impacts how computations are executed on each other, affecting their power draw and energy consumption. 

## **D The ML.ENERGY Leaderboard and Benchmark** 

On July 2023, we launched the ML.ENERGY Leaderboard and Benchmark, the first inference energy leaderboard for modern generative AI models.<sup>4</sup> Our goal was to measure and understand the energy consumption of generative AI models, and we provided a web-based leaderboard to allow everyone to browse the results. The leaderboard started with only LLM chat with tens of different LLMs, but gradually expanded to include more tasks, models, and datasets. Our benchmarking suite to supply data to the leaderboard is what we dub the ML.ENERGY Benchmark. This paper shares our design philosophy and principles we have acquired over time by gradually maintaining and upgrading the ML.ENERGY Benchmark and the Leaderboard, and highlights notable results we have obtained from the latest iteration of the benchmark. 

## **E Limitations** 

The ML.ENERGY Benchmark is not without limitations. First, we note that the benchmark is not exhaustive and does not cover all possible tasks, models, and datasets. This is particularly true as time passes and new models and tasks are developed. We are aware of newer open-weight models and worthy tasks that were released after the latest iteration of the benchmark was finalized. However, we cannot add each model or task one by one incrementally as they are released, due to the prohibitive monetary cost of running the benchmark on representative hardware; rather, we collect new advances in a window of time and then mass-update the whole benchmark, accompanied by upgrades in hardware, software, and datasets. Second, the benchmark is not exhaustive in terms of hardware. We currently mainly support flagship NVIDIA GPUs, which arguably dominates the market especially when it comes to real-world generative AI services. Furthermore, we do not have access to all possible hardware configurations, nor do they always provide a way for us to measure energy consumption from software. Regardless, we are working to expand the benchmark to support more hardware configurations. 

## **F Broader Impacts** 

By allowing everyone to accurate measure, understand, and optimize the energy consumption of generative AI models, we believe the ML.ENERGY Benchmark can help reduce the energy consumption of generative AI models across our community. This can lead to better democratization 

> 4 `https://github.com/ml-energy/leaderboard/releases/tag/2023-07-06` 

18 



of the technology we develop by lowering the cost and power provisioning barrier, and enhance environmental sustainability. The authors are not aware of any negative societal impacts of the benchmark. 

19 

