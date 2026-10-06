1 

# The Price of Prompting: Profiling Energy Use in Large Language Models Inference 

Erik Johannes Husom, Arda Goknil, Lwin Khin Shar, and Sagar Sen 

✦ 

**Abstract** —In the realm of Artificial Intelligence (AI), the deployment of Large Language Models (LLMs) has become fundamental to advancing technology and enriching user experiences across various applications. However, as these models grow in complexity and size, they pose significant computational and environmental challenges due to their substantial energy consumption. The increasing deployment of LLMs accentuates the need for sustainable practices to manage their energy demands effectively, especially in resource-constrained environments. In this paper, we introduce MELODI – **M** onitoring **E** nergy **L** evels and **O** ptimization for **D** ata-driven **I** nference – a multifaceted framework crafted to monitor and analyze the energy consumed during LLM inference processes. Unlike existing tools that measure energy at the system level, MELODI uniquely combines process-level CPU monitoring via Scaphandre with GPU tracking via nvidia-smi, employing configurable temporal buffers to capture complete inference cycles with high precision. MELODI enables detailed observations of power consumption dynamics and facilitates the creation of datasets reflective of energy efficiency across various deployment scenarios. We apply MELODI to profile inference-time energy consumption under a diverse set of scenarios. Our findings reveal pronounced energy efficiency disparities: large models ( _≥_ 70B parameters) consume roughly two orders of magnitude more energy per token than smaller models, while response token length is a strong predictor of energy use (R<sup>2</sup> _>_ 0.95 in most settings). We further derive a predictive model with an R<sup>2</sup> value of 0.9962, utilizing response length, model type, and hardware. Prompt characteristics exhibit minimal correlation with energy consumption. In contrast, hardware choice has a substantial impact—laptop deployments consistently consume more energy than workstations for comparable models—highlighting significant opportunities for optimization and more sustainable LLM deployment. **1 INTRODUCTION** The swift progression of artificial intelligence (AI) has 

The swift progression of artificial intelligence (AI) has precipitated the emergence of advanced natural language processing (NLP) systems. Large language models (LLMs) have ascended to prominence among these systems, proving indispensable across a vast spectrum of applications. Their utility ranges from simple text predictions to managing complex dialogues. In various domains, LLMs have shown great promise in automating and enhancing a variety of tasks that traditionally require significant human expertise and effort [1]–[5]. For instance, LLMs can assist in content generation [6]–[8], customer support [9]–[11], machine translation [12]–[14], sentiment analysis [15]–[17], and legal 

- _E.J. Husom, A. Goknil, and S. Sen are affiliated with SINTEF, Norway. L.K. Shar is affiliated with Singapore Management University, Singapore. E-mails: erik.johannes.husom@sintef.no arda.goknil@sintef.no lkshar@smu.edu.sg sagar.sen@sintef.no_ 

- _Manuscript received June 2025; revised January 2026._ 

document review, thereby streamlining operations across diverse sectors. The ability of LLMs to generate humanlike text with high accuracy and relevance has made them invaluable tools for professionals, enabling them to focus on more complex and creative aspects of their disciplines. 

**Context and Motivation.** As these models grow in computational complexity, their energy demands escalate correspondingly [18]–[20]. The widespread deployment of LLMs across various applications, from code generation tools used in iterative workflows to general-purpose chatbots with hundreds of millions of users, makes individual usage accumulate to massive resource consumption. Unlike training, which occurs once per model, inference is a continuous operational cost that scales with usage. In an epoch that prioritizes sustainable computing, the energy expenditure of LLMs is a consideration of mounting importance and one that commands focused attention [2], [21]. Therefore, the integration of LLMs in diverse applications from healthcare to customer service without compounding the carbon footprint becomes paramount, thus underscoring the need for techniques that can mitigate the energy demands of LLMs while maintaining their functionality and accessibility. 

**Problem Statement.** Despite the widespread adoption of LLMs, understanding their energy implications remains a significant challenge [22]. Prior studies have predominantly concentrated on the training phase of traditional machine learning (ML) models, with the inference phase often relegated to a secondary focus despite its cumulative energy footprint over the lifespan of a model [23]. Tools such as Green Algorithms [24] provide estimations of carbon emissions for computational tasks and yet lack real-time monitoring capabilities. Furthermore, while initiatives like CodeCarbon [25] and its integration to ML pipelines [26] offer a more automated tracking of emissions, they neither cater to the nuanced demands of LLMs nor provide granular data concerning inference-related energy consumption. 

Recent research [21], [27]–[33] highlights the significant energy and carbon footprints of LLM configurations. Wilkins et al. [28] introduce an offline, workload-based framework to optimize energy consumption for LLM inference on mixed GPU-CPU systems, focusing on energy usage and runtime based on token interactions. Their method faces limitations like computational overhead due to frequent sampling and lack of granularity. Conversely, Samsi et al. [27] examine the energy consumption of GPUs in LLaMA models, excluding CPU impacts. Prior work neither 



2 

systematically applies a comprehensive framework for monitoring and analyzing LLM inference energy across diverse operational conditions, nor investigates real-time optimization across heterogeneous hardware. To address this gap, we aim to develop a holistic energy monitoring framework tailored to LLM deployments and use it to analyze how prompt characteristics and hardware configurations influence energy consumption. The framework should advance understanding of LLM energy use while enabling datadriven optimization strategies to support more efficient and sustainable AI engineering. 

**Our Design.** This paper introduces the MELODI framework designed to monitor and analyze energy usage during LLM inference. The urgent need for such a framework stems from the growing environmental and economic costs associated with the operation of LLMs. MELODI stands at the intersection of ML, software engineering, and sustainability, encapsulating a holistic approach. MELODI leverages two specialized power usage monitoring tools (Scaphandre [34] and Nvidia-smi [35]). Scaphandre tracks the CPU’s power consumption for the LLM process, while nvidia-smi measures the GPU’s total power usage. For accurate measurements, the GPU must be exclusively used for the LLM inference, with no concurrent processes. MELODI records the energy consumption for each inference process, in addition to the prompt and the LLM’s response. Furthermore, it facilitates the collection of metadata associated with LLM inference tasks, enabling a deeper analysis of the relationship between task complexity and energy utilization. 

**Exploratory Study.** We conducted a study using MELODI to profile inference-time energy consumption under a diverse set of LLM inference scenarios. We deployed multiple open-source LLMs across heterogeneous hardware platforms, ranging from CPU-only laptops to GPUequipped workstations and servers, and evaluated them on two representative prompt datasets (Alpaca [36] and Code-Feedback [37]). For each inference request, MELODI recorded prompt–response pairs, token-level metadata, timestamps, and CPU/GPU power traces, which were integrated into an energy consumption dataset enabling comparative analyses across model types, model sizes, hardware configurations, and workload characteristics. 

Our study reveals pronounced variability in LLM inference energy across deployment conditions: 70B-class models consume up to two orders of magnitude more energy per token than smaller models, and laptop deployments are generally less efficient than workstations, with measurable differences even between models of similar size. Energy consumption is driven primarily by response characteristics (response length and duration), while prompt complexity features show weak associations. Using response length, we achieve highly accurate energy prediction and derive an interpretable model with near-perfect performance ( _R_<sup>2</sup> = 0 _._ 9962) based on response length, model type, and hardware. Finally, repeated runs show model-dependent variability, and comparisons with other monitoring tools expose substantial measurement discrepancies, highlighting the need for robust inference-time energy monitoring. 

To summarize, our contributions include: 

- **MELODI Framework:** We introduce MELODI, an 

open-source and extensible framework for finegrained monitoring and analysis of CPU and GPU energy consumption during LLM inference, enabling reproducible energy profiling at the level of individual inference requests. 

- **Energy Consumption Dataset:** Using MELODI, we construct and release a comprehensive inferencetime energy dataset spanning multiple hardwares, LLM families and model sizes, and prompt datasets, supporting systematic benchmarking of energy efficiency across deployment conditions. 

- **Empirical Characterization of Energy Drivers:** We conduct an extensive exploratory study identifying how model size, model type, hardware configuration, and prompt/response characteristics influence inference energy consumption, providing evidencebased insights for energy-aware deployment. 

- **Prediction and Interpretable Modeling:** We develop and assess predictive models for inference energy, demonstrating that energy can be reliably modeled from response length, model type, and hardware. 

The paper is structured as follows. Section 2 presents the background regarding energy monitoring tools. In Section 3, we present MELODI. Section 4 reports on the experiments. Section 5 presents some future work. Section 6 discusses the related work. We conclude the paper in Section 7. 

## **2 BACKGROUND: MONITORING TECHNOLOGIES** 

Advanced monitoring tools and technologies are essential for tracking and optimizing energy consumption. In this context, MELODI employs the following tools: 

**NVIDIA System Management Interface (nvidiasmi)** [35] is a command-line utility, based on NVIDIA Management Library (NVML) [38], providing the real-time monitoring of GPU, including power consumption, for systems equipped with NVIDIA GPUs. This allows administrators to query GPU device state and with the appropriate privileges, permits administrators to modify GPU device state [35]. 

**Scaphandre** [34] is a metrology agent designed to enhance energy management across various computing environments. It is one of the few tools providing detailed insights into the energy usage of computing systems at the process level [39]. It offers comprehensive features including the ability to measure power consumption directly on bare metal hosts and within qemu/kvm virtual machines from the host. Additionally, Scaphandre facilitates the integration of power metrics into virtual machines as if they were bare metal setups and supports exporting these metrics as a Prometheus HTTP exporter [40]. 

## **3 MELODI FOR MONITORING ENERGY USAGE** 

To analyze LLM energy consumption, we developed the MELODI framework, which monitors energy use during LLM inference. Figure 1 outlines its architecture, highlighting its process of sampling prompts, submitting them to an LLM, and recording the prompts, responses, metadata, and energy metrics. This collected data allows for a comprehensive analysis of energy trends during LLM operations. 



3 

![Figure](assets/figure_0001_page_0003.svg)Fig. 1: Overview of the MELODI framework. 

- **Input Data (Prompt Dataset in Figure 1):** MELODI extracts prompts from a pre-existing dataset to query a deployed LLM through an API client. 

- **Power Usage Monitoring and Data Collection:** MELODI utilizes two complementary power monitoring tools (i.e., Scaphandre [34] and nvidiasmi [35]) to track the power consumption of LLMs like the Ollama model in Figure 1. Scaphandre records CPU power consumption specific to the LLM process, while nvidia-smi measures total GPU power usage, requiring exclusive GPU operation (no other processes on the GPU) during measurements for accuracy. MELODI calculates energy consumption for each prompt, aggregating data on the prompt, response, and energy metrics for detailed analysis. 

- **Dataset Generation:** MELODI iterates through all prompts from the prompt dataset. The data obtained for each prompt are aggregated into a dataset, encapsulating the energy consumption of LLM inference. 

MELODI relies on LLM services that can be queried via API endpoints for generating responses. Our implementation supports services adhering to Ollama or OpenAI API specifications, covering most LLM deployment services that provide natural language responses and metadata such as timestamps and token lengths for prompts and responses. The landscape of LLM deployment tools is diverse, featuring platforms such as Llama.cpp [41], GPT4All [42], vLLM [43], Llamafile [44], and Ollama [45]. We utilized Ollama for our experiments due to its easy setup, compatibility with both CPU and GPU architectures, and support for several open-source LLMs. 

Scaphandre operates with process-level granularity, monitoring and recording the power usage of active processes on a computing device. We apply a regular expression to the process names to isolate the specific processes associated with the LLM service. The frequency at which Scaphandre samples power metrics is configurable with granularity fine enough to reach the nanosecond scale. Although a higher sampling frequency increases the resolution and accuracy of the data, it also requires more storage for data retention and increases the energy consumption of Scaphandre. Therefore, we also monitor the energy consumption of Scaphandre, enabling an assessment of the energy overhead incurred by the measurement process. 

Nvidia-smi tracks the power usage of the entire GPU and cannot distinguish among multiple processes concurrently utilizing the GPU. To guarantee precision, we limit the GPU to solely operating the LLM inference process. The two tools measure the power consumption _P_ of a process in microwatts ( _µ_ W). We compute the energy consumption _E_ by integrating the power consumption over the duration _t_ , applying the trapezoidal rule. With _t_ (in seconds), we convert _E_ into kilowatt-hours (kWh) by dividing by 3 _._ 6 _×_ 10<sup>12</sup> , facilitating a standardized energy usage assessment. 

In each inference, MELODI activates 60 Scaphandre to monitor 40 CPU power usage at the process level, 20 LLM service (CPU) targeting the LLM Power consumption (W) 0 LLM service (GPU) service to ensure 6.50 6.75 7.00Time (s)7.25 7.50 7.75 measurement accuracy. Concurrently, nvidia(a) Power draw during inference ( _M_ 0 = _M_ 1 = 0 and _R_ 0 = _R_ 1 = 0). smi tracks GPU power usage at the device 60 LLM service (CPU) LLM service (GPU) level, necessitating 40 the restriction of GPU 20 activities solely to LLM inference to eliminate Power consumption (W) external application 8.50 8.75 9.00 9.25 9.50 9.75 Time (s) interference. MELODI logs the prompt, (b) Power draw during inference response, token counts, ( _M_ 0 = _M_ 1 = 0 _._ 5 seconds and and the timestamps _R_ 0 = _R_ 1 = 0). 80 marking the start and LLM service (CPU) 60 LLM service (GPU) end of the inference, while recording the 40 energy consumption 20 and power draw as a Power consumption (W) time series. 0.25 0.50 0.75 1.00 1.25 1.50 1.75 Time (s) To enhance energy 

Time (s) To enhance energy measurement accuracy (c) Power draw during inference during inference, we ( _M_ 0 = _M_ 1 = 0 _._ 5 seconds, and _R_ 0 = 0 and and _R_ 1 = 0 _._ 2 seconds). seconds). implement two types of buffers in MELODI to Fig. 2: Power draw during a compensate for moniprompt inference as monitored toring tool delays. The by MELODI with varying buffer first, _monitoring buffer_ settings. Thick semi-transparent ( _M_ ), introduces a delay lines indicate active monitoring before and after the inintervals by Scaphandre (CPU) ference process to enand nvidia-smi (GPU), while the sure Scaphandre and solid line with dots are data nvidia-smi are fully oppoints recorded by MELODI. erational also before and after the inference. This buffer is divided into two configurable intervals: _M_ 0, the delay from monitoring startup to inference start, and _M_ 1, the delay from inference end to monitoring shutdown, effectively preventing data loss at the critical start and end inference points due to tool response times. 

(c) Power draw during inference ( _M_ 0 = _M_ 1 = 0 _._ 5 seconds, and _R_ 0 = 0 and and _R_ 1 = 0 _._ 2 seconds). seconds). 

Figures 2a and 2b demonstrate the impact of employing a monitoring buffer on data capture. Figure 2a, utilizing no buffer ( _M_ 0 = _M_ 1 = 0), shows that nvidia-smi fails to capture data during extremely brief inference runs, resulting in incomplete energy measurements. Conversely, Figure 2b, using a 0.5-second buffer for pre- and post- 



4 

TABLE 1: Capture completeness rate (CCR) for different buffer settings, measured as the percentage of runs (50 per setting) in which the full power spike was captured. In some cases, achieving 100% CCR includes excess baseline power outside the inference window. 

|(a)_M_ =|_M_0 =_M_1|(b)|_R_0|(c)|_R_1|
|---|---|---|---|---|---|
|_M_ (s)|CCR|_R_0 (s)|CCR|_R_1 (s)|CCR|
|0.2|0%|0.0|94%|0.0|34%|
|0.4|48%|0.1|100%|0.1|36%|
|0.5|100%|0.2|100%|0.2|88%|
|1.0|100%|0.3|100%|0.3|100%|



inference ( _M_ 0 = _M_ 1 = 0 _._ 5), shows improved reliability in data capture, accounting even for brief inference periods. We tested multiple values of _M_ (Table 1a) and found that _M_ = 0 _._ 5 provides optimal data capture without including unnecessary measurements. A monitoring interval was considered successful when the power profile was stable both before and after the inference window. 

Even with a monitoring buffer, we observe that GPU power spikes occasionally persist briefly post-inference, as depicted in Figure 2b. To address this, we introduce a _recording buffer_ ( _R_ ), extending the monitoring period before and/or after the actual inference window. This buffer is split into _R_ 0, extending the recording time before inference, and _R_ 1, extending after. _R_ 1 is crucial for capturing lingering GPU power spikes after inference concludes, ensuring comprehensive data collection (see Figure 2c). This delayed power consumption decline, also observed in other studies [46], underscores the necessity of an end buffer for precise measurements. 

Determining optimal buffer values of _R_ involves a tradeoff between measurement completeness and estimation bias. Larger buffers ensure full capture of transient power spikes but can overestimate energy by including baseline consumption outside the inference window. Smaller buffers reduce this bias but may miss lingering power decay, particularly on GPUs (Figure 2c). As shown in Table 1b, _R_ 0 = 0 _._ 1 achieves complete capture but consistently includes 0.1 seconds of pre-inference baseline power, whereas _R_ 0 = 0 _._ 0 removes this offset while still capturing 94% of cases. Similarly, Table 1c shows that _R_ 1 = 0 _._ 3 guarantees full capture of post-inference power decay but often extends beyond the actual spike, while _R_ 1 = 0 _._ 2 provides a better balance, achieving 88% completeness with reduced baseline inclusion. We therefore set _R_ 0 = 0 _._ 0 and _R_ 1 = 0 _._ 2 for all experiments, balancing measurement accuracy and completeness. This choice avoids systematic overestimation while tolerating a small, controlled loss of data. 

## **4 EXPERIMENTS** 

We analyze the energy consumption dataset to address three Research Questions (RQ)s: 

- **_RQ1. How does the energy consumption of LLM inference vary across different hardware, models, and prompt datasets?_** 

- **_RQ2. What is the relationship between prompt complexity, response characteristics, and the energy consumption of LLMs during the inference process?_** 

- **_RQ3. Can we develop a predictive model accurately forecasting the energy consumption of LLMs based on prompt features and response characteristics?_** 

- **_RQ4. To what extent can LLM inference energy consumption be modeled using an interpretable mathematical function of response length, model type, and hardware configuration?_** 

- **_RQ5. How does the variability in energy consumption for the same prompt manifest across multiple executions and different models?_** 

- **_RQ6. What are the differences in energy measurements among MELODI and other tools, and what factors contribute to these variations?_** 

### **4.1 Subjects of the Experiments** 

This section outlines the energy consumption dataset collected with MELODI, using diverse computing machines (from high-end servers to laptops) and multiple prompt datasets for several LLMs. Table 2 lists the prompt datasets in dataset collection, i.e., Alpaca [36] and Code-Feedback [37], and their average energy consumption and response token length. The Alpaca dataset, produced using OpenAI’s text-davinci-003 engine, contains 52,000 prompts designed to enhance instruction-following capabilities in language models. The dataset primarily includes text generation tasks and is tailored to train pretrained language models to follow complex instructions more effectively. The Code-Feedback dataset supports the OpenCodeInterpreter model [47] designed to refine code by integrating code execution and human feedback. This dataset features 68,000 multi-turn interactions, combining user instructions with compiler responses to enhance model training in coding scenarios. 

TABLE 2: Prompt datasets. 

|Prompt<br>dataset|Average energy<br>consumption(kWh)|Average response<br>token length|
|---|---|---|
|Alpaca|4.49e-04|213.98|
|Code-feedback|5.68e-04|405.96|



TABLE 3: LLMs in the experiments. 

|Model name/size|Average energy<br>consumption(kWh)<br>Average<br>token|response<br>length|
|---|---|---|
|codellama-7b|2.35e-04|535.03|
|codellama-70b|3.33e-03|342.64|
|gemma-2b|8.18e-05|279.11|
|gemma-7b|1.36e-04|249.33|
|llama3-8b|1.34e-04|255.20|
|llama3-70b|2.26e-03|251.46|
|phi-2b|3.14e-05|179.30|
|qwen2-7b|8.37e-05|199.34|
|qwen2-72b|2.25e-03|210.82|



Table 3 lists the LLMs with their average energy consumption and response token lengths. The models range from codelama-7b to llama3-70b, with energy consumption varying from as low as 8 _._ 20 _×_ 10<sup>_−_5</sup> kWh for gemma-2b to 3 _._ 33 _×_ 10<sup>_−_3</sup> kWh for codellama-70b. The average response token lengths vary, with codelama-7b generating the longest responses at 535.03 tokens, while phi-2b records shorter lengths at 179.30 tokens. 

Table 4 details the hardware used in our data collection. The diversity of the hardware covers a wide range of processing power, memory capacities, and graphics capabilities, crucial for a comprehensive analysis of energy consumption across different computing environments. 



5 

TABLE 4: Hardware used in the data collection. 

|Machine|CPU|Memory|GPU|GPU memory|
|---|---|---|---|---|
|Server|AMD EPYC 7643 48-Core Processor|528GB|NVIDIA RTX A5000|24GB|
|Workstation|Intel Xeon W-2223 8-core @ 3.6GHz|128GB|NVIDIA RTX A2000|12GB|
|Laptop 1|Intel i5 11th Gen 12-core @ 2.4GHz|16GB|None|None|
|Laptop2|Intel i7 10th Gen 12-core @ 2.7GHz|32GB|NVIDIAQuadro RTX 4000|8GB|



TABLE 5: Overview of the energy consumption dataset. 

|ID|Prompt dataset|Model|Model size|Hardware|No. of prompts|Average energy consumption<br>per response(kWh)|Average response<br>token length|
|---|---|---|---|---|---|---|---|
|1|code-feedback|codellama|7b|laptop1|5347|1.85e-04|403.63|
|2|code-feedback|codellama|7b|laptop2|3572|2.46e-04|518.40|
|3|code-feedback|codellama|7b|workstation|1000|2.73e-04|685.06|
|4|code-feedback|codellama|70b|workstation|1000|3.33e-03|342.64|
|5|alpaca|gemma|2b|laptop1|5347|1.85e-04|401.64|
|6|alpaca|gemma|2b|laptop2|5101|4.70e-05|181.52|
|7|alpaca|gemma|2b|workstation|1000|3.84e-05|184.30|
|8|code-feedback|gemma|2b|laptop2|5008|7.33e-05|303.98|
|9|code-feedback|gemma|2b|workstation|1000|6.60e-05|324.14|
|10|alpaca|gemma|7b|laptop2|5099|9.81e-05|160.60|
|11|alpaca|gemma|7b|workstation|1000|8.05e-05|164.90|
|12|code-feedback|gemma|7b|laptop2|3405|2.00e-04|332.62|
|13|code-feedback|gemma|7b|workstation|1000|1.65e-04|339.21|
|14|alpaca|llama3|8b|laptop2|5101|1.34e-04|255.20|
|15|alpaca|llama3|70b|server|1026|2.26e-03|251.46|
|16|alpaca|phi|2b|laptop2|674|3.22e-05|221.63|
|17|alpaca|phi|2b|workstation|1000|3.15e-05|136.98|
|18|alpaca|qwen2|7b|laptop2|821|1.22e-04|300.70|
|19|alpaca|qwen2|7b|workstation|1000|1.11e-04|97.99|
|20|alpaca|qwen2|72b|workstation|1000|2.25e-03|210.82|



Table 2 summarizes the energy dataset collected by MELODI across multiple hardware setups and LLMs under diverse operational conditions. Each sample includes the prompt and response, prompt/response token counts, API timestamp, time-series power traces for the LLM process and Scaphandre during monitoring, and the aggregated inference energy consumption. 

### **4.2 Experiments Setup** 

**_Data Preparation._** Our raw data consists of power usage over time, which we aggregate to calculate the energy consumption for each inference process as outlined in Section 3. These aggregated values are compiled into distinct energy consumption datasets, categorized by the specific hardware, model, and prompt dataset utilized during the inference. 

**_Statistical Analysis._** Our analysis of the energy consumption datasets focuses on two metrics: energy per token and energy per response. Energy consumption per token is crucial as it evaluates resource use across different inference setups regardless of text volume. Meanwhile, energy per response is beneficial for estimating resource usage relative to the input prompt. We conduct statistical analyses on these metrics to compare across model types, sizes, hardware setups, and the specific prompt datasets used during inference, providing a comprehensive view of energy dynamics. 

**_Visualization._** We employ box plots to visualize the distribution of energy consumption for each model and hardware. The box plots depict the median energy consumption as a red line, the interquartile range (IQR) with the main box, and extend to 1.5 times the IQR with whiskers, providing a clear summary of variability and central tendency. 

**_Interpretation._** We interpret results through a comparative analysis of energy consumption across different scenarios. The relationship between prompt complexity and energy consumption is assessed using the Pearson correlation coefficient, which ranges from 0 to 1 to indicate variable correlation. Additionally, predictive models for energy consumption are evaluated using the _R_<sup>2</sup> score. 

### **4.3 Results** 

### _4.3.1 RQ1 (energy consumption of LLM inference across different setups)_ 

To address RQ1, we performed a comparative analysis of energy consumption across various LLMs operating on different hardware (see Figures 3 and 4). 

**_Model Size._** Figures 3a and 4 categorize energy consumption per token by the model utilized for inference. Notably, the largest models (codellama-70b with Code-Feedback, qwen2-72b and llama3-70b with Alpaca) exhibit energy usage approximately 100 times greater than their smallest counterparts (codellama-7b with Code-Feedback, qwen2-7b and llama3-8b with Alpaca). Additionally, the larger models (gemma-7b with both Alpaca and Code-Feedback) show an energy consumption that is roughly ten times higher than their smaller versions (gemma-2b with Alpaca and Code-Feedback). 

**_Hardware._** Figure 3b organizes results by the hardware. Energy consumption is noticeably higher for the laptops than the workstation. Laptop 1 (CPU-only) shows similar energy usage when operating different models (codellama-7b with Code-Feedback and gemma-2b with Alpaca), suggesting inefficiencies in CPU-based processing for LLM tasks since we would expect a significant difference when running a 2b model vs a 7b model. Comparisons between the workstation and Laptop 2, despite the former’s more robust GPU (12GB vs. 8GB), reveal roughly equivalent energy usage, underscoring potential disparities in hardware efficiency. 

**_Model._** In energy assessments of different models of the same size (codellama-7b with Code-Feedback vs. gemma-7b with Code-Feedback) run on the workstation and Laptop 2, gemma-7b displayed higher energy consumption than codellama-7b. This observation suggests that codellama-7b is a slightly more energy-efficient model than gemma-7b. phi-2b shows comparable energy consumption per token to the gemma-2b of similar size but has a slightly higher consumption on the workstation. 



6 

![Figure](assets/figure_0002_page_0006.svg)(b) Comparison of energy consumption per token across hardware setups (Codellama-70b is excluded for the workstation to not skew the plot). 

Fig. 3: Comparative energy consumption per token across inference setups. 

**_Prompt Datasets._** To evaluate the energy consumption per output token across two prompt datasets, we analyzed the box plots for gemma-2b and gemma-7b in Figure 3a, the only models tested with both prompt datasets. The analysis reveals that, except in one instance, the Alpaca dataset consistently led to higher energy consumption per token. The IQR, representing variability in energy consumption, is notably wider in all cases, suggesting greater fluctuation in energy use when using the Alpaca dataset. 

Table 2 facilitates a comparison of energy consumption per prompt, revealing that the median consumption for the Code-Feedback prompt dataset exceeds that of the Alpaca dataset during inference. This discrepancy likely arises from longer response length of Code-Feedback prompts. The response length of Code-Feedback prompts averages at 405.96 tokens whereas the response length of Alpaca prompts averages at 213.98 tokens. 

**RQ1 Conclusion.** Larger models like codellama-70b and llama3-70b use roughly 100 times more energy per token than smaller ones. Energy demands are higher for LLMs running on laptops than workstations, particularly due to inefficiencies in CPU processing. Models of the same size exhibit varying energy efficiencies. The Code-Feedback dataset leads to greater energy use per token than Alpaca, likely due to longer response lengths. 

### _4.3.2 RQ2 (relationship between prompt complexity, response characteristics and the energy consumption)_ 

To respond to RQ2, we used Python libraries, namely, Spacy [48], nltk [49], textblob [50], and textstat [51], to derive 55 text-based features. We integrated the features 

![Figure](assets/figure_0003_page_0006.svg)Fig. 4: Comparison of energy consumption per token across models with 70-72 billion parameters. 

with our energy consumption dataset and calculated the Pearson correlation coefficient between energy consumption per prompt and each feature. 

We extracted prompt features spanning multiple categories, including lexical richness and word-level complexity (e.g., average word length, long/polysyllabic word counts, lexical diversity), part-of-speech frequencies (e.g., noun and adjective counts), readability indices (e.g., Flesch, SMOG, Coleman–Liau, Dale–Chall), sentence-level structure (e.g., mean sentence length, punctuation counts), and sentiment scores. Table 6 reports the top 20 correlated features. 

Our analysis revealed that significant correlations with energy consumption per prompt are primarily linked to response characteristics rather than the prompt complexity. Key factors such as response duration, total duration, and response token length show strong positive correlations with energy consumption (with coefficients of 0.770, 0.766, and 0.755, respectively). This observation suggests that energy consumption escalates with an increase in response 



7 

TABLE 6: Correlations between text features and energy consumption per response across all samples. Features are derived from the prompt unless marked with (*), which indicates response-based features. 

|Feature|Correlation|Feature|Correlation|
|---|---|---|---|
|*response<br>duration|0.770|reading<br>time|0.076|
|*total<br>duration|0.766|lexicon<br>count|0.075|
|*response<br>token<br>length|0.755|adverb<br>count|0.073|
|prompt<br>duration|0.129|prompt<br>token<br>length|0.073|
|adj<br>count|0.101|stop<br>word<br>count|0.070|
|polysyllabcount|0.087|noun<br>count|0.070|
|long<br>word<br>count|0.085|verb<br>count|0.068|
|syllable<br>count|0.079|monosyllabcount|0.067|
|letter<br>count|0.079|word<br>count|0.065|
|char<br>count|0.076|sentence<br>count|0.058|



tokens due to more extensive processing within the model, highlighting that longer and more time-intensive responses drive higher energy usage. 

We observe only low positive correlations with prompt complexity features. Attributes like prompt duration, adjective count, and syllable count demonstrate modest correlations with energy consumption, with coefficients of 0.129, 0.101, and 0.087, respectively. These results indicate that prompt complexity (given our set of features) has a minimal impact on energy consumption. Instead, the length and duration of responses are more significant factors. This insight suggests that optimizing the response generation process could be more effective in reducing energy consumption than merely simplifying the input prompts. 

**RQ2 Conclusion.** The results show that response characteristics such as token length and duration are strongly correlated with energy usage, indicating higher consumption with longer responses. Conversely, prompt complexity features like adjective and syllable counts exhibit low correlations, suggesting minimal impact on energy use. Managing response generation may offer more significant energy savings than simplifying prompts. 

_4.3.3 RQ3 (predictive model for LLM energy consumption)_ To address RQ3, we trained ML models on the energy dataset using either prompt features or response characteristics as inputs. We evaluated Random Forest (RF), Gradient Boosting (GB), Linear Regression (LR), XGBoost (XGB), Decision Tree (DT), and Support Vector Machine (SVM), training two models per dataset: one using response token length and one using prompt-derived features. 

Table 6 shows that energy consumption is strongly correlated with response characteristics such as token length and duration, suggesting high potential for accurate prediction. Figure 5 indicates that several ML models can predict energy consumption from response length with high accuracy: the best model achieves R<sup>2</sup> _>_ 0 _._ 78 across all datasets, and R<sup>2</sup> _>_ 0 _._ 95 for all but the bottom four. As summarized in Table 7, LR performs best overall, with an average R<sup>2</sup> = 0 _._ 96 and top performance in 13 of 20 datasets. The lowest scores are observed on Laptop1 (R<sup>2</sup> = 0 _._ 77), indicating possible measurement inaccuracies in this CPU-only setup. 

Our results suggest that limiting response length (e.g., by specifying a maximum output size in the prompt) can substantially reduce inference energy consumption. In contrast, models trained solely on prompt-derived features show limited predictive performance (Figure 7). 

![Figure](assets/figure_0004_page_0007.svg)Fig. 5: R<sup>2</sup> scores across different ML algorithms, using only response length for forecasting energy consumption per response. Negative R<sup>2</sup> scores are clipped at -1.0. 

![Figure](assets/figure_0005_page_0007.svg)Fig. 6: R<sup>2</sup> scores across different ML algorithms, using prompt characteristics for forecasting energy consumption per response. Negative R<sup>2</sup> scores are clipped at -1.0. 

![Figure](assets/figure_0006_page_0007.svg)Fig. 7: R<sup>2</sup> scores of GB models using prompt characteristics for forecasting energy consumption per response. 

Figure 6 reports R<sup>2</sup> scores for models trained solely on prompt-derived features, excluding any response information, to assess whether inference energy can be predicted a priori. Overall performance is low: only four datasets achieve R<sup>2</sup> _>_ 0 _._ 44, and none exceed 0.50. As summarized in Table 7, LR attains the highest average R<sup>2</sup> , while GB performs best in roughly half of the datasets, indicating that predicting inference energy from prompt characteristics alone remains challenging. Notably, predictive accuracy varies substantially across datasets and appears primarily driven by model type: in Figure 7, GB models yield mostly 



8 

TABLE 7: Average R<sup>2</sup> scores and number of wins for various ML algorithms using two feature sets: (i) only response length, and (ii) prompt characteristics. 

|ML alg.|Response|length only|Prompt ch|aracteristics|
|---|---|---|---|---|
||Avg. R<sup>2</sup>|No. of wins|Avg. R<sup>2</sup>|No. of wins|
|DT|0.87|0|-156.14|0|
|GB|0.89|5|-134.72|10|
|LR|0.96|13|-18.34|0|
|RF|0.89|2|-48.01|6|
|SVM|-12392.08|0|-12392.08|0|
|XGB|-1.17|0|-0.47|4|



positive R<sup>2</sup> for Gemma, low negative scores for Phi, and the worst performance for CodeLlama. In contrast, hardware, model size, and prompt dataset show no consistent effect. We hypothesize that model-specific training differences affect the predictability of response length from prompt features, but more controlled experiments are needed. 

**RQ3 Conclusion.** ML models, particularly Linear Regression, predict inference energy accurately when using response length as input, with most datasets achieving R<sup>2</sup> _>_ 0 _._ 95. In contrast, models trained solely on prompt features perform poorly, though results vary widely across datasets. This suggests that some LLM families exhibit a stronger, more predictable relationship between prompt characteristics and response length than others. 

### _4.3.4 RQ4 (mathematical model for energy consumption based on response length, model and hardware)_ 

The consistently high predictive performance of the LR models in RQ3 suggests that inference-time energy consumption can be approximated through a compact and interpretable mathematical formulation. To investigate this in RQ4, we aggregated data from all experiments and trained an LR model with response length, model attributes, and hardware configuration as predictors. We model the energy consumption _E_ (kWh) of a single inference instance as: 



$$
E = β0 + β1 \cdot{}{} ntokens + β2 \cdot{}{} smodel + � i αi \cdot{}{} Itypei+ � j γj \cdot{}{} Ihwj + � i,j δi,j \cdot{}{} (ntokens \times{}{} Icati,j) (1) where:
$$



where:

- _β_ 0 is the base energy cost (intercept), 

- _β_ 1 is the baseline energy per response token, 

- _n_ tokens is the number of generated response tokens, 

- _β_ 2 is the coefficient for model size (parameters), 

- _s_ model is the model size, 

- _αi_ are model type effects encoded as categorical indicators (codellama, gemma, llama3, phi, qwen2) 

- _I_ type _i_ are indicator variables for model types, 

- _• γj_ are hardware effects encoded as categorical indicators (laptop, workstation, server), 

- _I_ hw _j_ are indicator variables for hardware setups, 

- _δi,j_ are interaction coefficients that allow the energyper-token rate to vary across model types and hardware configurations. 

Interaction terms were included because the empirical results indicate that token-level energy scaling differs across model architectures and hardware setups, making a single 

global slope insufficient to capture observed behavior. Numerical predictors include response length (primary driver) and model size, while categorical predictors include 11 models (codellama, codellama-70b, codellama-7b, gemma-2b, gemma-7b, llama3, llama3-70b, phi, qwen2, qwen2-72b, qwen2-7b) and 4 hardware configurations (Laptop 1, Laptop 2, Workstation, Server). We encoded categorical variables with one-hot encoding and removed the first category to avoid multicollinearity. 

**Model Performance.** The fitted LR model achieves very strong performance across all combined datasets, reaching _R_<sup>2</sup> = 0 _._ 9962. This result indicates that inference energy can be predicted accurately using a simple, interpretable formulation grounded in response length, model identity, and deployment context. 

**Estimated Coefficients and Interpretation.** The estimated baseline coefficient for response length was _β_ 1 = 5 _._ 28 _×_ 10<sup>_−_7</sup> kWh/token, capturing the average marginal cost per generated token across all configurations. In contrast, model size exhibited only a negligible direct effect ( _β_ 2 = 3 _._ 96 _×_ 10<sup>_−_18</sup> kWh/billion parameters), suggesting that model size primarily impacts energy indirectly through **model-family and configuration-specific token-level scaling** captured by categorical and interaction terms. 

Model family effects showed substantial variability in intercept shifts. For example: 

- codellama-70b exhibited a strong negative intercept shift ( _−_ 5 _._ 67 _×_ 10<sup>_−_5</sup> kWh - energy offset), 

- qwen2-7b showed the largest positive shift (+3 _._ 98 _×_ 10<sup>_−_5</sup> ), 

- gemma and llama3 showed more moderate shifts in the range of +2 _._ 4 to +3 _._ 2 _×_ 10<sup>_−_5</sup> kWh. 

Hardware-specific intercept terms differed, with **server** having the highest baseline energy, consistent with higher system-level power draw even for comparable workloads. 

   - **Server:** +3 _._ 24 _×_ 10<sup>_−_5</sup> kWh (highest baseline cost, likely due to higher power draw) 

   - **Workstation:** _−_ 1 _._ 42 _×_ 10<sup>_−_6</sup> kWh (slight reduction in baseline cost) 

- **Laptop 2:** +2 _._ 31 _×_ 10<sup>_−_6</sup> kWh (small positive shift) 

- More importantly, token-level scaling varied substan- 

- tially across configurations. For example: 

   - qwen2-72b and codellama-70b showed the steepest energy-per-token scaling, 

- llama3-70b exhibited similarly high scaling, 

- _•_ Smaller models such as gemma-2b and phi-2b showed considerably lower scaling slopes. 

- These findings reinforce that inference energy is not 

- determined solely by parameter count, but also by **architecture-dependent decoding behavior and hardwaredependent execution efficiency** . 

**Ablation Study.** To quantify the relative impact of different feature groups, we performed an ablation study by fitting reduced LR models: 

- Both interactions (full model): _R_<sup>2</sup> = 0 _._ 9962 

- Hardware interactions only: _R_<sup>2</sup> = 0 _._ 8361 

- Model type interactions only: _R_<sup>2</sup> = 0 _._ 9954 

These results indicate that **model type and response length explain most of the variance** , while hardware contributes a smaller but non-negligible portion. The findings 



9 

suggest that, for practical energy-aware deployment, optimization should prioritize **selecting more efficient model families** , and **controlling response length** , while hardware choice plays a secondary role compared to these factors. 

**RQ4 Conclusion.** The mathematical model demonstrates that LLM inference energy consumption is highly predictable (R<sup>2</sup> = 0.9962), with response length being the dominant factor and model type accounting for nearly all categorical variation. Our ablation study reveals that hardware configuration has minimal impact once model type is known, indicating that energy optimization should prioritize model selection and response length control over hardware choices. 

### _4.3.5 RQ5 (variability in energy consumption for the same prompt across multiple executions and different models)_ 

![Figure](assets/figure_0007_page_0009.svg)To address RQ5, we analyzed the energy consumption and response token length for 100 iterations of a prompt ( _Give three tips for staying healthy_ ) from the Alpaca dataset on seven models using laptop 2. We evaluated variability by using the prompt across multiple trials. This ensures uniformity in our analysis, as the chosen prompt is unlikely to influence variations in energy consumption and response length when repeatedly processed. Figure 8 presents differences in energy consumption and response variability among models. 

(a) Variation in energy consumption per prompt for the same prompt run multiple times on seven models. 

(b) Variation in energy consumption per token for the same prompt run multiple times on seven models. 

Figure 8a presents the energy consumption per prompt across seven models, revealing substantial variability. Qwen2-7b, the most expensive per prompt overall, exhibits the highest IQR of energy consumption. Gemma-7b exhibits consistently similar consumption per prompt as the smaller version of the model (gemma-2b). This aligns with its much lower response token length in Figure 8c. Shorter responses generally reduce the total energy for processing. The 

![Figure](assets/figure_0008_page_0009.svg)(c) Variation in response token length for the same prompt run multiple times on seven models. 

Fig. 8: Variability in energy consumption and response token length for 100 iterations of the prompt ’Give three tips for staying healthy’ from Alpaca. 

variability is the smallest in all the models (see its IQR), suggesting similar energy use across repeated inferences. 

Figure 8b displays the energy consumption per token, noting a wide IQR for gemma-2b, which is significantly greater than that of other models. While gemma-7b also shows a higher IQR, it is less pronounced than gemma-2b. The largest models (gemma-7b, llama3-8b, and qwen2-7b) exhibit the highest mean energy consumption per token, as they require more computation. This direct relationship between model size and energy expenditure is consistent with our findings, underscoring that the models with 7B and 8B parameters are more energy-intensive than their counterparts due to their complex inference processes. 

Llama3-8b and qwen2-7b produce the longest responses, averaging 249 and 264 tokens, respectively (see Figure 8c), contributing to their higher energy consumption per prompt. Despite their lengthy outputs, these models display moderate variability in response length, indicating consistent response generation for similar prompts. 

**RQ5 Conclusion.** The results revealed significant differences in energy usage and response length variability among models. Qwen2-7b demonstrated larger fluctuations in energy consumption per prompt than others, suggesting factors such as model architecture and operational conditions may influence energy usage consistency. 

### _4.3.6 RQ6 (comparative analysis of energy measurement tools in LLM deployments)_ 

To address RQ6, we evaluated per-token energy consumption using 467 to 480 prompts from the Alpaca dataset across four tools: MELODI, PyJoules [52], CodeCarbon [25], and EnergyMeter [33]. We selected these tools for comparison due to their diverse methodologies in monitoring and reporting energy consumption in LLM deployments, allowing us to thoroughly assess and benchmark the performance of MELODI against established tools. We run phi2-2b [53] on laptop 2 for the analysis. We opted for phi2-2b primarily for its rapid inference capabilities. Given that our study prioritizes the assessment of measurement tools over model performance, utilizing a fast-processing model ensures uniform results across various tools and minimizes delays that could arise from complex model computations. The model choice is not expected to influence the observed differences between the monitoring tools, as the effects are normalized by averaging over the number of tokens. 

Figure 9 displays considerable variation in per-token energy consumption across the tools. For CPU energy consumption, CodeCarbon recorded a mean of 1 _._ 40 _×_ 10<sup>_−_7</sup> kWh, EnergyMeter logged 8 _._ 13 _×_ 10<sup>_−_8</sup> kWh, and PyJoules reported 1 _._ 46 _×_ 10<sup>_−_7</sup> kWh. Notably, the processlevel measurements of MELODI showed the lowest average CPU energy at 6 _._ 40 _×_ 10<sup>_−_11</sup> kWh. For GPU, CodeCarbon and EnergyMeter were closely aligned with means around 2 _._ 57 _×_ 10<sup>_−_7</sup> kWh and 2 _._ 53 _×_ 10<sup>_−_7</sup> kWh, respectively. MELODI reported a mean GPU energy consumption of 1 _._ 22 _×_ 10<sup>_−_7</sup> kWh, while PyJoules recorded a significantly lower mean GPU energy at 2 _._ 81 _×_ 10<sup>_−_10</sup> kWh, possibly reflecting differences in how idle states or sampling frequencies impact measurements. 



10 

TABLE 8: Comparison of energy measurement tools for LLM deployments. 

![Figure](assets/figure_0009_page_0010.svg)The variability in energy measurements among different tools suggests tool-specific biases or variations in interface handling for CPU and GPU measurements. For CPU, the per-process measurement approach of MELODI yielded significantly lower energy readings, potentially isolating energy consumption more effectively in the mixed-use environments of laptops. In contrast, PyJoules and CodeCarbon, which measure CPU energy at the system level, reported higher averages, likely including additional background activities that influence overall energy assessments. 

The similar GPU energy consumption readings between CodeCarbon and EnergyMeter suggest similar NVMLbased GPU data handling (Table 8). Conversely, MELODI reported a slightly lower mean GPU energy consumption, indicating a more restrictive approach to GPU monitoring and potentially reflecting a more precise monitoring method. Notably, the lower IQR in MELODI suggests more accurate data collection, supported by its higher sampling frequency of 10 Hz compared to the lower frequency of CodeCarbon (Table 8). PyJoules, having dramatically lower readings and outdated maintenance (its last update was in October 2021), raises concerns about its measurement accuracy. The variations in GPU energy measurements across tools highlight the potential impact of how each tool interacts with NVMLbased monitoring on the reliability of GPU data. 

Fig. 9: Comparison of the energy consumption per token, as measured by four different monitoring frameworks. 

on Intel’s RAPL interface for power reasings may also affect data reliability. Variability in hardware power efficiency could further skew results, underscoring the need for more precise measurement tools and standardized testing across different hardware configurations in future research. 

**_External Validity._** The generalizability of our findings is limited by the scope of our inference scenarios, particularly concerning the hardware setups and LLMs tested. Given the sample size, drawing broad conclusions is challenging. The ability to extend our results more widely hinges on whether the hardware and model configurations used accurately reflect the diversity in larger-scale LLM deployments. 

MELODI distinguishes itself by its capability to measure CPU energy consumption at the per-process level, effectively isolating model-specific usage more accurately than whole-system methods. Its precision minimizes the interference from background activities prevalent in multitasking environments, enhancing its reliability. 

## **5 DISCUSSION** 

**Scope of Our Study.** Our study concentrates on CPU and GPU energy consumption for LLM inference, chosen for its measurable impact. While significant, this focus omits other energy contributors in data center environments like networking, memory management, orchestration, and cooling, which can exceed computational energy use. This research addresses only direct CPU and GPU power use, emphasizing the need for broader energy evaluations in future work. 

**RQ6 Conclusion.** The analysis of energy measurement tools for LLM deployments revealed significant differences in per-token energy consumption among MELODI, PyJoules, CodeCarbon, and EnergyMeter. MELODI’s perprocess CPU monitoring yielded more precise and generally lower energy readings compared to whole-system methods, pinpointing model-specific consumption. Moreover, the variability in GPU energy measurements, especially the abnormally low readings from PyJoules, suggests discrepancies that could stem from how the tools interact with NVIDIA’s GPU measurements, potentially affected by sampling frequencies and recording intervals. 

**Response Token Length as an Energy Indicator.** Our results reveal that response token length is a reliable indicator of energy consumption, suggesting that managing response lengths could effectively control energy use. However, the connection between prompt complexity and energy consumption is less definitive. While models based on prompt characteristics show potential, their performance varies, indicating these correlations may not be strong enough for reliable predictions in all contexts. We plan to explore the potential of using LLMs or other NLP models to analyze prompts for energy consumption prediction, balancing the benefits against the potential energy costs of such analyses. 

### **4.4 Threats to validity** 

**_Internal Validity._** The accuracy of the power usage monitoring tools significantly impacts the internal validity of our study. The nvidia-smi tool, according to NVIDIA documentation [54], has an accuracy of _±_ 5 Watts, though Yang et al. suggest this error margin could be as high as _±_ 5%. Ensuring no concurrent GPU processes is crucial for accurate readings, and while most experiments isolated the GPU, residual base processes had minimal impact. Scaphandre’s reliance 

**Exploring Response Quality and Energy Use.** Our current analysis does not address the quality of responses generated. Our dataset, however, provides a rich foundation to explore the relationship between response quality and energy consumption. Understanding whether higher-quality 



11 

responses inherently require more energy could inform the development of more efficient models and prompt designs. 

**Enhancing Generalizability and Model Comparisons.** To enhance our study’s generalizability, we plan to test a wider array of hardware configurations and model types, allowing us to better understand energy consumption variations. Comparing the energy efficiency of task-specific models against general-purpose models in different applications would also be valuable. This analysis could reveal if specialized models yield significant energy savings for specific tasks, ensuring that comparisons maintain a consistent base model for accuracy. 

**Investigating Energy Consumption Patterns.** Future research could explore patterns in energy consumption across models, focusing beyond mere size to include architectural differences. This could reveal how specific design choices impact energy demands and help optimize models for better energy efficiency without sacrificing performance. 

## **6 RELATED WORK** 

### **6.1 Energy and Carbon Footprint of AI and LLMs** 

The advancement of deep neural networks, particularly in natural language processing (NLP), has intensified concerns about their environmental impact. Strubell et al. [23] quantified the energy cost of training large NLP models, showing that a single training run can emit as much CO2 as multiple passenger vehicles over their lifetimes. Subsequent studies expanded this perspective: Luccioni et al. [55] analyzed inference-time energy consumption for large transformer models such as BLOOM, while Luccioni et al. [56] demonstrated that general-purpose generative models incur orders-of-magnitude higher energy costs than task-specific alternatives, even when controlling for model size. 

Recent work has shifted attention from training to the operational footprint of LLMs. De Vries [57] highlights the substantial contribution of AI-driven data center operations to global energy demand. Similarly, Jegham et al. [58] examined energy, water, and carbon metrics across state-of-the-art LLM inference deployments, and Poddar et al. [59] identify strong correlations between inference energy, output length, model size, and hardware configuration. Meanwhile, Google Cloud’s analysis of their Gemini model estimate median prompt energy usage at 0.24 W·h and CO2 emission at 0.03 g, underscoring how inference, though individually modest, scales massively [60]. 

Collectively, this body of work underscores a critical insight: while training remains energy-intensive, **inference increasingly dominates the lifecycle energy and carbon footprint of LLMs** due to continuous, large-scale deployment. Despite this shift, most prior studies still emphasize training or rely on coarse-grained measurements, motivating the need for precise, deployment-aware methodologies to analyze inference-time energy consumption and support sustainable AI engineering. 

### **6.2 Energy Monitoring Frameworks for LLMs** 

Prior work has proposed several frameworks for monitoring and analyzing the energy use and carbon footprint of LLMs, revealing substantial variation across model configurations [21], [27]–[29], [33]. Wilkins et al. [28] introduce an 

offline, workload-based framework for LLM inference on heterogeneous CPU–GPU systems, modeling energy consumption and runtime as functions of input and output tokens. However, their reliance on AMD _µ_ Prof entails notable limitations, including measurement overhead from frequent polling, limited real-time granularity, and hardware specificity to AMD CPUs, which restricts generalizability and increases setup complexity. In contrast, Samsi et al. [27] focus exclusively on GPU energy consumption for LLaMA models, omitting CPU contributions. 

Our work complements these efforts above by providing a more comprehensive framework that measures energy consumption across multiple LLMs, using Scaphandre and nvidia-smi for real-time tracking of both CPU and GPU energy usage. This broader approach enables a more holistic understanding of energy dynamics during LLM inference. 

Argerich et al. [33] present a related framework based on pyRAPL [61], nvidia-smi [54], and eBPF [62] to estimate CPU, memory, GPU, and storage energy on Linux, supported by extensive analyses across many LLMs and model sizes. In contrast, MELODI supports a broader range of hardware configurations beyond a single bare-metal environment and extends energy monitoring with predictive analysis, providing a more flexible and comprehensive framework for studying LLM inference energy consumption. Henderson et al. [63] utilize RAPL in a similar manner to Scaphandre to monitor energy use but derive per-process energy consumption based on the CPU’s relative utilization. Scaphandre, however, provides a more comprehensive explanation of how it calculates per-process energy metrics. 

### **6.3 Inference-Time Energy Profiling and Optimization** 

The growing deployment of LLMs in production highlights the importance of inference-time energy profiling and optimization. Samsi et al. [27] analyze inference energy costs for different LLaMA model sizes on NVIDIA V100 and A100 GPUs, demonstrating how model scale, hardware, and task characteristics influence energy consumption. Similarly, Fernandez et al. [64] show that inference optimizations—such as quantization, batching strategies, and hardware-aware mapping—can reduce energy usage by up to 73% compared to unoptimized deployments. Wilhelm et al. [65] introduce the energy-per-token metric and demonstrate strong correlations between response length, model size, and energy consumption, while also exploring dynamic reasoning strategies to reduce energy without significant accuracy loss. Stojkovic et al. [66] introduce a dynamic inference-cluster management framework that exploits workload heterogeneity to optimize inference energy, achieving average energy savings of 52% under service-level objectives. 

Despite these advances, most existing approaches lack fine-grained attribution of energy usage to specific processes (e.g., model decoding) or integrations across diverse hardware and prompt workloads. This gap motivates frameworks capable of capturing per-token, per-process, and hardware-aware energy profiles for real-world LLM deployment scenarios. 

### **6.4 Hardware-Level Power Monitoring Techniques** 

Hardware-level power monitoring is fundamental for analyzing and optimizing energy consumption in computing 



12 

systems. Intel’s RAPL (Running Average Power Limit) interface is widely used, providing on-chip energy counters for CPU and DRAM domains. Prior studies show that RAPL measurements strongly correlate with external power meters, although limitations remain due to driver support, update timing, and domain-specific inaccuracies [67]. In particular, RAPL’s memory-domain estimates can overestimate power consumption and exhibit temporal inconsistencies on heterogeneous memory systems [68], highlighting the need for careful interpretation of such measurements. 

On the GPU side, power monitoring commonly uses NVML (NVIDIA Management Library) and the nvidia-smi interface. You et al. [69] show that GPU power limits strongly affect the energy–performance trade-off of DNN workloads. Another study by Yang et al. [70] reveals that nvidia-smi often samples only a portion of runtime, leading to substantial under- or over-estimation of energy use—sometimes by as much as 35–65%. Furthermore, Arafa et al. [71] demonstrate how hardware counters and instrumentation can measure energy consumption at the instruction level on NVIDIA GPUs, reinforcing that architecturespecific factors (e.g., micro-architecture generation, instruction type) influence energy behavior significantly. 

In summary, hardware-level measurement techniques provide essential visibility into component-level power usage (CPU, DRAM, GPU), but each comes with its own limitations—granularity, accuracy, sampling frequency, and applicability vary. For energy-aware systems especially in the context of LLM deployment, selecting and validating these techniques is critical before interpreting results. 

### **6.5 Sustainable AI Engineering and Future Directions** 

Recent work on sustainable AI increasingly emphasizes energy- and resource-aware design, deployment, and operation of large models. A comprehensive survey on Green AI [72] highlights the need to account for lifecycle impacts, including carbon and water usage, when evaluating AI systems. Another study on LLM sustainability [73] extend this perspective by considering economic and resource costs alongside computation, advocating greater transparency and standardized benchmarking in AI energy and carbon accounting. Fernandez et al. [64] show that inference-level optimizations can reduce energy consumption by up to 73% with negligible impact on accuracy. Similarly, S´anchezMomp´o et al. [74] empirically demonstrate that larger models do not necessarily incur higher energy per request, particularly under low serving utilization. 

Collectively, these studies underscore that sustainable AI engineering demands a holistic perspective encompassing model design, hardware selection, inference architecture, measurement transparency, and deployment practices. They point to key directions such as standardized efficiency metrics, energy-aware architectures, cross-layer co-design, and system-level optimizations. MELODI contributes to this agenda by providing fine-grained inference-time energy measurements supporting energy-conscious AI systems. 

### **6.6 Research Gaps and Motivation for Our Framework** 

Although significant advances have been made in quantifying the environmental footprint of artificial intelligence 

systems, several research gaps remain—especially regarding **fine-grained, inference-time energy measurement of LLMs** . Current literature and tools leave important methodological, technical, and analytical challenges unresolved, motivating the need for a more comprehensive framework. 

**(1) Lack of inference-level granularity.** Most prior research emphasizes training-phase emissions and provides only coarse-grained reporting for inference. Existing tools such as **CodeCarbon** , **PyJoules** , and **EnergyMeter** measure energy at the system level, aggregating consumption across all active processes. This approach obscures the specific energy cost of the LLM process, especially in multi-tasking environments, thereby limiting the interpretability of results and their use for optimization. 

**(2) Hardware-dependent and non-standardized monitoring.** Many tools rely on vendor-specific interfaces (e.g., Intel’s RAPL for CPUs and NVIDIA’s NVML for GPUs) whose accuracy, update frequency, and calibration differ across architectures. These inconsistencies introduce measurement bias and hinder the reproducibility and comparability of studies. Moreover, most frameworks do not jointly monitor CPU and GPU energy, overlooking crosscomponent dependencies shaping overall power dynamics. 

**(3) Limited linkage between operational factors and energy behavior.** Empirical analyses rarely explore how prompt features (e.g., length, complexity), model configurations (e.g., size, architecture), or deployment hardware collectively shape energy consumption. This lack of finegrained correlation hinders the development of targeted optimization strategies for sustainable inference. 

**(4) Absence of unified, open, and process-level measurement solutions.** While recent works call for transparency and standardized reporting of energy metrics, there is no openly available framework that provides processlevel monitoring, configurable sampling, and reproducible data collection across heterogeneous hardware setups. 

**Rationale.** To address these challenges, we propose an extensible framework for high-resolution monitoring and analysis of LLM inference energy consumption. MELODI is designed to (i) capture CPU and GPU energy at the process level with configurable sampling intervals; (ii) log inference metadata, including token counts and timestamps; and (iii) enable empirical studies linking energy use to model, prompt, and hardware characteristics. By bridging measurement accuracy, analytical depth, and reproducibility, our framework advances the methodological foundations of sustainable AI engineering, enabling more transparent and data-driven optimization of LLM inference efficiency. 

## **7 CONCLUSION** 

In conclusion, MELODI represents a noteworthy leap forward in advancing energy-aware practices within the domain of LLMs. By formulating a precise methodology and unveiling an open-source instrument tailored for real-time monitoring of energy consumption throughout the LLM inference process, MELODI effectively bridges a vital chasm in sustainable computing. Our data collection, encompassing various LLMs, hardware setups, and prompt datasets, provides a robust platform for rigorous comparison and nuanced analysis. This facilitates a deeper understanding 



13 

of the energy dynamics associated with LLM deployment scenarios, offering critical insights that contribute to the sustainability of LLMs. 

## **ACKNOWLEDGMENT** 

The work has been conducted as part of the ENFIELD project (101120657) funded by the European Commission within the HEU Programme. 

## **REFERENCES** 

- [1] A. Vaswani, “Attention is all you need,” _Advances in Neural Information Processing Systems_ , 2017. 

- [2] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill _et al._ , “On the opportunities and risks of foundation models,” _arXiv preprint arXiv:2108.07258_ , 2021. 

- [3] T. B. Brown, “Language models are few-shot learners,” _arXiv preprint arXiv:2005.14165_ , 2020. 

- [4] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, “On the dangers of stochastic parrots: Can language models be too big?” in _ACM FAccT’21_ , 2021, pp. 610–623. 

- [5] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever _et al._ , “Language models are unsupervised multitask learners,” _OpenAI blog_ , vol. 1, no. 8, p. 9, 2019. 

- [6] S. Sudhakaran, M. Gonz´alez-Duque, M. Freiberger, C. Glanois, E. Najarro, and S. Risi, “Mariogpt: Open-ended text2level generation through large language models,” _Advances in Neural Information Processing Systems_ , vol. 36, 2024. 

- [7] X. Liang, H. Wang, Y. Wang, S. Song, J. Yang, S. Niu, J. Hu, D. Liu, S. Yao, F. Xiong _et al._ , “Controllable text generation for large language models: A survey,” _arXiv preprint arXiv:2408.12599_ , 2024. 

- [8] J. Li, T. Tang, W. X. Zhao, J.-Y. Nie, and J.-R. Wen, “Pre-trained language models for text generation: A survey,” _ACM Computing Surveys_ , vol. 56, no. 9, pp. 1–39, 2024. 

- [9] J. Wulf and J. Meierhofer, “Exploring the potential of large language models for automation in technical customer service,” _arXiv preprint arXiv:2405.09161_ , 2024. 

- [10] S. Kolasani, “Optimizing natural language processing, large language models (llms) for efficient customer service, and hyperpersonalization to enable sustainable growth and revenue,” _Transactions on Latest Trends in Artificial Intelligence_ , vol. 4, no. 4, 2023. 

- [11] J. Chen, Z. Liu, X. Huang, C. Wu, Q. Liu, G. Jiang, Y. Pu, Y. Lei, X. Chen, X. Wang _et al._ , “When large language models meet personalization: Perspectives of challenges and opportunities,” _World Wide Web_ , vol. 27, no. 4, p. 42, 2024. 

- [12] J. Li, H. Zhou, S. Huang, S. Cheng, and J. Chen, “Eliciting the translation ability of large language models via multilingual finetuning with translation instructions,” _Transactions of the Association for Computational Linguistics_ , vol. 12, pp. 576–592, 2024. 

- [13] B. Zhang, B. Haddow, and A. Birch, “Prompting large language model for machine translation: A case study,” in _International Conference on Machine Learning_ . PMLR, 2023, pp. 41 092–41 110. 

- [14] L. Wang, C. Lyu, T. Ji, Z. Zhang, D. Yu, S. Shi, and Z. Tu, “Document-level machine translation with large language models,” _arXiv preprint arXiv:2304.02210_ , 2023. 

- [15] Q. Zhong, L. Ding, J. Liu, B. Du, and D. Tao, “Can chatgpt understand too? a comparative study on chatgpt and fine-tuned bert,” _arXiv preprint arXiv:2302.10198_ , 2023. 

- [16] X. Zhu, S. Gardiner, T. Rold´an, and D. Rossouw, “The model arena for cross-lingual sentiment analysis: A comparative study in the era of large language models,” _arXiv preprint arXiv:2406.19358_ , 2024. 

- [17] W. Zhang, Y. Deng, B. Liu, S. J. Pan, and L. Bing, “Sentiment analysis in the era of large language models: A reality check,” _arXiv preprint arXiv:2305.15005_ , 2023. 

- [18] M. C. Rillig, M. Agerstrand, M. Bi, K. A. Gould, and U. Sauerland,<sup>˚</sup> “Risks and benefits of large language models for the environment,” _Environmental Science & Technology_ , vol. 57, no. 9, pp. 3464– 3466, 2023. 

- [19] L. Weidinger, J. Uesato, M. Rauh, C. Griffin, P.-S. Huang, J. Mellor, A. Glaese, M. Cheng, B. Balle, A. Kasirzadeh _et al._ , “Taxonomy of risks posed by language models,” in _ACM FAccT’22_ , 2022, pp. 214–229. 

- [20] L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang, M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh _et al._ , “Ethical and social risks of harm from language models,” _arXiv preprint arXiv:2112.04359_ , 2021. 

- [21] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar _et al._ , “Holistic evaluation of language models,” _arXiv preprint arXiv:2211.09110_ , 2022. 

- [22] E. J. Husom, A. Goknil, M. Astekin, L. K. Shar, A. K A¥<sup>˜</sup> sen, S. Sen, B. A. Mithassel, and A. Soylu, “Sustainable llm inference for edge ai: Evaluating quantized llms for energy efficiency, output accuracy, and inference latency,” _ACM Transactions on Internet of Things_ , vol. 6, no. 4, pp. 1–35, 2025. 

- [23] E. Strubell, A. Ganesh, and A. McCallum, “Energy and policy considerations for deep learning in nlp,” _arXiv preprint arXiv:1906.02243_ , 2019. 

- [24] L. Lannelongue, J. Grealey, and M. Inouye, “Green algorithms: quantifying the carbon footprint of computation,” _Advanced science_ , vol. 8, no. 12, p. 2100707, 2021. 

- [25] CodeCarbon, https://codecarbon.io/, Visited in 2024. 

- [26] E. J. Husom, S. Sen, and A. Goknil, “Engineering carbon emissionaware machine learning pipelines,” in _CAIN’24_ , 2024, pp. 118–128. 

- [27] S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones, W. Bergeron, J. Kepner, D. Tiwari, and V. Gadepally, “From words to watts: Benchmarking the energy costs of large language model inference,” in _IEEE HPEC’23_ , 2023, pp. 1–9. 

- [28] G. Wilkins, S. Keshav, and R. Mortier, “Offline energy-optimal llm serving: Workload-based energy models for llm inference on heterogeneous systems,” _arXiv preprint arXiv:2407.04014_ , 2024. 

- [29] ——, “Hybrid heterogeneous clusters can lower the energy consumption of llm inference workloads,” in _ACM e-Energy’24_ , 2024, pp. 506–513. 

- [30] A. A. Chien, L. Lin, H. Nguyen, V. Rao, T. Sharma, and R. Wijayawardana, “Reducing the carbon impact of generative ai inference (today and in 2035),” in _HotCarbon’23_ , 2023, pp. 1–7. 

- [31] J. Stojkovic, E. Choukse, C. Zhang, I. Goiri, and J. Torrellas, “Towards greener llms: Bringing energy-efficiency to the forefront of llm inference,” _arXiv preprint arXiv:2403.20306_ , 2024. 

- [32] B. Everman, T. Villwock, D. Chen, N. Soto, O. Zhang, and Z. Zong, “Evaluating the carbon impact of large language models at the inference stage,” in _IEEE IPCCC’23_ , 2023, pp. 150–157. 

- [33] M. F. Argerich and M. Pati˜no-Mart´ınez, “Measuring and improving the energy efficiency of large language models inference,” _IEEE Access_ , vol. 12, pp. 80 194–80 207, 2024. 

- [34] Scaphandre, https://github.com/hubblo-org/scaphandre. [35] NVIDIA, https://developer.nvidia.com/nvidia-systemmanagement-interface, Visited in 2024. 

- [36] Alpaca, https://huggingface.co/datasets/tatsu-lab/alpaca. [37] Code-Feedback, https://huggingface.co/datasets/m-a-p/CodeFeedback, Visited in 2026. 

- [38] NVIDIA, https://developer.nvidia.com/management-librarynvml, Visited in 2024. 

- [39] M. Jay, V. Ostapenco, L. Lef`evre, D. Trystram, A.-C. Orgerie, and B. Fichel, “An experimental comparison of software-based power meters: focus on cpu and gpu,” in _IEEE/ACM CCGRID’23_ , 2023, pp. 106–118. 

- [40] Prometheus, https://prometheus.io/docs/instrumenting/exporters/. 

- [41] Llama.cpp, https://github.com/ggerganov/llama.cpp. 

- [42] Y. Anand, Z. Nussbaum, A. Treat, A. Miller, R. Guo, B. Schmidt, G. Community, B. Duderstadt, and A. Mulyar, “Gpt4all: An ecosystem of open source compressed language models,” _arXiv preprint arXiv:2311.04931_ , 2023. 

- [43] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in _SOSP’23_ , 2023, pp. 611–626. 

- [44] Llamafile, https://github.com/Mozilla-Ocho/llamafile. 

- [45] Ollama, https://github.com/ollama/ollama, Visited in 2026. 

- [46] Z. Yang, K. Adamek, and W. Armour, “Part-time power measurements: nvidia-smi’s lack of attention,” 2024. 

- [47] T. Zheng, G. Zhang, T. Shen, X. Liu, B. Y. Lin, J. Fu, W. Chen, and X. Yue, “Opencodeinterpreter: Integrating code generation with execution and refinement,” _https://arxiv.org/abs/2402.14658_ , 2024. 

- [48] M. Honnibal and I. Montani, “spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing,” 2017. 



14 

- [49] S. Bird, E. Klein, and E. Loper, _Natural language processing with Python: analyzing text with the natural language toolkit_ . ” O’Reilly Media, Inc.”, 2009. 

- [50] S. Loria, “textblob documentation,” _Release 0.15_ , vol. 2, 2018. [51] textstat, https://github.com/textstat/textstat, Visited in 2024. [52] PowerAPI, “Pyjoules: Python-based energy measurement library,” https://github.com/powerapi-ng/pyJoules, Visited in 2024. 

- [53] phi 2, https://huggingface.co/microsoft/phi-2, Visited in 2025. [54] NVIDIA System Management Interface, https://docs.nvidia.com/deploy/nvidia-smi/index.html. 

![Figure](assets/figure_0010_page_0014.svg)**Erik Johannes Husom** is a Research Scientist in the Trustworthy Green IoT Software research group at SINTEF Digital, 0373 Oslo, Norway. His research interests include applied ML, AI engineering, and responsible AI. Husom received his M.Sc. in computational physics from the University of Oslo. 

- [55] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, “Estimating the carbon footprint of bloom, a 176b parameter language model,” _Journal of machine learning research_ , vol. 24, no. 253, pp. 1–15, 2023. 

- [56] S. Luccioni, Y. Jernite, and E. Strubell, “Power hungry processing: Watts driving the cost of ai deployment?” in _ACM FAccT’24_ , 2024, pp. 85–99. 

- [57] A. De Vries, “The growing energy footprint of artificial intelligence,” _Joule_ , vol. 7, no. 10, pp. 2191–2194, 2023. 

- [58] N. Jegham, M. Abdelatti, L. Elmoubarki, and A. Hendawi, “How hungry is ai? benchmarking energy, water, and carbon footprint of llm inference,” _arXiv preprint arXiv:2505.09598_ , 2025. 

**Arda Goknil** received his Ph.D. degree in computer science from the University of Twente, in the Netherlands, in 2011. He is a senior research scientist at SINTEF, Norway. He was a research associate at the University of Luxembourg. His research concerns AI Engineering, Sustainable AI, Software Testing, Software Security, and Intermittent Computing. He is active on EU-funded and national research projects with several academic and industry partners. 

- [59] S. Poddar, P. Koley, J. Misra, S. Podder, N. Ganguly, and S. Ghosh, “Towards sustainable nlp: Insights from benchmarking inference energy in large language models,” _arXiv preprint arXiv:2502.05610_ , 2025. 

- [60] Google, https://cloud.google.com/blog/products/infrastructure/measuringthe-environmental-impact-of-ai-inference/, Visited in 2025. 

- [61] pyRAPL, https://pypi.org/project/pyRAPL/, Visited in 2026. [62] eBPF, https://ebpf.io/, Visited in 2026. [63] P. Henderson, J. Hu, J. Romoff, E. Brunskill, D. Jurafsky, and J. Pineau, “Towards the systematic reporting of the energy and carbon footprints of machine learning,” _Journal of Machine Learning Research_ , vol. 21, no. 248, pp. 1–43, 2020. 

- [64] J. Fernandez, C. Na, V. Tiwari, Y. Bisk, S. Luccioni, and E. Strubell, “Energy considerations of large language model inference and efficiency optimizations,” _arXiv preprint arXiv:2504.17674_ , 2025. 

- [65] P. Wilhelm, T. Wittkopp, and O. Kao, “Beyond test-time compute strategies: Advocating energy-per-token in llm inference,” in _EuroMLSys’25_ , 2025, pp. 208–215. 

- [66] J. Stojkovic, C. Zhang, I.<sup>´</sup> Goiri, J. Torrellas, and E. Choukse, “Dynamollm: Designing llm inference clusters for performance and energy efficiency,” in _IEEE HPCA’25_ , 2025, pp. 1348–1362. 

- [67] K. N. Khan, M. Hirki, T. Niemi, J. K. Nurminen, and Z. Ou, “Rapl in action: Experiences in using rapl for power measurements,” _ACM Transactions on Modeling and Performance Evaluation of Computing Systems_ , vol. 3, no. 2, pp. 1–26, 2018. 

- [68] L. Alt, A. Kozhokanova, T. Ilsche, C. Terboven, and M. S. Mueller, “An experimental setup to evaluate rapl energy counters for heterogeneous memory,” in _ICPE’24_ , 2024, pp. 71–82. 

- [69] J. You, J.-W. Chung, and M. Chowdhury, “Zeus: Understanding and optimizing _{_ GPU _}_ energy consumption of _{_ DNN _}_ training,” in _NSDI’23_ , 2023, pp. 119–139. 

- [70] Z. Yang, K. Adamek, and W. Armour, “Part-time power measurements: nvidia-smi’s lack of attention,” _arXiv preprint arXiv:2312.02741_ , 2023. 

![Figure](assets/figure_0011_page_0014.svg)and machine learning. 

**Lwin Khin Shar** (Member, IEEE) received a Ph.D. degree in software engineering from Nanyang Technological University (NTU), Singapore, in 2014. He is an Associate Professor at the School of Computing and Information Systems, Singapore Management University, Singapore. He was a Postdoctoral Research Associate with SnT of the University of Luxembourg, Esch-sur-Alzette, Luxembourg, and then a Research Scientist with NTU. His research interests include software engineering, security & privacy, 

- [71] Y. Arafa, A. ElWazir, A. ElKanishy, Y. Aly, A. Elsayed, A.-H. Badawy, G. Chennupati, S. Eidenbenz, and N. Santhi, “Verified instruction-level energy consumption measurement for nvidia gpus,” in _ACM CF’20_ , 2020, pp. 60–70. 

- [72] V. Bol´on-Canedo, L. Mor´an-Fern´andez, B. Cancela, and A. AlonsoBetanzos, “A review of green artificial intelligence: Towards a more sustainable future,” _Neurocomputing_ , vol. 599, p. 128096, 2024. 

- [73] A. Singh, N. P. Patel, A. Ehtesham, S. Kumar, and T. T. Khoei, “A survey of sustainability in large language models: Applications, economics, and challenges,” in _IEEE CCWC’25_ , 2025, pp. 00 008– 00 014. 

- [74] A. S´anchez-Momp´o, I. Mavromatis, P. Li, K. Katsaros, and A. Khan, “Green mlops to green genops: An empirical study of energy consumption in discriminative and generative ai operations,” _Information_ , vol. 16, no. 4, p. 281, 2025. 

![Figure](assets/figure_0012_page_0014.svg)**Sagar Sen** is a senior research scientist at SINTEF Digital, 0373 Oslo, Norway. His research interests are in engineering and testing of lifelong AI systems for application domains such as manufacturing and health. Sen received his Ph.D. in computer science from the University of Rennes 1/INRIA. He is a Member of IEEE. 

