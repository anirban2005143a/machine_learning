_Sustainability via LLM Right-sizing_ 

# **Sustainability via LLM Right-sizing** 

_Date: 04/17/2025_ 

**Jennifer Haase** 

**Finn Klessascheck** 

Weizenbaum Institute and Humboldt University TU Munich and Weizenbaum Institute, Germany Berlin, Germany Heilbronn and Berlin, Germany jennifer.haase@hu-berlin.de finn.klessascheck@tum.de 

**Jan Mendling** 

Weizenbaum Institute and Humboldt University Berlin, Germany jan.mendling@hu-berlin.de 

**Sebastian Pokutta** 

TU Berlin and Zuse Institute Berlin Berlin, Germany pokutta@zib.de 

##### **Abstract** 

Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model “good enough”? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups –premium all-rounders, competent generalists, and limited but safe performers– highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice. 

**_Keywords:_** _Large Language Models, benchmarking, task performance, sustainability_ 

## **Introduction** 

Artificial Intelligence (AI) is becoming an integral component of business and organizational processes (Raisch and Fomina 2024), transforming how decisions are made (del Valle and Lara 2024), services are delivered (Hui et al. 2024), and tasks are automated (Haase et al. 2024). This rapid integration has sparked growing concerns over energy consumption, escalating operational costs, and dependence on external cloud providers (Fioravante 2024; Hogan 2024). As organizations face increasing regulatory pressure and environmental accountability, the question of how to deploy AI systems sustainably is gaining urgency (Tripathi and Kumar 2025; Verdecchia et al. 2023). 

Sustainability in the context of AI not only concerns carbon emissions but also issues of data sovereignty and operational costs. First, the energy demand of large-scale AI models contributes significantly to CO2 emissions, with model size and inference compute being key drivers. Second, many AI solutions rely on centralized cloud infrastructure, raising questions about data privacy, ownership, and data residency. Third, the financial cost of running large models can be prohibitive, especially when usage is scaled across organizational tasks. Here, local models promise significantly reduced cost-per-token, often down to a fraction of a cent. 

**1** 



_Sustainability via LLM Right-sizing_ 

Despite increasing awareness of these challenges, existing AI evaluation measures remain narrowly focused on technical performance metrics such as accuracy, scalability, and generalization (Li et al. 2024b; White et al. 2024). These benchmarks, while important, rarely consider the full spectrum of real-world deployment constraints. Crucially, professionals lack guidance on a core question: When is a smaller, locally deployed model “good enough” to perform a task effectively, affordably, and autonomously? And how should different AI models be selected for various types of tasks, considering trade-offs in sustainability, sovereignty, and cost? 

To address this research problem, we tested 11 LLMs across 10 typical office tasks (like writing an email, summarizing a text, conceptualizing an idea) and evaluated the individual outputs through LLMs. This approach enables the empirical comparison of LLMs, ranging from powerful, cloud-hosted systems to lightweight, local alternatives, across a set of representative occupational tasks. We ask two central research questions: (1) Can smaller, locally deployed models effectively perform occupational tasks? and (2) Which models are most suitable for which types of tasks? Our analysis incorporates core quality aspects of the output generated, its factual integrity, and ethical and social responsibility. We further discuss sustainability metrics—including compute-based CO2 proxies, deployment location (local vs. cloud), and cost per million tokens—to evaluate AI readiness in practical organizational contexts. In doing so, we contribute to the growing discourse in _Green Information Systems_ (Green IS) and propose a structured approach to more sustainable LLM adoption. 

The remainder of this paper first discusses the sustainability aspects from the Green IS perspective, provides a brief analysis of the technical capabilities of different LLMs, describes our LLM-based evaluation method, and presents empirical results on model performance across varied deployment scenarios. We conclude with a discussion of practical implications, limitations, and future directions for sustainable AI integration in organizations. 

## **Background** 

To situate our contribution, this section outlines key debates on sustainable AI deployment in organizations. We build on the triad model of sustainability, discuss technical and economic considerations in LLM tooling, and argue for a shift toward task- and deployment-aware benchmarking practices. Taken together, these strands motivate the need for evaluating AI systems not only in terms of raw capability, but also in terms of operational feasibility and responsible adoption. 

### **_Sustainability and AI Deployment_** 

The Information Systems (IS) community increasingly recognizes sustainability as a critical imperative, particularly through the subfield of Green IS, which seeks to leverage information technology for environmental and social benefit (Kirchner-Krath et al. 2024; Seidel et al. 2017; Watson et al. 2010). Green IS emphasizes practices and processes enabled by digital systems that improve sustainability within organizational and societal contexts (Veit and Thatcher 2023). Within this discourse, two contrasting roles of digital technologies are discussed: On the one hand, technology can exacerbate environmental degradation and social inequality through resource-intensive systems. On the other hand, it is positioned as an enabler of a more sustainable future when applied effectively (Veit and Thatcher 2023). This duality underlines the importance of a clear and multi-dimensional understanding of _sustainability_ . 

To contextualize the broader implications of AI deployment, it is essential to frame the discussion within the established model of sustainability, which provides a structured lens to assess environmental, economic, and social trade-offs. The widely accepted triad model of sustainability comprises three interdependent pillars (Purvis et al. 2019): (1) _environmental_ sustainability refers to preserving ecosystems and reducing emissions; (2) _economic_ sustainability involves long-term viability and cost-efficiency of systems; and (3) _social_ sustainability encompasses equitable access to resources, autonomy, and data control (Klessascheck et al. 2025). The increasing adoption of AI, particularly LLMs, introduces sustainability challenges across all three pillars (Bossert and Loh 2025). On the environmental side, high energy demands and carbon emissions during training and inference contribute to AI’s growing ecological footprint (Cowls et al. 2023; Li et al. 2023; Schütze 2024). Economically, proprietary cloud-hosted models incur substantial operational costs, limiting scalability and excluding smaller organizations (Fioravante 2024; Hogan 2024). Socially, centralized 

**2** 



_Sustainability via LLM Right-sizing_ 

AI infrastructures raise concerns over digital sovereignty, data ownership, and fairness in access to powerful tools (Lazar and Manuali 2024; Sarker et al. 2024). 

Beyond technical performance, the choice between centralized and decentralized LLM deployment models carries significant implications for data governance, transparency, and organizational autonomy. While centralized models offer state-of-the-art performance, they may reinforce power asymmetries by consolidating control over infrastructure and data flows. In contrast, smaller or open-weight models—especially when deployed locally—can foster transparency, autonomy, and control over data. Moreover, from a data governance perspective, locally hosted LLMs can become essential in scenarios where sensitive information must remain under strict organizational control. Avoiding cloud-based inference pipelines helps minimize risks associated with data leakage, external surveillance, or jurisdictional conflicts related to data residency and compliance. 

Despite increasing awareness of these multifaceted concerns, practical guidance for selecting sustainable AI models in organizational settings remains underdeveloped (Kotlarsky et al. 2023). In the following, we translate this conceptual foundation into practical tooling decisions, comparing common LLM deployment options and their implications for sustainable use. 

### **_Sustainability Considerations in LLM Tooling_** 

The trend toward deploying increasingly large, proprietary LLMs hosted on cloud platforms (e.g., GPT-4o and Claude-3 models) has intensified sustainability concerns due to their substantial energy demands, high operational costs, and infrastructure centralization (Leon 2024; Singh et al. 2025). These models typically incur high per-token costs, as illustrated in Table 1, creating significant financial and environmental burdens for organizations seeking broad deployment. Conversely, smaller open-weight models such as Llama-3.3 or Phi-4, which can operate locally on existing organizational hardware, offer substantial economic and environmental advantages, dramatically reducing energy consumption per inference (Chen et al. 2023; Wu et al. 2022). These models also enable greater flexibility for optimizing deployment, e.g., via quantization or model pruning, which further reduces compute cycles and energy use (Verdecchia et al. 2023). Although these strategies incur additional emissions during training, operational savings often outweigh initial costs, especially in contexts that restrict cloud usage due to privacy or legal constraints (Chen et al. 2023). 

From a cost standpoint, token pricing demonstrates stark disparities. As shown in Table 1, input and output costs for proprietary models such as GPT-4.5-preview or Claude-3 can range from $10 to over $150 per million tokens. In contrast, smaller open-weight models like Llama-3.3-70B or Qwen-2.5-Coder are typically priced below $1 per million tokens; we report the cost for Microsoft Azure deployment of these models per https://models.litellm.ai/. Notably, these figures represent hosted deployments; actual costs may be lower when models run locally on in-house infrastructure, particularly in organizations with idle compute resources. The difference in cost structures is not merely economic—it also implies environmental benefits through reduced compute intensity, aligning with green IT strategies (Unhelkar 2012). 

Beyond cost and energy efficiency, deploying open-weight models on-premises enables organizations to retain complete control over data flows, ownership, and residency. This is particularly relevant in sectors governed by strict compliance and privacy regulations (e.g., GDPR regulations, Nayak et al. 2024). From a data sovereignty perspective, reliance on external APIs from global providers introduces risk, mainly when sensitive or regulated information is processed during inference. Locally hosted LLMs offer a viable pathway to mitigate these concerns while maintaining strategic autonomy over data infrastructure and usage policies. 

These technical and economic factors may suggest that open-weight, locally hosted models are not only a feasible but often preferable option for sustainable LLM deployment in practice. They align with all three pillars of sustainability: lowering energy use, reducing cost, and enabling greater data control and compliance. 

### **_Rethinking AI Benchmarking for Sustainable Deployment_** 

Current AI benchmarking frameworks predominantly assess models based on technical performance metrics such as accuracy, robustness, scalability, and generalization (Li et al. 2024b; White et al. 2024). While essential for academic and technical comparison, these evaluations often omit critical deployment factors 

**3** 



_Sustainability via LLM Right-sizing_ 

|**Model**<br>**Input Cost ($)**|**Output Cost ($)**|
|---|---|
|**Proprietary Models**||
|GPT-4.5-preview<br>75.00|150.00|
|GPT-4o<br>2.50|10.00|
|Claude-3.7-Sonnet<br>3.00|15.00|
|Claude-3.5-Sonnet<br>3.00|15.00|
|Grok-2-Latest<br>2.00|10.00|
|Grok-Beta<br>5.00|15.00|
|o3-mini<br>1.10|4.40|
|Gemini-2.0-Flash<br>0.10|0.40|
|**Open-Weight Models**||
|DeepSeek-R1-Distill-Llama-70B<br>0.75|0.99|
|Llama-3.3-70B-Instruct<br>0.72|0.72|
|Qwen-2.5-Coder-32B<br>0.18|0.18|
|Mistral-Nemo-Instruct-2407<br>0.15|0.15|
|Phi-4<br>0.13|0.50|
|Gemma-3<sup>_∗_</sup><br>0.13|0.50|



**Table 1. Comparison of Token Costs Across AI Models (per million tokens)** 

**Note:** Prices are from https://models.litellm.ai/ and https://llm-stats.com/.<sup>_∗_</sup> No data for Gemma-3 was available at the time of writing, we report Phi-4 pricing as a proxy due to similar size and deployment costs. 

such as energy consumption, operational cost, and data sovereignty—elements that play a decisive role in organizational adoption (Bolón-Canedo et al. 2024). As a result, traditional benchmarks offer limited guidance when selecting models for real-world use under sustainability constraints (Tripathi and Kumar 2025). 

Emerging task-oriented benchmarks aim to address some of these gaps by evaluating reasoning capabilities and linguistic robustness. For example, SimpleBench assesses compositional skills in zero-shot settings (Vaccaro et al. 2024). However, such approaches still focus on cognitive performance and do not account for broader factors like environmental impact, infrastructure costs, or regulatory context, which are crucial for organizational use. 

Platforms such as LMArena.ai<sup>1</sup> , developed by UC Berkeley’s SkyLab, introduce a crowdsourced evaluation methodology using anonymous, randomized model comparisons and the Elo rating system to capture relative performance (Chiang et al. 2024). The approach to reflect user preferences across diverse tasks such as coding or instruction-following is a common way to define “good enough” of LLM output: the fit between LLM output and human output for a comparable task (e.g., Virk et al. 2024) with the goal to mimic human performance, which is not perfect nor does it need to be perfect to be useful (Ferreira et al. 2002). Still, this approach suffers from two limitations: First, it lacks alignment with specific professional task contexts; second, it risks skewed results due to its participants’ demographic or experiential bias. 

In contrast, benchmarking efforts focused on model efficiency and deployment feasibility are beginning to emphasize alternative evaluation criteria. Techniques like post-training quantization, which reduces the precision of model weights, can drastically lower memory and compute requirements while preserving output quality to a large extent (Yao et al. 2024). For instance, recent assessments of quantized Llama-3.1 models show full accuracy recovery across academic and applied tasks, demonstrating their practical suitability (Kurtić et al. 2024). Similarly, structured evaluation of inference efficiency, energy use, and compliancereadiness is critical to understanding a model’s sustainability profile. 

These developments raise the central question: When is a smaller LLM sufficient to support occupational tasks effectively? While larger models tend to improve readability and informativeness, they may still intro- 

> 1https://lmarena.ai/ [Accessed: 08/04/2024] 

**4** 



_Sustainability via LLM Right-sizing_ 

duce fidelity issues or generate hallucinated content (Qi et al. 2025). In contrast, in some experiments (Both et al. 2024), it was shown that smaller models may sometimes be more robust in high-constraint environments—e.g., when output must remain tightly faithful to input sources. This highlights the importance of task specificity: The optimal model depends not only on raw performance but also on how its capabilities align with task complexity and operational needs. 

Neural scaling research further complicates the picture: larger models do not always yield better results unless trained with proportionally larger data sets and compute budgets. For example, the Chinchilla model (70B parameters) outperformed the larger Gopher model (280B) by optimizing the balance between model size and training dataset size (Hoffmann et al. 2022; Kaplan et al. 2020). These findings caution against an overreliance on size alone as a proxy for capability and suggest a need for more nuanced evaluation standards; in fact, the recent Gemma-3 model family from Google achieved impressive performance, often rivaling much larger models (Team et al. 2025). 

## **Methodology** 

To systematically compare the performance of different LLMs across realistic occupational tasks, we developed an evaluation framework based on two LLMs as we detail below; cf. Figure 1. This setup allows for scalable and standardized testing of multiple models on fixed prompts, with consistent evaluation using high-performing LLM. The basic idea behind this approach is that the high-performing LLMs will find output similar to their own answers and knowledge bases more acceptable, akin to the apprenticeship learning paradigm, see e.g., (Grover et al. 2024). Using this approach, we evaluated a wide range of different LLMs and tested which ones deliver acceptable output quality (compared to their larger, more complex siblings) for typical organizational tasks while considering sustainability and cost dimensions. 

### **_LLM-based Evaluation Setup_** 

Our system follows a two-component LLM-based testing and evaluation architecture: an _execution component_ generates output for a given task using a specific LLM, and an _evaluation component_ scores this output for 10 defined criteria using a fixed evaluation prompt. 

Each task LLM is a model instance (e.g., GPT-4o, Llama-3.3, DeepSeek-R1) that receives a structured task prompt (cf. Section Occupational Tasks as Evaluation Prompts) and produces a one-shot output without post-processing or iterative refinement. The generated output is then rated by three evaluator LLMs (Gemini-2.0-Flash, GPT-4o, o3-mini), across ten predefined criteria (cf. Section Evaluation Criteria and Scoring). Ratings were on a scale from 1 (poor) to 10 (excellent). This approach allowed for fully automated comparisons across all model-task combinations. 

![Figure](assets/figure_0001_page_0005.svg)**Figure 1. Overview of the applied Testing and Evaluation Framework** 

### **_Tested Models_** 

We evaluated 11 different LLMs, including both proprietary cloud-based models (e.g., GPT-4o, Claude-3.7, and Gemini-2.0-Flash) and open-weight models that can be run locally on a single A100 GPU (e.g., Llama3.3, DeepSeek-R1, Gemma-3), effectively allowing for local deployment. Table 2 provides an overview of the models, including size, openness, and model access. 

**5** 



_Sustainability via LLM Right-sizing_ 

|**Model Name**|**Size**|**Open**<br>**Weights**|**Thinking**|**Provider**|**Access**|
|---|---|---|---|---|---|
|GPT-4o|Large|✗|✗|OpenAI|OpenAI API|
|o3-mini|Small|✗|✓|OpenAI|OpenAI API|
|Claude-3.5-sonnet|Large|✗|✗|Anthropic|Anthropic API|
|Claude-3.7-sonnet|Large|✗|✗|Anthropic|Anthropic API|
|Gemini-2.0-Flash|Medium|✗|✗|Google|Google API|
|Gemma-3-27b|27B|✓|✗|Google|google/gemma-3-27b-it|
|DeepSeek-R1-70b|70B|✓|✓|DeepSeek|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|
|Llama-3.3-70B-Instruct|70B|✓|✗|Meta|meta-llama/Llama-3.3-70B-Instruct|
|Phi-4|14B|✓|✗|Microsoft|microsoft/phi-4|
|Qwen-2.5-7B-Instruct-1M|7B|✓|✗|Qwen|Qwen/Qwen2.5-7B-Instruct-1M|
|Mistral-Nemo-Instruct-2407|12B|✓|✗|Mistral|mistralai/Mistral-Nemo-Instruct-2407|



**Table 2. Large Language Models used in the Experiments. For open-weight models, the** **_Hugging Face Hub_ identifier has been provided.** 

### **_Occupational Tasks as Evaluation Prompts_** 

To simulate realistic workplace scenarios, we created a set of ten occupational tasks commonly associated with AI assistance in knowledge work. We took inspiration for these tasks from surveys of how LLMs are widely used at the workplace (Chkirbene et al. 2024; Gallagher et al. 2021; Haase 2025). The ten tasks can be found in Table 3. These tasks were selected to span different types of cognitive demand, including _aggregation_ (e.g., meeting minutes), _conceptual_ generation (e.g., outlining a proposal), and _transformation_ (e.g., rewriting information). The logic is that any input is either condensed, enhanced, or transformed into a different form or output style through the LLM. Each task contained structured input material and a clear instruction format. The complete list of tasks, along with the prompting, is available in Appendix A on OSF<sup>2</sup> . 

### **_Evaluation Criteria and Scoring_** 

To assess the quality of outputs generated by each LLM, we implemented a standardized evaluation procedure using three different LLMs as evaluators: _GPT-4o_ , _Gemini-2.0-Flash_ , and _o3-mini_ . This triad was selected to ensure robustness across evaluator types: GPT-4o represents a state-of-the-art, high-performance model, Gemini-2.0-Flash offers a lightweight, general-purpose option, and o3-mini exemplifies a compact “thinking model” architecture. Using multiple evaluators allows us to triangulate scoring consistency and assess potential biases introduced by individual LLM preferences. 

Each evaluator received identical scoring prompts and independently rated the outputs of the eleven tested models across ten evaluation criteria (cf. Table 4). These criteria builds on prior work in LLM evaluation (Chiang et al. 2024; Ji et al. 2023; Long et al. 2024) and are adapted to reflect core dimensions of output quality especially relevant for occupational tasks: (1) _core qualities_ , (2) _factual integrity_ , and (3) _ethical and social responsibility_ . 

Each criterion was scored on a 10-point Likert scale, where 1 indicated poor performance and 10 represented excellent performance. The evaluators processed outputs without knowledge of which model generated them and, importantly, also assessed their own outputs in a blind fashion. This introduces a potential for selffavoring bias. However, our comparative evaluation across the three evaluators (cf. Figure 1 in Appendix B on OSF) shows that GPT-4o was not rated disproportionately higher by itself compared to Gemini-2.0-Flash or o3-mini. This is important as for the following inference statistics and analyses, we use GPT-4o only: using the intra-class correlation coefficient (ICC), a two-way random-effects model with absolute agreement was applied (ICC[2,3]). The resulting ICC was _< ._ 50, indicating bad agreement according to the interpretation guidelines by Koo and Li (2016). Thus, we proceeded to report the main results using scores from GPT-4o as the primary evaluator with the most robust and reliable scores. This is in line with other recent work, 

> 2https://osf.io/p5t8m/?view_only=c694110912c94936809d040def2555e8 

**6** 



_Sustainability via LLM Right-sizing_ 

|**Task Title**|**Brief Description**|
|---|---|
|_Aggregation Tasks_||
|Summarizing a Complex Text|Produce three levels of summary (sentence, executive summary, bullets) from<br>a dense editorial policy text, balancing clarity and information preservation.|
|Generating Meeting Minutes|Transforming jumbled, informal notes into professional meeting minutes, in-<br>cluding participants, agenda, discussion, and action items.|
|Extracting Key Information|Extract action-relevant information (goals, deadlines, open questions) from a<br>long email and organizing it in a clear, scannable format.|
|Creating a Comparison Table|Building a comparison table (features, pros/cons, pricing, use cases) based on<br>unstructured product info, followed by a short selection recommendation.|
|_Conceptual Tasks_||
|Creating a Time Schedule|Generating a structured daily schedule based on task demands, energy levels,<br>and meeting constraints, with justifications for time block choices.|
|Drafting<br>a<br>Project<br>Proposal<br>Outline|Turning a short campaign description into a structured proposal outline (title,<br>goals, methods, timeline, budget) using a bullet-point format.|
|Writing a To-Do List Based on<br>a Goal|Break down a vague event planning goal into actionable steps, grouped by<br>phase, with suggested deadlines and explanatory notes.|
|Planning and Structuring a Pre-<br>sentation|Creating a compelling presentation outline (title, slide structure, opening hook)<br>for a general audience, balancing accessibility and insight.|
|_Transformation Tasks_||
|Writing an Email (Formal and<br>Informal)|Composing two formal and informal emails based on the same situation, ad-<br>justing tone, style, and structure while preserving content.|
|Rewriting for Clarity and Tone|Rewrite a convoluted paragraph into three versions with different tones: neu-<br>tral/professional, friendly/conversational, and concise/direct.|



**Table 3. Occupational Tasks used for Model Evaluation, Grouped by Task Category** 

where GPT-4o has been proposed as a gold standard for LLM-based evaluation due to its high alignment with human judgment (Chiang et al. 2024; Zhao et al. 2024) and comparably bad evaluation performance by other LLMs like Gemini-2.0-Flash and o3-mini (Petrov et al. 2025). Further, both Gemini-2.0-Flash and o3-mini show weaker performances across the tasks and overall very high positive evaluations across tasks (cf. Figure 1 in Appendix B on OSF), lowering their potential to provide proper evaluations. 

Finally, each model-task combination was evaluated only once to reflect typical “one-shot” usage patterns in real-world applications. We intentionally avoided prompt engineering or multi-shot refinement to simulate how an average user or employee might interact with LLMs in everyday organizational settings. This 

|**Category**|**Criterion**|**Explanation**|
|---|---|---|
||Usefulness<br>Relevance|Practical value of the response for the end-user<br>Appropriateness of content given the question and in-|
|Core Qualities|Completeness<br>Clarity<br>Tone Appropriateness|puts<br>Extent to which the answer covers all necessary aspects<br>Coherence and readability of the output<br>Appropriateness of tone for the target audience|
|Factual Integrity|Accuracy<br>Hallucination Freeness|Correctness of factual content<br>Lack of invented or misleading content|
||Ethical Bias|Presence of inclusive, non-discriminatory content|
|Ethical & Social Respon|sibility<br>Political Bias<br>Censorship|Absence of politically slanted or agenda-driven framing<br>Avoidance of overly filtered or withheld responses|



**Table 4. Evaluation Criteria Grouped by Category** 

**7** 



_Sustainability via LLM Right-sizing_ 

ecological approach allows us to assess sufficiency rather than maximal performance and better informs the realistic applicability of each model across tasks. 

### **_Analytical Strategy_** 

We aggregated output scores across tasks to systematically compare the evaluated LLMs and conducted analyses at both the model and task levels. We worked with the plain scores from the 1-10 evaluations for the following analysis. For a more precise comparison between models across tasks, we applied _z_ -transformation to all task scores because individual tasks varied in difficulty and output expectations. This allowed us to compare relative model performance across tasks on a standardized scale and to interpret performance differences in terms of deviation from average task difficulty (cf. Appendix C on OSF). 

We pursued three complementary analytical steps to address our research questions: 

1. **Model-level comparison:** We conducted one-way ANOVAs to test whether different models produced significantly different outputs in terms of overall quality. This analysis established whether model choice systematically influences the usability and quality of generated content, particularly in light of sustainability trade-offs (e.g., smaller vs. larger models). 

2. **Task-type comparison:** To explore how performance varies by cognitive demand, we grouped tasks into three categories— _aggregation_ , _transformation_ , and _conceptual_ —and performed ANOVAs with post-hoc Games-Howell tests. This helped us identify which task types posed particular challenges for LLMs, and whether model suitability is task-specific. 

3. **Cluster analysis:** We performed K-means clustering based on the ten evaluation criteria to group models into distinct performance profiles. This allowed us to go beyond individual criteria and detect overarching patterns—e.g., premium all-rounders vs. reliable minimalists—supporting decision-making for context-sensitive model selection. 

All analyses reported in the main paper were based on evaluation scores generated by GPT-4o, which we used as a consistent, high-performance evaluator to visualize the evaluation pattern of all three evaluators. Supplementary analyses using the other evaluators are available in Appendix B on OSF. 

By combining standardized scoring, inferential statistics, and unsupervised pattern detection, this analytical strategy provides both detailed and holistic insights into LLM performance, informing the trade-offs between model size, sustainability, and sufficiency for occupational work. 

## **Results** 

We present the results of our comparative evaluation of eleven LLMs across ten everyday occupational tasks. Analyses were conducted at the level of evaluation dimensions, task categories, and individual tasks, and were complemented by a cluster analysis to identify model performance profiles. 

### **_Performance Analysis across Evaluation Criteria_** 

Individual ANOVA for evaluation criteria across models revealed significant differences in the task performance, indicating that model choice significantly affects the quality of outputs across everyday workplace tasks (cf. Table 5). Importantly, despite these differences, average scores across models were generally high, reflecting the maturity of current LLMs for occupational applications. Breaking down performance by evaluation criterion based on GPT-4o ratings, we found statistically significant differences for: _Accuracy_ , _F_ (10 _,_ 99) = 2 _._ 79, _p_ = _._ 004; _Clarity_ , _F_ (10 _,_ 99) = 2 _._ 47, _p_ = _._ 011; _Completeness_ , _F_ (10 _,_ 99) = 2 _._ 21, _p_ = _._ 023; _Relevance_ , _F_ (10 _,_ 99) = 3 _._ 22, _p_ = _._ 001; _Tone Appropriateness_ , _F_ (10 _,_ 99) = 2 _._ 65, _p_ = _._ 007; and notably, _Usefulness_ , _F_ (10 _,_ 99) = 3 _._ 48, _p < ._ 001. These criteria reflect practical dimensions of output quality, especially usefulness, which captures whether human users can directly apply, adapt, or build upon the generated text. In this sense, usefulness acts as a proxy for “operational sufficiency” and may be especially relevant when considering task automation or AI augmentation in real-world settings. 

In contrast, no significant differences emerged in criteria related to factual safety or ethics: _Hallucination_ 

**8** 



_Sustainability via LLM Right-sizing_ 

_Freeness_ , _F_ (10 _,_ 99) = 1 _._ 71, _p_ = _._ 089; _Ethical Bias_ , _F_ (10 _,_ 99) = 0 _._ 08, _p_ = 1 _._ 000; _Political Bias_ , _F_ (10 _,_ 99) = 0 _._ 08, _p_ = 1 _._ 000; and _Censorship_ , _F_ (10 _,_ 99) = 0 _._ 10, _p_ = 1 _._ 000. This suggests that most models—even smaller open-weight ones—have achieved a robust baseline in minimizing inappropriate or misleading content, likely due to consistent alignment training and safety reinforcement across providers. 

These results show that while LLMs are broadly capable across core dimensions, meaningful differences remain in how well they deliver user-relevant and context-sensitive outputs. These differences are particularly pronounced in criteria that directly impact downstream usability—underscoring the importance of task-aware evaluation beyond simple correctness or style. 

|**Model**||**Cor**|**e Qualities**|||**Factual I**|**ntegrity**|**Re**|**sponsibilit**|**y**|
|---|---|---|---|---|---|---|---|---|---|---|
||Usefulness|Relevance|Completeness|Clarity|Tone|Accuracy|Halluc.|Eth. Bias|Pol. Bias|Censor.|
||||**Proprietar**|**y Model**|**s**||||||
|Claude-3.5-sonnet-latest|8.70|9.20|8.85|8.85|9.35|9.20|9.60|9.75|9.80|9.85|
|Claude-3.7-sonnet-latest|8.85|9.25|8.90|9.10|9.40|9.25|9.75|9.75|9.85|9.90|
|Gemini-2.0-Flash|8.85|8.95|8.85|8.85|9.15|9.05|9.70|9.80|9.85|9.95|
|GPT-4o|9.50|9.70|9.25|9.30|9.70|9.40|9.95|9.85|9.85|9.90|
|o3-mini|8.90|9.25|8.90|8.65|9.05|9.00|9.65|9.80|9.80|9.90|
|||**Open**|**-Weight Model**|**s Locally**|**Deplo**|**yed**|||||
|DeepSeek-R1-70b|8.05|8.50|8.30|8.55|9.00|8.55|9.40|9.75|9.80|9.90|
|Gemma-3-27b|8.60|8.90|8.60|8.75|9.05|8.90|9.45|9.75|9.85|9.90|
|Llama-3.3-70B-Instruct|8.50|8.80|8.55|8.50|8.80|8.70|9.15|9.70|9.75|9.85|
|Mistral-Nemo-Instruct-2407|8.15|8.50|8.40|8.35|8.80|8.55|9.35|9.75|9.80|9.85|
|Phi-4|8.65|8.85|8.70|8.75|9.05|8.90|9.40|9.70|9.80|9.85|
|Qwen-2.5-7B-Instruct-1M|8.35|8.60|8.25|8.55|8.85|8.65|9.40|9.70|9.75|9.80|



**Table 5. Evaluation of LLMs Across Core Qualities, Integrity, and Responsibility** 

### **_Task Category Effects_** 

We computed an overall evaluation score per model by averaging across the ten tasks (cf. Table 6). A oneway ANOVA confirmed significant performance differences, _F_ (10 _,_ 99) = 3 _._ 51, _p < ._ 001, with an effect size of _η_<sup>2</sup> = _._ 26. This indicates that model selection explains approximately 26% of the variance in output quality. 

To test whether LLM performance varied across different types of tasks, we grouped the ten tasks into three categories: _Aggregation tasks_ (e.g., summarizing, listing); _Transformation tasks_ (e.g., rewriting, reorganizing), and _Conceptual tasks_ (e.g., ideation, proposal drafting). 

A one-way ANOVA with task category as the independent variable and overall evaluation score as the dependent variable revealed a significant effect, _F_ (2 _,_ 107) = 15 _._ 80, _p < ._ 001, with a large effect size ( _η_<sup>2</sup> = _._ 228). Levene’s test indicated unequal variances ( _p_ = _._ 002), so Games-Howell post-hoc tests were used. These revealed that: _Conceptual_ tasks received significantly lower scores than both _Aggregation_ ( _p < ._ 001) and _Transformation_ tasks ( _p_ = _._ 012). No significant difference was found between _Aggregation_ and _Transformation_ tasks ( _p_ = _._ 624). These results show that LLMs struggled most with open-ended conceptual prompts, suggesting limits in “generative reasoning” or “creative structuring”. 

### **_Task-Specific Performance Differences_** 

A one-way ANOVA was conducted to assess whether LLM performance varied significantly across specific tasks using _overall_score_ as the dependent variable and _task_name_ as the independent variable. The analysis revealed a significant main effect of task, _F_ (9 _,_ 100) = 8 _._ 14, _p < ._ 001, indicating substantial differences in performance across tasks. The effect size was large, with _η_<sup>2</sup> = _._ 423. For more detailed analyses, across all three evaluators, for each task and LLM, see Appendix B on OSF. 

Levene’s test for homogeneity of variance was significant ( _p < ._ 001), violating the assumption of equal variances; thus, the Games-Howell post-hoc test was used. The results showed that the task _Summarizing a Complex Text_ received significantly lower scores than several other tasks, including _Writing an Email (Formal and Informal)_ , _Generating Meeting Minutes_ , _Rewriting Text for Clarity and Tone_ , _Extracting Key Information from a Document_ , _Creating a Comparison Table_ , and _Planning and Structuring a Presentation_ 

**9** 



_Sustainability via LLM Right-sizing_ 

|**Model**|**Overall**|**Aggregation**|**Conceptual**|**Transformation**|
|---|---|---|---|---|
||**Prop**|**rietary Models**|||
|Claude-3.5-sonnet|9.16|9.05|9.13|9.25|
|Claude-3.7-sonnet-latest|9.25|9.15|9.15|9.40|
|Gemini-2.0-Flash|9.18|9.40|8.83|9.43|
|GPT-4o|9.62|9.55|9.45|9.83|
|o3-mini|9.14|9.15|8.80|9.48|
|**Open**|**-Weight **|**Models Locally **|**Deployed**||
|DeepSeek-R1-70b|8.71|8.70|8.23|9.20|
|Gemma-3-27b|8.92|8.85|8.73|9.15|
|Llama-3.3-70B-Instruct|8.82|8.50|8.68|9.13|
|Mistral-Nemo-Instruct-2407|8.68|8.70|8.18|9.18|
|Phi-4|8.98|8.60|8.83|9.33|
|Qwen-2.5-7B-Instruct-1M|8.75|8.35|8.50|9.20|



**Table 6. Performance Scores Across Tasks and Task Categories (on 0–10 scale)** 

( _p < ._ 05). Potentially because the text to be summarized required a relatively long input sequence. In contrast, the task _Planning and Structuring a Presentation_ received significantly higher scores than tasks such as _Writing an Email_ , _Generating Meeting Minutes_ , _Rewriting Text_ , _Extracting Key Information_ , and _Creating a Comparison Table_ . For a detailed comparison, based on z-scores for each LLM for all 10 tasks, see Appendix C on OSF. 

These findings suggest that the specific task prompt had a notable impact on LLM output quality. Tasks requiring conceptualizations and idea generation posed more significant challenges for most models, and tasks like rewriting in style and tone appeared to better align with the capabilities of the models under evaluation. 

### **_Clustering of LLMs Based on Evaluation Profiles_** 

To identify overarching patterns in model performance, we conducted a K-means cluster analysis based on the ten evaluation criteria ( _Usefulness, Relevance, Completeness, Clarity, Tone Appropriateness, Accuracy, Hallucination Freeness, Ethical Bias, Political Bias,_ and _Censorship_ ). The analysis included the mean scores across all tasks for each LLM. A three-cluster solution was specified, which successfully grouped the eleven models into distinct performance profiles. The three-cluster solution grouped the models into clearly distinct sets: Cluster 1 (n = 1), Cluster 2 (n = 6), and Cluster 3 (n = 4). The K-means solution explained a large proportion of variance in key criteria, with high between-cluster variation in _Usefulness, Completeness, Accuracy_ , and _Relevance_ . A series of one-way ANOVAs was conducted to descriptively assess the differences in evaluation criteria across the three clusters. The analysis revealed large and statistically significant differences in most dimensions, including _Usefulness_ , _F_ (2 _,_ 8) = 28 _._ 01, _p < ._ 001, completeness, _F_ (2 _,_ 8) = 24 _._ 56, _p < ._ 001, _Accuracy_ , _F_ (2 _,_ 8) = 22 _._ 38, _p < ._ 001, and _Relevance_ , _F_ (2 _,_ 8) = 19 _._ 44, _p < ._ 001. More minor but still significant differences were observed for _Hallucination Freeness, Clarity, Tone Appropriateness, Ethical Bias_ , and _Political Bias_ ( _p < ._ 05). No significant differences were found in _Censorship_ scores, which remained consistently high across all clusters. 

To estimate the strength of these differences, we computed the proportion of variance explained by the clustering (expressed as _η_<sup>2</sup> ). The clustering explained 88% of the variance in _Usefulness_ scores, 83% in _Completeness_ , 82% in _Accuracy_ , 71% in _Relevance_ , and 68% in _Clarity_ . These large effect sizes confirm that the clustering structure reflects divergent performance profiles across models. It is important to note that these ANOVAs serve descriptive purposes, as the clustering algorithm itself optimizes separation across groups. Nevertheless, they provide strong evidence that the identified clusters capture meaningful differences in LLM performance across evaluation dimensions. 

**10** 



_Sustainability via LLM Right-sizing_ 

#### **Cluster 1: Premium All-Rounder.** 

This cluster contained a single model: _GPT-4o_ . It was characterized by the highest performance across all evaluation dimensions, including _Clarity_ ( _M_ = 9 _._ 30), _Hallucination Freeness_ ( _M_ = 9 _._ 95), usefulness ( _M_ = 9 _._ 50), and _Relevance_ ( _M_ = 9 _._ 70). GPT-4o showed no notable weaknesses, excelling in factual accuracy, stylistic dimensions, and ethical criteria. This model thus stands out as a premium, high-performance allrounder. 

#### **Cluster 2: Competent and Responsible Models.** 

This group included _Claude-3.5_ , _Claude-3.7_ , _Gemini-2.0-Flash_ , _Gemma-3_ , _o3-mini_ , and _Phi-4_ . These models demonstrated solid overall performance, particularly in responsibility-related criteria such as _Censorship_ ( _M_ = 9 _._ 89), _Ethical Bias_ ( _M_ = 9 _._ 76), and _Tone Appropriateness_ ( _M_ = 9 _._ 17). While slightly behind GPT4o on _Clarity_ , _Usefulness_ , and _Completeness_ , they still maintained high levels of consistency across evaluation criteria. This cluster represents balanced, reliable models with responsible and stylistically sound output. 

#### **Cluster 3: Limited but Safe Models.** 

The third cluster consisted of _DeepSeek_ , _Llama-3_ , _Mistral-Nemo-Instruct_ , and _Qwen-2.5_ . These models received the lowest average scores across nearly all dimensions, particularly _Usefulness_ ( _M_ = 8 _._ 26), _Clarity_ ( _M_ = 8 _._ 49), and _Completeness_ ( _M_ = 8 _._ 38). While all these scores are relatively high, their output can in part be considered unuseful enough to require second runs in practice or a considerable effort by the human user to make it usable. Despite these limitations, they still performed relatively well on ethical criteria, including _Censorship_ ( _M_ = 9 _._ 85) and _Political Bias_ ( _M_ = 9 _._ 78). This group can be interpreted as containing technically and ethically safe models, yet underwhelming in terms of expressive output quality or task richness. 

These findings provide further nuance to the comparative evaluation of LLMs, showing that while some models offer broad excellence across evaluation dimensions, others specialize in either responsibility or baseline task completion. The cluster profiles provide practical insights for selecting LLMs based on application priorities, such as creative fluency versus safety and bias mitigation. 

## **Discussion** 

In our context, “good enough” does not refer to reaching the absolute best possible output quality, but rather to whether an LLM produces results that are (a) functionally usable, (b) informationally sufficient, and (c) stylistically appropriate for professional downstream use. In other words, does the output enable a knowledge worker to continue or complete a task without major revisions, corrections, or risk exposure? This operational definition differs from generic benchmarking or human-substitution goals often found in crowdbased evaluations such as Chatbot Arena (Chiang et al. 2024), which measure relative preference across diverse tasks in anonymous settings. While such platforms offer useful performance gradients, they tend to conflate general helpfulness with task fitness and overlook domain constraints. Our approach therefore, grounds the evaluation in concrete occupational tasks with clear criteria, aligning “good enough” with practical utility in real-world workflows rather than abstract model superiority. 

### **_Performance Versus Sufficiency: Task Matters, So Does Context_** 

Our results show that model choice does matter: significant performance differences were observed across the eleven tested LLMs. The model GPT-4o consistently outperformed all other models across core quality criteria (usefulness, clarity, completeness, accuracy), establishing itself as a strong all-rounder. Claude-3.7 also performed reliably, especially in tasks requiring structured planning or synthesis. These models represent the current benchmark for high-quality LLM support in business contexts. At the same time, the even more powerful “thinking models” like o3-mini or Claude-3.7-thinking are not that well suited for our test tasks, not because they are not effective but rather because they require a very different prompting approach, which is not akin to the dialog style most users are used to and employ; they are much more tailored 

**11** 



_Sustainability via LLM Right-sizing_ 

to solving complex tasks with a single answer with very specific and extensive prompting<sup>3</sup> . At the same time, certain smaller (and open-weight) models, particularly Gemma-3 and Phi-4, achieved solid and stable performance. While they did not reach top-tier scores, they delivered consistent and usable results. This positions them as viable alternatives in settings where autonomy, cost-efficiency, or deployment control take priority over peak performance. In contrast, most of the open-weight models evaluated displayed consistently weak performance, rendering them less suitable for reliable occupational use. Specifically, Mistral-NemoInstruct, DeepSeek-R1, Llama-3.3, and Qwen-2.5 frequently received negative to strongly negative ratings across multiple tasks. Common deficiencies included difficulties in producing clear, structured outputs, inadequate information summarization, and poor clarity or tone in rewriting tasks. 

It is also important to note that model performance varied not just by architecture but by task type. Conceptual prompts (e.g., ideation or synthesis) proved more challenging than structured or transformationoriented tasks. This reinforces the need for task-aware model selection: some applications may be well-served by efficient smaller models, while others still demand the cognitive flexibility of more advanced systems. These patterns underline the necessity of careful selection of LLMs according to task requirements. For versatile everyday applications, GPT-4o and Claude-3.7 are highly recommended due to their consistent quality and adaptability. Gemini-2.0-Flash, Gemma-3, and Phi-4 offer niche advantages in certain use cases, but require cautious deployment to ensure alignment with their specific strengths. The remaining models, including o3-mini, Claude-3.5, Mistral-Nemo-Instruct, DeepSeek-R1, Llama-3.3, and Qwen-2.5, are generally not recommended for occupational tasks due to their frequent shortcomings in quality and reliability. Future research should explore targeted improvements for these weaker-performing models to enhance their task effectiveness. 

### **_Reframing Sustainable AI Deployment: Cost, Carbon, and Control_** 

Our findings contribute to a broader rethinking of what sustainable AI deployment entails. While environmental sustainability is often framed in terms of carbon emissions, a more complete view across all three dimensions of sustainability also includes financial viability and data governance. These additional dimensions are especially salient when LLMs are deployed at scale in cost-sensitive, compliance-constrained, or infrastructure-limited environments. 

Inference cost is a central, yet often overlooked, factor in operational sustainability. Unlike model training, which is a one-time event, inference occurs repeatedly—often millions of times—and accumulates substantial energy use and emissions over time (Li et al. 2024a; Luccioni et al. 2022). Yet estimating inference energy consumption is notoriously difficult. Key determinants include model size, GPU efficiency, batch size, hardware type, and system-level optimizations such as KV caching or quantization; see e.g., Aschenbrenner (2024). Empirical estimates suggest that inference for current LLMs consumes between 0.5 and 1.3 kWh per million tokens, resulting in 150-300 gCO2e under a typical energy mix (Samsi et al. 2023; You 2025). Larger models like GPT-4 typically lie at the upper end of this range, while quantized and smaller models like Mistral-7B or LLaMA2-7B show lower per-token footprints. Importantly, these figures exclude the embodied emissions of hardware and vary depending on whether models are deployed in the cloud or on local infrastructure. 

Cost and energy efficiency can also be optimized through deployment strategies. Hardware matters: H100 GPUs, for example, offer up to 4.6x the inference throughput of A100s (You 2025). Similarly, quantization methods (e.g., INT8) and effective batching can reduce energy consumption significantly. However, poor utilization—as seen, e.g., in the case of one open deployment of BLOOM—can lead to high per-request emissions due to idle resource overhead (Luccioni et al. 2022). In practice, organizations must balance these cost and carbon considerations against deployment constraints. Public cloud APIs provide state-of-the-art performance but introduce privacy risks, data transfer costs, and potential legal barriers to inference on sensitive documents. Even hybrid architectures may exclude confidential datasets, as seen in AI assistants within public research institutions (Weber et al. 2024). 

Based on the performance results from this study’s analyses, especially the cluster analyses showed clear differences between proprietary models and open weights, with the former typically scoring better. Figure 2 

> 3https://www.latent.space/p/o1-skill-issue [Accessed: 08/04/2025] 

**12** 



_Sustainability via LLM Right-sizing_ 

visualizes the trade-off between overall performance and cumulative input/output cost across models, illustrating the performance differences of different LLMs. Gemini-2 has the best price/performance ratio, but with Google as the hosting company leveraging Google’s in-house TPUs, which are optimized ASICs for ML loads. Gemma-3 and Phi-4, therefore, show the best combination of performance/price for the general user; in particular, both models are open-weight and can run locally on a single (mid-range to high-end) GPU. This underscores the value of locally deployable models that can run efficiently on in-house GPUs in terms of sustainability, data sovereignty, and cost-efficiency. Our findings show that such models offer sufficient performance while drastically reducing cost-per-token and improving organizational control over data flows. 

![Figure](assets/figure_0002_page_0013.svg)**Figure 2. LLM Performance vs. Sum of Input + Output Cost (by Marker Shape)** 

### **_Toward Task-Aware, Context-Driven Benchmarking_** 

Although our study does not yet offer a complete decision framework, it provides empirical grounding for a broader benchmarking perspective that aligns model choice with real-world goals and constraints. We argue that evaluating LLMs solely through technical benchmarks risks overlooking critical dimensions of sustainability. Instead, sufficiency must be evaluated contextually: what is “good enough” depends on the task, the infrastructure, the cost model, and the data environment. 

Our LLM-based evaluation method—using fixed tasks, standardized criteria, and LLM-based scoring—offers a scalable approach to capture such trade-offs. By running each model-task pair once, we simulate real-world use rather than optimized performance. This design choice highlights the practical gaps between idealized capabilities and applied value. Ultimately, our findings support a shift toward task-aware, context-driven LLM selection. For certain creative or high-stakes applications, premium models like GPT-4o may be necessary. But for many tasks, especially those requiring privacy, control, and cost-efficiency, smaller models are already good enough. The future of sustainable AI will depend not just on improving model capabilities but on improving our ability to choose wisely. 

## **Limitations** 

Several limitations should be noted when interpreting the results of this study. First, while our evaluation spanned eleven diverse LLMs and ten representative tasks, the experimental setup was intentionally limited to single-shot outputs without prompt optimization or few-shot prompting. This design choice enhances 

**13** 



_Sustainability via LLM Right-sizing_ 

ecological validity–reflecting how many users interact with LLMs in practice–but may underrepresent the full performance potential of specific models under optimized conditions. 

Second, all inference statistics like ANOVAs were conducted using GPT-4o as the sole evaluator. While GPT-4o has demonstrated strong capabilities in evaluation tasks (e.g. (Chiang et al. 2024; Zhao et al. 2024)), relying on a single LLM introduces potential biases based on its preferences or blind spots. We mitigated this by using clear, rubric-based scoring criteria. Still, future work should incorporate human raters or multiple evaluators to validate the robustness of LLM-based scoring. 

Third, our study focused on output quality and sustainability indicators but did not include direct energy consumption or latency measurements. We relied on publicly available estimates for model efficiency and token-level costs. While useful for comparison, actual deployment costs and carbon footprints may vary significantly depending on infrastructure, batch size, and inference patterns. 

Finally, our sufficiency argument remains exploratory. We do not claim that a single score threshold defines adequacy for all contexts. Instead, we highlight the need for contextualized evaluation strategies that take into account task complexity, user goals, and organizational constraints. 

## **Conclusion** 

This paper contributes to the growing conversation on sustainable and context-aware deployment of LLM in organizational settings. A systematic comparison of eleven proprietary and open-weight LLMs across ten everyday tasks shows that smaller, locally deployable models can be “good enough” for many real-world use cases, particularly when data privacy, cost efficiency, and infrastructure autonomy are essential. 

Our findings demonstrate that while models like GPT-4o offer best-in-class performance, specific open-weight models, such as Gemma-3 and Phi-4, can often provide consistent and sufficient output quality at a fraction of the cost and energy footprint. These insights support a more nuanced understanding of AI sufficiency that goes beyond technical benchmarking to include sustainability, sovereignty, and practicality. 

We argue that evaluating LLMs solely through traditional metrics risks overlooking critical operational considerations. Our agent-based evaluation method offers a replicable and scalable approach to model assessment, and our results suggest the value of task-aware, context-driven model selection. 

Future work should extend this approach by developing sufficiency frameworks incorporating task characteristics, infrastructure constraints, and user needs. In some cases, the most sustainable choice may not be to deploy an LLM at all—a decision that requires as much empirical grounding as adoption. As LLMs become embedded in organizational life, responsible deployment must begin with asking not only which model is best, but which model is _right_ . 

## **Acknowledgments** 

The authors’ work was partially supported by the German Federal Ministry of Education and Research (BMBF), grant number 16DII133 (Weizenbaum-Institute), by Deutsche Forschungsgemeinschaft under grants 496119880 (VisualMine), 531115272 (ProImpact), and SFB 1404/2 (FONDA), as well as Deutsche Forschungsgemeinschaft (DFG) through the DFG Cluster of Excellence MATH+ (grant number EXC-2046/1, project ID 390685689), as well as the Zuse Institute Berlin via the RISE@ZIB services hosting the LLM models. 

## **References** 

Aschenbrenner, L. 2024. _Situational Awareness_ . Tech. rep. situational-awareness.ai, 2024, p. 165. Bolón-Canedo, V., Morán-Fernández, L., Cancela, B., and Alonso-Betanzos, A. 2024. “A review of green artificial intelligence: Towards a more sustainable future,” _Neurocomputing_ (599), p. 128096. 

Bossert, L. N. and Loh, W. 2025. “Why the Carbon Footprint of Generative Large Language Models Alone Will Not Help Us Assess Their Sustainability,” _Nature Machine Intelligence_ (7:2) 2025, pp. 164–165. 

Both, C., Hoover, B., Strobelt, H., Krotov, D., Weidele, D. K. I., Martino, M., and Dehmamy, N. 2024. “The Role of Task Complexity in Emergent Abilities of Small Language Models,” () 2024. 

**14** 



_Sustainability via LLM Right-sizing_ 

- Chen, Z., Wu, M., Chan, A., Li, X., and Ong, Y.-S. 2023. “Survey on AI Sustainability: Emerging Trends on Learning Algorithms and Research Challenges [Review Article],” _IEEE Computational Intelligence Magazine_ (18:2), pp. 60–77. 

- Chiang, W.-L. et al. 2024. “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” in: _Forty-First International Conference on Machine Learning,_ 2024. 

- Chkirbene, Z., Hamila, R., Gouissem, A., and Devrim, U. 2024. “Large Language Models (LLM) in Industry: A Survey of Applications, Challenges, and Trends,” in: _2024 IEEE 21st International Conference on Smart Communities: Improving Quality of Life Using AI, Robotics and IoT (HONET),_ 2024, pp. 229– 234. 

- Cowls, J., Tsamados, A., Taddeo, M., and Floridi, L. 2023. “The AI Gambit: Leveraging Artificial Intelligence to Combat Climate Change—Opportunities, Challenges, and Recommendations,” _AI & SOCIETY_ (38:1) 2023, pp. 283–307. 

- del Valle, J. I. and Lara, F. 2024. “AI-powered Recommender Systems and the Preservation of Personal Autonomy,” _AI & SOCIETY_ (39:5) 2024, pp. 2479–2491. 

- Ferreira, F., Bailey, K. G., and Ferraro, V. 2002. “Good-Enough Representations in Language Comprehension,” _Current Directions in Psychological Science_ (11:1) 2002, pp. 11–15. 

- Fioravante, R. 2024. “Beyond the Business Case for Responsible Artificial Intelligence: Strategic CSR in Light of Digital Washing and the Moral Human Argument,” _Sustainability_ (16:3) 2024, p. 1232. 

- Gallagher, K. M., Cameron, L., De Carvalho, D., and Boulé, M. 2021. “Does Using Multiple Computer Monitors for Office Tasks Affect User Experience?: A Systematic Review,” _Human Factors: The Journal of the Human Factors and Ergonomics Society_ (63:3) 2021, pp. 433–449. 

- Grover, R., Vats, A., Moorman, N., Agrawal, A., and Gombolay, M. 2024. “Better Apprenticeship Learning with LLM Explanations,” in: _Proceedings of the AAAI Symposium Series,_ vol. 4. 1, pp. 130–136. 

- Haase, J. 2025. _Augmenting Coaching with GenAI: Insights into Use, Effectiveness, and Future Potential_ . 2025. arXiv: 2502.14632 `[cs]` . 

- Haase, J., Kremser, W., Leopold, H., Mendling, J., Onnasch, L., and Plattfaut, R. 2024. “Interdisciplinary Directions for Researching the Effects of Robotic Process Automation and Large Language Models on Business Processes,” _Communications of the Association for Information Systems_ (54:1) 2024. 

- Hoffmann, J. et al. 2022. _Training Compute-Optimal Large Language Models_ . 2022. arXiv: 2203.15556 `[cs]` . 

- Hogan, M. 2024. “AI Is a Hot Mess,” in: _Training the Archive. Cologne: Verlag Der Buchhandlung Walther Und Franz König, Forthcoming,_ Training the Archive. Verlag der Buchhandlung Walther und Franz König. 

- Hui, Z., and and Khan, N. A. 2024. “When Service Quality Is Enhanced by Human–Artificial Intelligence Interaction: An Examination of Anthropomorphism, Responsiveness from the Perspectives of Employees and Customers,” _International Journal of Human–Computer Interaction_ (40:22) 2024, pp. 7546–7561. 

- Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., Chen, B., Sun, R., Wang, Y., and Yang, Y. 2023. “BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset,” _Advances in Neural Information Processing Systems_ (36) 2023, pp. 24678–24704. 

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. 2020. _Scaling Laws for Neural Language Models_ . 2020. arXiv: 2001.08361 `[cs]` . 

- Kirchner-Krath, J., Morschheuser, B., Sicevic, N., Xi, N., Von Korflesch, H. F., and Hamari, J. 2024. “Challenges in the Adoption of Sustainability Information Systems: A Study on Green IS in Organizations,” _International Journal of Information Management_ (77) 2024, p. 102754. 

- Klessascheck, F., Weber, I., and Pufahl, L. 2025. “SOPA: A Framework for Sustainability-Oriented Process Analysis and Re-Design in Business Process Management,” _Information Systems and e-Business Management_ () 2025. 

- Koo, T. K. and Li, M. Y. 2016. “A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research,” _Journal of Chiropractic Medicine_ (15:2) 2016, pp. 155–163. 

- Kotlarsky, J., Oshri, I., and Sekulic, N. 2023. “Digital Sustainability in Information Systems Research: Conceptual Foundations and Future Directions,” _Journal of the Association for Information Systems_ (24:4), pp. 936–952. 

- Kurtić, E., Marques, A., Kurtz, M., and Alistarh, D. 2024. _We Ran over Half a Million Evaluations on Quantized LLMs—Here’s What We Found_ . 2024. 

- Lazar, S. and Manuali, L. 2024. _Can LLMs Advance Democratic Values?_ 2024. arXiv: 2410.08418 `[cs]` . 

**15** 



_Sustainability via LLM Right-sizing_ 

- Leon, M. 2024. “The Escalating AI’s Energy Demands and the Imperative Need for Sustainable Solutions,” _WSEAS TRANSACTIONS ON SYSTEMS_ (23) 2024, pp. 444–457. 

- Li, B., Jiang, Y., Gadepally, V., and Tiwari, D. 2024a. “SPROUT: Green Generative AI with CarbonEfficient LLM Inference,” _arXiv preprint arXiv:2403.12900_ (). 

- Li, P., Yang, J., Islam, M. A., and Ren, S. 2023. “Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models,” (). 

- Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. 2024b. _From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline_ . 2024. arXiv: 2406.11939 `[cs]` . 

- Long, L., Wang, R., Xiao, R., Zhao, J., Ding, X., Chen, G., and Wang, H. 2024. _On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey_ . 2024. arXiv: 2406.15126 `[cs]` . 

- Luccioni, A. S., Viguier, S., and Ligozat, A.-L. 2022. “Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model,” _arXiv preprint arXiv:2211.02001_ (). 

- Nayak, S. P., Pasumarthi, S., Rajagopal, B., and Verma, A. K. 2024. “GDPR Compliant ChatGPT Playground,” in: _2024 International Conference on Emerging Technologies in Computer Science for Interdisciplinary Applications (ICETCS),_ 2024, pp. 1–6. 

- Petrov, I., Dekoninck, J., Baltadzhiev, L., Drencheva, M., Minchev, K., Balunović, M., Jovanović, N., and Vechev, M. 2025. _Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad_ . 2025. arXiv: 2503.21934 `[cs]` . 

- Purvis, B., Mao, Y., and Robinson, D. 2019. “Three Pillars of Sustainability: In Search of Conceptual Origins,” _Sustainability Science_ (14:3), pp. 681–695. 

- Qi, Z., Luo, H., Huang, X., Zhao, Z., Jiang, Y., Fan, X., Lakkaraju, H., and Glass, J. 2025. “Quantifying Generalization Complexity for Large Language Models,” (). 

- Raisch, S. and Fomina, K. 2024. “Combining Human and Artificial Intelligence: Hybrid Problem-Solving in Organizations,” _Academy of Management Review_ () 2024, amr.2021.0421. 

- Samsi, S., Yuen, S., Sundar, M. V., Bates, N., Morrow, J., Elliot, J., et al. 2023. “From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference,” _arXiv preprint arXiv:2310.03003_ (). 

- Sarker, S., Susarla, A., Gopal, R., and Thatcher, J. B. 2024. “Democratizing Knowledge Creation through Human-AI Collaboration in Academic Peer Review,” _Journal of the Association for Information Systems_ (1), pp. 158–171. 

- Schütze, P. 2024. “The Problem of Sustainable AI: A Critical Assessment of an Emerging Phenomenon,” _Weizenbaum Journal of the Digital Society_ (4:1). 

- Seidel, S. et al. 2017. “The Sustainability Imperative in Information Systems Research,” _Communications of the Association for Information Systems_ (40), pp. 40–52. 

- Singh, A., Patel, N. P., Ehtesham, A., Kumar, S., and Khoei, T. T. 2025. “A Survey of Sustainability in Large Language Models: Applications, Economics, and Challenges,” in: _2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC),_ 2025, pp. 00008–00014. 

- Team, G. et al. 2025. _Gemma 3 Technical Report_ . 2025. arXiv: 2503.19786 `[cs]` . 

- Tripathi, A. and Kumar, V. 2025. “Ethical Practices of Artificial Intelligence: A Management Framework for Responsible AI Deployment in Businesses,” _AI and Ethics_ () 2025. 

- Unhelkar, B. 2012. “Enterprise Green IT Strategy,” in: _Harnessing Green It,_ S. Murugesan and G. R. Gangadharan (eds.). 1st ed. Wiley, 2012, pp. 149–165. 

- Vaccaro, M., Almaatouq, A., and Malone, T. 2024. “When Combinations of Humans and AI Are Useful: A Systematic Review and Meta-Analysis,” _Nature Human Behaviour_ (8:12) 2024, pp. 2293–2303. 

- Veit, D. J. and Thatcher, J. B. 2023. “Digitalization as a Problem or Solution? Charting the Path for Research on Sustainable Information Systems,” _Journal of Business Economics_ (93:6-7) 2023, pp. 1231– 1253. 

- Verdecchia, R., Sallou, J., and Cruz, L. 2023. “A systematic review of Green AI,” _WIREs Data Mining and Knowledge Discovery_ (13:4), e1507. 

- Virk, Y., Devanbu, P., and Ahmed, T. 2024. _Enhancing Trust in LLM-Generated Code Summaries with Calibrated Confidence Scores_ . 2024. arXiv: 2404.19318 `[cs]` . 

- Watson, Boudreau, and Chen 2010. “Information Systems and Environmentally Sustainable Development: Energy Informatics and New Directions for the IS Community,” _MIS Quarterly_ (34:1), p. 23. 

**16** 



_Sustainability via LLM Right-sizing_ 

- Weber, I., Linka, H., Mertens, D., Muryshkin, T., Opgenoorth, H., and Langer, S. 2024. _FhGenie: A Custom, Confidentiality-Preserving Chat AI for Corporate and Scientific Use_ . 

- White, C. et al. 2024. _LiveBench: A Challenging, Contamination-Free LLM Benchmark_ . 2024. arXiv: 2406. 19314 `[cs]` . 

- Wu, C.-J. et al. 2022. “Sustainable AI: Environmental Implications, Challenges and Opportunities,” _Proceedings of Machine Learning and Systems_ (4) 2022, pp. 795–813. 

- Yao, Z., Wu, X., Li, C., Youn, S., and He, Y. 2024. “Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation,” _Proceedings of the AAAI Conference on Artificial Intelligence_ (38:17) 2024, pp. 19377–19385. 

- You, J. 2025. _How much energy does ChatGPT use?_ Epoch AI Gradient Updates (blog). Available at https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use. 2025. 

- Zhao, C., Tan, Z., Wong, C.-W., Zhao, X., Liu, H., and Chen, T. 2024. “SCALE: Augmenting Content 

   - Analysis via LLM Agents and AI-Human Collaboration,” () 2024. 

**17** 

