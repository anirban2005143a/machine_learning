# Generalizable Machine Learning Models for Predicting Data Center Server Power, Efficiency, and Throughput 

Nuoa Lei<sup>1</sup> , Arman Shehabi<sup>1</sup> , Jun Lu<sup>2</sup> , Zhi Cao<sup>3</sup> , Jonathan Koomey<sup>4</sup> , Sarah Smith<sup>1</sup> , Eric Masanet<sup>1,5</sup> 

1 Energy Technologies Area, Lawrence Berkeley National Laboratory, Berkeley, CA, USA 

2  Division of Epidemiology and Biostatistics, University of Illinois, Chicago, IL, USA 

3 College of Environmental Science and Engineering, Nankai University, 38 Tongyan Road, Jinnan District, Tianjin, 300350, China 

4 Koomey Analytics, Bay Area, CA, USA 

- 5 Bren School of Environmental Science and Management, University of California Santa Barbara, Santa Barbara, CA, USA 

## Highlights 

- Server models: power, performance-to-power ratio, maximum throughput. 

- Models based on configuration, workload level, and hardware availability date. 

- Accurate and generalizable machine learning based server models. 

- Analysis of important predictive factors. 

- Investigated reliability in modeling prospective servers. 

## Abstract 

In the rapidly evolving digital era, comprehending the intricate dynamics influencing server power consumption, efficiency, and performance is crucial for sustainable data center operations. However, existing models lack the ability to provide a detailed and reliable understanding of these intricate relationships. This study employs a machine learning-based approach, using the SPECPower_ssj2008 database, to facilitate user-friendly and generalizable server modeling. The resulting models demonstrate high accuracy, with errors falling within approximately 10% on the testing dataset, showcasing their practical utility and generalizability. Through meticulous analysis, predictive features related to hardware availability date, server workload level, and specifications are identified, providing insights into optimizing energy conservation, efficiency, and performance in server deployment and operation. By systematically measuring biases and uncertainties, the study underscores the need for caution when employing historical data for prospective server modeling, considering the dynamic nature of technology landscapes. Collectively, this work offers valuable insights into the sustainable deployment and operation of servers in data centers, paving the way for enhanced resource use efficiency and more environmentally conscious practices. 

## Keywords 

Server, data center, power models, power efficiency, machine learning 

1. Introduction 



Data centers are foundational to the sustenance of contemporary digital life, facilitating an array of digital services and supporting the burgeoning digital economy. The exponential growth in artificial intelligence and data-driven applications has precipitated a marked surge in the data processing, storage, and communication services provided by these centers [1–3]. This upward trajectory is poised to persist, raising concerns about the escalating consumption of resources and its consequential environmental implications [4–8]. 

The resource consumption of computing data centers encompasses the energy used by servers, external storage devices, network equipment, and supporting infrastructure, as well as water used in cooling systems and the generation of offsite electricity generation [4,9]. Within this context, the electricity consumption of servers emerges as a pivotal factor influencing the sustainability of these facilities. Servers, integral to data centers, play a vital role in providing essential data, resources, and computing services. These power-hungry machines consistently draw substantial electrical power to uphold reliable data center operations [10]. Crucially, the electricity consumed by servers is converted into heat, resulting in a notable overhead in resource consumption, covering both electricity and water, for data center cooling systems and offsite electricity generation [9]. In 2018, global computing data centers were estimated to consume approximately 205 𝑇𝑊ℎ of electricity, with servers accounting for 103 𝑇𝑊ℎ of this consumption and an additional 61 𝑇𝑊ℎ dedicated to cooling these servers and covering power distribution loss [1]. The U.S. computing data centers in 2014 reported an electricity consumption of around 71 𝑇𝑊ℎ and a water usage of 639 billion liters [11], with servers directly consuming about 30 𝑇𝑊ℎ of electricity and cooling systems using an additional 22 𝑇𝑊ℎ, alongside about 552 billion liters of water for cooling and electricity generation purposes [11]. These figures underscore the significant resource footprint attributable to servers within computing data center operations. 

To address the escalating challenges of resource efficiency and sustainability in data centers, professionals, including operators, engineers, energy analysts, and policymakers, rely on modeling tools. These tools are designed to gauge resource consumption accurately and make informed decisions on efficiency-enhancing technologies, strategies, and policies [11–13]. Numerous prior studies have proficiently and reliably modeled the energy and water usage of data center infrastructure [9,14–17], delivering nuanced insights into the intricate connections between resource consumption and factors such as cooling system technologies, free cooling strategies, climate conditions, equipment efficiencies, and operational settings. These insights are invaluable for advancing sustainable data center operations. Similarly, extensive research efforts have been dedicated to developing server models for tasks such as load prediction [18,19], cooling system sizing [20,21], virtual power metering [22,23], dynamic voltage and frequency scaling-based energy conservation [24–27], thermal-aware data center design and control [21,27–29], and workload optimization [30–32], underscoring the utility of employing server models to improve data center energy efficiency. 

Despite the importance of servers and numerous modeling efforts, a notable gap remains in understanding and modeling server power usage and efficiency (refer to section 2 for details). Criticisms from recent review papers highlight the lack of user-friendliness and generalizability in publicly available server models [33,34]. Some models, while accurate, are limited to a single or small subset of servers [19,21,22,24– 26,28,30,32,35] or rely on hard-to-acquire operational (e.g., run-time signals from server operations) or circuit-level data [18,27,36–38]. Other models offer wider applicability at the expense of accuracy due to oversimplified assumptions [20]. Furthermore, our updated literature review indicates an overemphasis on power usage modeling, leaving the modeling of server power efficiency and performance relatively unexplored. Moreover, earlier models fall short in depicting variations in server power and efficiency associated with diverse configurations and temporal effects. The limitations inherent in existing server models impede decision-makers from tailoring general and systematic approaches to accommodate the anticipated growth in data center resource demand, as these models lack the ability to provide a detailed 



and reliable understanding of the intricate relationships between server power consumption, efficiency, and performance, considering a spectrum of technological, operational, and temporal factors. 

This paper addresses these knowledge gaps by introducing a machine learning-based server modeling approach, leveraging the SPECPower_ssj2008 database [39]. Our method demonstrates exceptional accuracy and generalizability in predicting server power consumption, efficiency (i.e., performance-topower ratio), and maximum throughput, without sacrificing user-friendliness. Through comprehensive analysis, we explore how technological, operational, and temporal factors influence server performance metrics. We conclude with a cautionary note on prospective server modeling and suggest avenues for future research in this domain. 

## 2. Previous work 

This section offers an overview of previous server models, outlining their key modeling approaches and objectives, and critically analyzing their strengths and weaknesses. 

### 2.1. Complementary metal-oxide semiconductor circuit power model 

Early research endeavors focused on estimating server power usage utilizing the complementary metaloxide semiconductor (CMOS) circuit power model [33,37,38], a well-established method for gauging the power consumption of digital circuits. This model delineates dynamic and static power as the primary components of power consumption in digital circuits, elucidating the relationship between power consumption and circuit parameters such as transistor count, circuit current, supply voltage, and switching capacitance. 

While effective at the circuit level, the CMOS circuit power model faces notable challenges when applied to estimating the power consumption of server systems. A key challenge is its requirement for intricate, often proprietary, circuit-level technological details, which are typically unavailable in the public domain. More critically, the validity and accuracy of the CMOS circuit power model are compromised at the server level, where higher-level system abstraction complicates the behavior of server power usage [33]. 

These challenges inherent in the CMOS circuit power model have spurred the evolution of two main streams of modeling approaches: data-driven approaches and hybrid modeling approaches. 

### 2.2. Data-driven server power models 

Data-driven approaches derive server models solely from experimental data or hardware specifications. Several researchers have employed both linear and non-linear regression analyses to capture the relationship between server power usage and CPU utilization [22–24,32,40,41]. A prominent study by Horvath et al. [24] introduced a linear power model achieving an R-squared value of 98%, using CPU utilization and clock frequency as parameters to explore how dynamic voltage scaling could optimize server energy consumption. Alternatively, Dhiman et al. [18] demonstrated the effectiveness of a classification approach for server power prediction, utilizing a Gaussian Mixture Model based on instructions per cycle, memory accesses per cycle, cache transitions per cycle, and CPU utilization, achieving an average prediction error of less than 10%. 

While these models can achieve high accuracy in estimating server power use, their applicability is confined to a specific server or a narrow selection of servers they were trained on, necessitating retraining for accuracy across different server types. In an effort to improve the generalizability of server power prediction 



models, Cheung et al. [20] incorporated server-specific information (such as rated processor speed, number of processors, and the maximum power output of the power supply unit (PSU)) into a linear regression model. This model predicted idle and maximum power across various servers, which was subsequently extrapolated to estimate server power consumption at different CPU utilization levels, assuming ideal power proportionality. However, Cheung’s model has limitations, notably that the maximum power output of the PSU is an outcome of server design rather than a predictive input for server design, and the model’s accuracy (mean absolute percentage error of 25.7%) raises concerns for practical use. Additionally, the assumption of perfect power proportionality may not accurately reflect the power behavior of diverse servers. 

### 2.3. Hybrid server power models 

Beyond data-driven approaches, hybrid modeling approaches have gained prominence in the realm of server power modeling. In a hybrid power model, the server’s power consumption is typically divided into multiple additive components, each representing the power consumption of a major server component. This approach blends experimental data with theoretical insights, often drawing on principles from the CMOS circuit power model. When certain server components are not considered in the power modeling, a bias term is used to compensate for potential discrepancies in power prediction. Hybrid modeling approaches offer the advantage of allowing modelers to incorporate specific parameters of interest from physics-based knowledge. For example, researchers have modeled server power usage by focusing on the dynamic power usage of the CPU [25,30]. They imposed a cubic relationship between CPU power and operating frequency and incorporated a bias term to compensate for power usage in other server components. Other studies also modeled server power consumption by considering the CPU and server fan power use [19,21,28], utilizing insights from the CMOS circuit power model and fan curve. This leads to server models that can facilitate thermal-aware data center design. Researchers like Xu et al. [26] conducted power modeling by examining the power consumption of CPUs and disks, considering the impact of CPU frequencies, CPU utilization, and disk utilization. This resulted in a framework that enables power conservation in data centers through dynamic voltage and frequency scaling. Similarly, Ge et al. [31] developed a power model for multicore servers by focusing on CPU and memory power consumption, which concurrently accounts for the impact of workload features, parallel processing across nodes and cores, and processor frequency scaling. More recently, Arroba et al. [27] focused on modeling the power usage of the CPU, memory, and server fan, leading to a server model supporting both thermal-aware data center design and energy savings through dynamic voltage and frequency scaling. Notably, the accuracy of the hybrid model can potentially be improved by incorporating additional server components into the modeling scope. For example, Basmadjian et al. [36] decomposed server power usage into six additive components (CPU power, memory power, hard disk power, network interface card power, fan power, and power loss due to the PSU), resulting in a power model with an error of 2-10%. In a similar vein, Song et al. [35] considered various server components such as CPU, memory, disk, network interface card, and network peripherals, achieving an average error of 3.5% when predicting server power usage. 

While hybrid modeling approaches rival the effectiveness of data-driven techniques with lesser data requirements, they are not without limitations. First, they typically assume that server power is the linear sum of component power, overlooking the complex interplay between components. Second, during the modeling stage, restrictive parametric relationships informed from physical knowledge are often imposed, which may not accurately represent the ground truth due to oversimplification or changes in applicable conditions. Lastly, akin to previous data-driven models, the precision of hybrid approaches is confined to the specific servers they are calibrated for, limiting their broader applicability in server configuration optimization or early-phase data center power demand forecasting. 



### 2.4. Performance and power-performance models 

In contrast to the considerable focus on server power modeling, only a limited number of studies have delved into modeling server performance or power efficiency (i.e., power-performance ratio). Lin et al. [32] developed models using regression analysis for server performance and power efficiency as functions of CPU utilization, reporting average errors of 0.32% for performance and 3.28% for power efficiency. Ge et al. [31] utilized Amdahl’s law and employed execution time as a performance metric to analyze the speedup achieved by multicore parallel processing. Song et al. [35] also utilized workload execution time as a performance metric, breaking it down into additive components, including computation time, main memory access time, storage access time, and communication overhead, with each component modeled as a regression function of problem size, parallelism, and hardware frequency. 

While these models provide valuable insights for enhancing power efficiency and optimizing workloads in data centers, they face several important challenges. First, similar to server power modeling, previous performance and efficiency modeling approaches are restrictive to a small set of servers, and different model parameters need to be fitted for different servers to maintain model accuracy. Second, previous models do not offer the capability to understand how server throughput and power-performance may vary across diverse server configurations, nor have they been tested for servers equipped with newer generations of technology. Third, previous performance metrics that rely solely on execution time fail to offer insights into the maximum throughput of servers and do not provide information on energy consumption on a perworkload basis. 

### 2.5. Proposed server models 

This study focuses on three primary modeling objectives: server power consumption (𝑃!, 𝑊), maximum %%&_()% %%&_()% throughput (𝑇ℎ"#$, ~~)~~ , and performance-to-power ratio (𝑃𝑒𝑟𝑓!, ~~)~~ . In our settings, server power % * consumption refers to the power utilized by servers at a specific workload level (𝐿, %). Maximum throughput quantifies the number of server workload<sup>1</sup> operations per second when the server operates at full capacity. The performance-to-power ratio is computed as the server’s throughput divided by its power consumption at a given workload level, serving as a metric for the server’s power efficiency [39]. The relationship between 𝑃!, 𝑃𝑒𝑟𝑓!, and 𝑇ℎ"#$ can be expressed by the Eq. (1) below: 



$$
𝑃𝑒𝑟𝑓! = !\times{}{},-!"\# .$ (1)
$$

Our approach distinguishes itself from previous studies in several aspects. Firstly, our study extends the scope of server modeling by simultaneously focusing on power consumption, maximum throughput, and performance-to-power ratio. We identify the intricate relationships of these model targets with server configuration, workload, and hardware availability year, which can be useful for data center design and optimization. Specifically, our modeling approach can reliably predict power consumption and performance-to-power ratio across various server workloads, providing insights into server power proportionality [42] and efficiencies at different workloads. Secondly, our server models, which leverage robust and reliable machine learning algorithms, outperform existing models in terms of generalizability. This is evident in their high accuracy on both observed and unobserved servers, a notable improvement compared to existing models that are confined to a single server or a small subset once trained. Finally, inputs to our server models can be easily and publicly obtained. Our approach does not need run-time 

> 1 The workload is a standardized Java program designed to assess the performance of the CPU(s), caches, memory, shared memory processors, Java Virtual Machine implementations, Just In Time compilers, garbage collection, threads, and specific aspects of the operating system in the System Under Test [39]. 



measurements from sensors, nor does it require circuit-level or detailed technology information to make predictions, greatly improving the applicability of the models. 

## 3. Data and methodology 

In the following sections, we present the data sources and the research methodology used in this study, elucidating how machine learning techniques are applied to model server performance and power consumption. 

### 3.1. Data source 

This study leverages the SPECpower_ssj2008 database [39] to demonstrate the efficacy of machine learning techniques in server modeling. The SPECPower_ssj2008 database stands as an industry-standard benchmark for volume servers, providing crucial insights into server configuration details, power use, and performance characteristics. This database encompasses servers from leading manufacturers, featuring hardware availability dating back to 2004, as depicted in Fig. 1. As of December 2023, the database includes 949 servers, each with 11 observations at various server workload levels ranging from 0% to 100% and is consistently updated on a quarterly basis. 

The publicly accessible SPECPower_ssj2008 database provides a moderate sample size, continuous observations over time, and detailed server characterization information. These attributes make it an valuable resource for employing a machine learning approach in server modeling and generating insightful analyses. However, several biases may limit its applicability: (1) vendors often optimize servers specifically for the SPECpower_ssj2008 benchmark, potentially creating the illusion of overly efficient servers; (2) it focuses exclusively on a Java-based workload benchmark, neglecting other common data center tasks like web hosting, database management, and machine learning training or inference; and (3) tests are performed in controlled environments, disregarding real-world factors such as varying temperatures and dynamic workloads. 

Nevertheless, this study emphasizes the feasibility of applying a machine learning approach to server modeling while exploring the impact of server characteristics on power use and efficiency. The proposed framework is designed with flexibility, allowing for updates and expansions as new data or external databases become available. 

![Figure](assets/figure_0001_page_0006.svg)Fig. 1. Distribution of server samples by hardware availability year in the SPECpower_ssj2008 database. 



### 3.2. Data preprocessing 

Data preprocessing is a pivotal initial step in building machine learning models, as it profoundly influences the capability of machine learning algorithms to extract meaningful information from a dataset. This study’s data preprocessing involves two essential steps: (1) data cleaning and unit conversion, and (2) feature selection. 

In the first step, data cleaning and unit conversion, we excluded extraneous information, such as hardware vendors and disclosure uniform resource locators (URLs), from the raw data. Additionally, multi-node servers were excluded to concentrate our analysis on the impact of server configuration on power, maximum throughput, and performance-to-power ratio. Textual data were transformed into either numeric or categorical variables using predefined rules from the SPECpower_ssj2008 result file fields [43]. Additionally, we normalized the measurement units of data features to ensure consistency across all observations (Table 1). 

The subsequent step, feature selection, was guided by the principle of maintaining model user-friendliness and practical application. Specifically, features were carefully chosen to include only those variables that are typically provided in manufacturers’ data sheets or accessible in the public domain. This approach guarantees that model users can rely on readily accessible data for predictive analytics, thereby facilitating informed decision-making. Consequently, the model incorporates features spanning three server configuration categories: central processing unit (CPU), memory, and storage drive. Additionally, server hardware availability date and server workload level were included to monitor server technology development over time and investigate server power proportionality. Collectively, these features offered insights into the technological, temporal, and operational impacts on server power, maximum throughput, and performance-to-power ratio. Table 1 summarizes the feature categories and corresponding server features considered in this study. 

<u>Table 1. Sever features for server modeling in this study.</u> 

|**Category**|**Sever feature (abbreviations)**|**Note**|
|---|---|---|
|**Central**|Chip count (CC)|CC is the number of chips in the server.|
|**processing unit**|Cores per chip (CPC)|CPC is the number of cores per chip.|
|**(CPU)**|Threads per core (TPC)|TPC is the number of hardware threads per<br>core.|
||Clock frequency (CF)|CF is the nominal clock frequency of the CPU<br>(𝑀𝐻𝑧).|
||L1 data cache size (CS-L1D)|CS-L1D is the CPU’s primary data cache,<br>normalized to the average size per core~~(~~<br>/0<br>1(23<br>),<br>it is a numeric variable derived from textual<br>data in the raw dataset.|
||L1 instruction cache size (CS-<br>L1I)|CS-L1I is the CPU’s primary instruction<br>cache, normalized to the average size per core<br>~~(~~<br>/0<br>1(23<br>), it is a numeric variable derived from<br>textual data in the raw dataset.|
||L2 cache size (CS-L2)|CS-L2 is the CPU’s secondary cache<br>(including data and instruction cache),<br>normalized to the average size per core~~(~~<br>40<br>1(23<br>),<br>it is a numeric variable derived from textual<br>data.|





||L3 cache size (CS-L3)|CS-L3 is the CPU’s tertiary cache (including<br>data and instruction cache), normalized to the<br>average size per chip~~(~~<br>40<br>1-5)<br>~~)~~, it is a numeric<br>variable derived from textual data.|
|---|---|---|
|**Memory**|DIMM count (MMC)|MMC is the number of memory modules in<br>the server, it is a numeric variable derived<br>from textual data.|
||DIMM size (MMS)|MMS represents the memory size (in𝐺𝐵) per<br>module, it is a numeric variable derived from<br>textual data.|
|**Storage**|Disk drive count (DDC)|DDC is the count of disk drives in the server,<br>it is a numeric variable derived from textual<br>data.|
||Disk drive size (DDS)|DDS represents the storage capacity (in𝐺𝐵)<br>per disk, it is a numeric variable derived from<br>textual data.|
||Disk drive type (DDT)|DDTs include hard disk drive (HDD) and<br>solid-state drive (SSD). It is a categorical<br>variable derived from textual data, which was<br>subjected to one-hot-encoding [44] prior to<br>model training.|
|**Others**|Hardware availability date (HAD)|HAD represents the datetime when all<br>hardware components of the server were<br>available, and it was converted to the Proleptic<br>Gregorian Ordinal format [45] before model<br>training.|
||Workload level (L)|𝐿was only used to model𝑃!and𝑃𝑒𝑟𝑓!, not<br>𝑇ℎ"#$.|



### 3.3. Data splitting and modeling pipeline 

The preprocessed dataset was subjected to data splitting and a meticulously structured modeling pipeline before resulting in the final server models, as illustrated in Fig. 2. 

This study designed two distinct data splitting schemes to address different scopes of server modeling. The first scheme focuses on modeling servers with existing technologies, aligning with the scope of most previous studies. In this scheme, the preprocessed dataset is randomly divided into training, validation, and testing sets, representing 80%, 10%, and 10% of the samples, respectively. Importantly, the dataset is split based on the server index to prevent data leakage [45], ensuring that machine learning algorithms infer server power curves (i.e., server power at different workload levels) solely based on server configuration and hardware availability date. Consequently, the models can reliably predict server power curves without relying on specific power usage data under other workloads from the same server. 

The second data splitting scheme is specifically crafted to assess the predictability of power consumption, maximum throughput, and performance-to-power ratio for future servers. This investigation holds substantial importance for forward-looking analysis, a dimension often overlooked in previous studies. To achieve this, we implemented a time-series-based data splitting approach, dividing the preprocessed dataset into training, validation, and testing sets based on hardware availability date (detailed in Appendix B). 



The split data were subsequently input into a dedicated modeling pipeline (Fig. 2). This pipeline is crafted to encompass various essential steps, including missing data imputation, feature normalization, model training with hyperparameter tuning, and model selection from models built using candidate machine learning algorithms. Specifically, the k-nearest neighbors (KNN) [46] was employed to impute missing values in the dataset, where the Euclidean distance metric was utilized to identify the nearest neighbors, and the missing values were filled using the average feature values of those neighbors. The standard scaler was employed to normalize the data features by subtracting the mean and scaling to unit variance, a process that can accelerate the model learning process and enhance model performance [47]. A set of candidate machine learning algorithms was selected for model training, with Bayesian optimization [48] applied to fine-tune the hyperparameters of each algorithm. The best server models, targeting power, maximum throughput, and performance-to-power ratio, were ultimately selected based on the model performance metrics on the testing set. A brief introduction to the candidate machine learning algorithms, Bayesian hyperparameter tuning, and model performance metrics can be found in Sections 3.4-3.6. The code for the implemented modeling pipeline is accessible through the link provided in Appendix A. 

![Figure](assets/figure_0002_page_0009.svg)Fig. 2. Flowchart of data splitting and server modeling pipeline. 

3.4. Machine learning algorithms 

In this study, we developed and compared server models employing a suite of candidate machine learning algorithms, including: (1) elastic-net linear regression (LR), (2) stochastic variational Gaussian process (SVGP), (3) extreme gradient boosting (XGBoost), (4) random forest (RF), and (5) artificial neural network (ANN). A concise overview of these algorithms is presented below, complemented by detailed model and code implementation accessible through the provided link in the Appendix A. Interested readers are directed to pertinent literature for a thorough understanding of these algorithms. The selection of these machine learning algorithms was driven by their varying suitability in addressing nonlinearity and their adaptability to datasets characterized by diverse scales. 

The baseline model employed for server modeling was elastic-net LR [49]. This model adeptly merges the regularization techniques of ridge and lasso regression, leading to coefficient shrinkage and variable selection. Such an integrated approach proves advantageous for addressing concerns related to overfitting and multicollinearity within the model. To facilitate a thorough comparative assessment in this research, the elastic-net LR model underwent training on both the original dataset and an augmented set with polynomial features (PFs). These PFs were specifically generated to capture nonlinear associations between 



independent and dependent variables, with the determination of feature orders accomplished through Bayesian hyperparameter optimization. 

SVGP [50] is a recently developed statistical machine learning algorithm designed to alleviate the time complexity associated with training GP models. Its innovation lies in introducing a set of inducing locations to efficiently summarize the entire dataset. SVGP maintains the robust performance of GP models when facing complex function inference and noisy observations. Particularly noteworthy is SVGP's demonstrated improvement in scalability for large datasets, making it a promising choice for our server modeling task, especially given the anticipated growth in the size of training datasets provided by server manufacturers in the future. 

XGBoost [51] is an ensemble machine learning algorithm based on decision trees, and it has garnered widespread popularity in the data science community due to its cost-effectiveness during training and its high predictive performance. Employing "gradient boosting" [52], XGBoost addresses model bias by sequentially training trees to fit model residuals, resulting in a robust final-stage model. Users have the flexibility to optimize the model by adjusting tree depth and regularization parameters, effectively reducing variance. Moreover, multiple studies have affirmed the superior performance of XGBoost in modeling nonlinear data [53,54], solidifying its position as another viable candidate algorithm for this study. 

RF [55] is another ensemble learning algorithm crafted to enhance model performance through the implementation of bootstrap aggregating. In the training phase, numerous decision trees are concurrently constructed, employing random samples with replacement from the training set and selecting a random subset of features. When predicting unseen samples, RF averages the results provided by the constructed decision trees. Consequently, the RF algorithm contributes to improved model performance by mitigating model variance. 

ANNs stand as a powerful and widely adopted model in both industry and academia, finding applications across various domains, including recommender systems, advertising, fraud detection, search ranking, and speech/visual understanding [56]. ANNs are frequently employed for universal function approximation due to their outstanding features such as nonlinear modeling, fault tolerance, and the capability to handle input data in diverse formats [57]. The flexibility of ANNs in data modeling is noteworthy, as they can adapt to different data patterns by adjusting their network structure. Hence, ANN is selected as another candidate algorithm for server modeling in this study. 

### 3.5. Bayesian hyperparameter optimization 

Hyperparameters dictate the structure and learning process of a machine learning model, making their appropriate setting fundamental for constructing a robust and efficient model. In this study, Bayesian optimization was employed for automatic hyperparameter tuning due to its advantages of reduced time and computational costs, coupled with superior performance when contrasted with conventional methods like manual search, grid search, or random search [48]. 

The hyperparameter tuning problem using Bayesian optimization can be formulated as Eq. (2), and Table 2 summarizes the Bayesian optimization algorithm utilized in this study. 



$$
𝑥∗= 𝑎𝑟𝑔𝑚𝑖𝑛$𝑓(𝑥) (2)
$$

where 𝑥<sup>∗</sup> represents the optimal set of hyperparameters, 𝐻𝑃 is the search space defined in Appendix C. Table. C.1, and 𝑓(∙) denotes a probabilistic surrogate model for the objective function 𝑓, representing the model's performance on the validation dataset. 



<u>Table 2. Bayesian optimization algorithm.</u> 

- Step 𝑇= 0 (initialization): Generate initial hyperparameter samples 𝑑<sup>,78</sup> = {(𝑥<sup>,78</sup> , 𝑦<sup>,78</sup> ), ⋯}, where 𝑥<sup>,78</sup> is a random hyperparameter sample from Appendix C. Table. C.1, and 𝑦<sup>,78</sup> is the corresponding model performance on the validation dataset. 

- ● For steps 𝑇= 1 to 𝑡: (1) Build a GP regression model 𝑓(𝑥) based on 𝑑<sup>,79:;</sup> . (2) Sample the next point 𝑥<sup>,79</sup> by maximizing the “expected improvement” acquisition function [58]. 

- (3) Train the machine learning model based on the new set of hyperparameters 𝑥<sup>,79</sup> and obtain <u>𝑦</u><sup>,79</sup> <u>.</u> 

### 3.6. Model performance metrics 

The evaluation of server models' performance encompassed metrics such as root mean squared error (𝑅𝑀𝑆𝐸), coefficient of determination (𝑅<sup><</sup> ), mean absolute percentage error (𝑀𝐴𝑃𝐸), and mean arctangent absolute percentage error (𝑀𝐴𝐴𝑃𝐸), as defined in Eqs. (3) - (6). Notably, the 𝑀𝐴𝐴𝑃𝐸 metric was selected to address potential numeric issues associated with the 𝑀𝐴𝑃𝐸 metric, which might yield infinite or undefined values for zero or close-to-zero observations [59]. This choice was made to prevent hindrance in judging model performance. Subsequently, these performance metrics were computed on the training, validation, and testing datasets to quantify the accuracy and generalizability of the models. 



$$
𝑅𝑀𝑆𝐸= E\sum{}{} (?\%:?@\%)\& ' \%() B (3) 𝑅< = 1 - \sum{}{} (?\%:?@\%)\& ' \%() \sum{}{} C?\%:?D \& ' \%() (4) 𝑀𝐴𝑃𝐸= ; B \sum{}{} I ?\%:?@\% ?\% I B 57; (5) 𝑀𝐴𝐴𝑃𝐸= ; B \sum{}{} 𝑎𝑟𝑐𝑡𝑎𝑛KI ?\%:?@\% ?\% IL B 57; (6)
$$

where 𝑦5 is the 𝑖th observation, 𝑦M5 is the 𝑖the predicted value, 𝑦 is the sample average, and 𝑛 is the number of observations. 

## 4. Results and discussion 

### 4.1. Performance of modeling servers with existing technologies 

This section concentrates on evaluating the performance of server models, particularly those pertaining to existing technologies, using the first data splitting scheme introduced in Section 3.3. Fig. 3. provides a summary of the 𝑀𝐴𝑃𝐸 and 𝑀𝐴𝐴𝑃𝐸 across various server models concerning target variables and machine learning algorithms. Notably, the XGBoost-based server models consistently outperform other algorithms in predicting server power consumption, maximum throughput, and performance-to-power ratio across all selected performance metrics. Furthermore, the XGBoost-based server models exhibit strong performance on the training, validation, and testing datasets, with less than about 10% on both 𝑀𝐴𝑃𝐸 and 𝑀𝐴𝐴𝑃𝐸. This underscores the low bias and high generalization power of the resulting server models. 



![Figure](assets/figure_0003_page_0012.svg)Fig. 4. illustrates the comparison between observations and predictions obtained from the XGBoost-based server models, highlighting the robust performance and practical utility of our proposed approach. Reflecting on our approach’s success, we identify three crucial factors that offer valuable insights for energy analysts and modelers seeking to understand server energy usage and performance through similar machine learning methodologies. Firstly, the SPECPower_ssj2008 database provides an abundant dataset that empowers the model to extract reliable information, even in the presence of data noise. Secondly, the carefully selected server features outlined in Section 3.2 demonstrate their ability to reasonably explain the variances in the target variables. Lastly, the optimized XGBoost-based server models exhibit effectiveness in capturing the nonlinear and complex relationships between the inputs and model targets. It is noteworthy that the model’s performance has the potential for improvement with additional data and server features, given the quarterly data updates to the SPECPower_ssj2008 database. However, acquiring additional server features necessitates enhanced reporting from manufacturers. For instance, supplementary details about server configuration, such as PSU efficiency [10] and operating temperature [29], have the potential to elevate the accuracy of future server modeling. 



![Figure](assets/figure_0004_page_0013.svg)4.2. Important features for modeling servers with existing technologies 

To gain additional insights from the trained XGBoost-based models focused on modeling servers with observed technologies, we utilized the permutation feature importance technique [55,60], which identifies and ranks important input variables based on their predictive power. The defined predictive power is measured by the decrease in the model performance metric, such as 𝑅<sup><</sup> , when a single feature value is randomly shuffled. Fig. 5. illustrates the results of the permutation feature importance generated by the XGBoost models. 

The XGBoost models predominantly depend on a few crucial variables for making predictions. It’s essential to acknowledge that excluding less important model features may impact model accuracy. Nevertheless, readers have the flexibility to omit lower-ranked features based on their availability and accuracy requirements for specific prediction purposes. 

![Figure](assets/figure_0005_page_0013.svg)Fig. 5. Permutation feature importance for predicting: (a) power consumption (𝑃!, 𝑊), (b) maximum %%&_()% %%&_()% throughput (𝑇ℎ"#$, ), and (c) performance-to-power ratio (𝑃𝑒𝑟𝑓!, ~~)~~ (Note: Refer to Table 1 % * for the nomenclature). 

Furthermore, the analysis highlights the crucial role of temporal and operational factors in server modeling, as indicated by the high rank of HAD and L in Fig. 5. Firstly, the server power at maximum throughput has 



visibly increased with HAD, though it demonstrates notable variations among servers with different chip counts (Fig. 6 (a)). Secondly, the interplay between HAD and L reveals nuanced interaction effects on server power consumption. This intricacy is portrayed in Fig. 6 (b), illustrating the scaled power of servers operating at different L values. Noteworthy is the observation that, over time, servers tend to become more power-proportional, displaying reduced idle power consumption and a more pronounced linear relationship between server power and L. Thirdly, the influence of HAD and L on server performance-to-power ratio and maximum throughput is evident, as illustrated in Fig. 6 (c) and (d). Both the performance-to-power ratio and maximum throughput show an upward trend with increasing HAD, and a positive correlation is observed between the performance-to-power ratio and L. Specifically, while CPC proves to be the more influential input for predicting maximum throughput, it's noteworthy that the maximum CPC available in servers also rises with HAD, underscoring the significant temporal effects of advancements in server technology. Collectively, the importance of L in server modeling reaffirms the established practice of incorporating it in modeling server power consumption [33,34]. Additionally, the elevated significance of both L and HAD emphasizes the critical role of considering server virtualization technologies and hardware refresh cycles for achieving energy-efficient data center operations [1,4,61]. This insight provides operators with the knowledge to make informed decisions, whether optimizing efficiency through server virtualization or upgrading servers with newer technologies to achieve enhanced energy savings and reduced electricity bills. 

Moreover, power consumption, maximum throughput, and performance-to-power ratio are significantly influenced by server configurations. Specifically, MMC and CC are key features for predicting power consumption and maximum throughput, CPC is crucial for predicting performance-to-power ratio and maximum throughput, while CS-L3 is consistently important for predicting power consumption, maximum throughput, and performance-to-power ratio. The importance of these server features aligns with fundamental design principles for computing systems: (1) higher MMC or CC contributes to a larger overall chip area and increased thermal budget, leading to higher maximum throughput and power consumption limit (Fig. 6 (e)) [62]; (2) multicore processors facilitate parallel execution of CPU instructions, enhancing performance-to-power ratio and maximum throughput [63]; and (3) the L3 cache is designed to expedite information flow between main memory and CPU, improving performance-to-power ratio and maximum throughput with additional power usage [64]. 

Finally, compared to CPU- and memory-specific features, storage-specific features have a less significant role in predicting power consumption, performance-to-power ratio, and maximum throughput. However, we observed a notable impact of DDT on server idle power consumption (Fig. 6 (f)). Servers with SSDs exhibit lower idle power consumption than those with HDDs. While this aligns with previous studies, it's crucial to note the increasing preference for SSDs in newer server generations, where temporal effects also contribute to the observed difference in idle power between the two groups. 



![Figure](assets/figure_0006_page_0015.svg)Fig. 6. Relationship between key model features and target variables: (a) server power at maximum throughput vs. hardware availability year and chip count; (b) scaled server power vs. workload level and hardware availability year; (c) performance-to-power ratio vs. hardware availability year and workload level; (d) maximum throughput vs. hardware availability year and cores per chip; (e) server power at maximum throughput vs. DIMM and Chip count; (f) server idle power vs. disk drive type and hardware availability year (Note: The hardware availability year is derived from the aggregation of HAD at the 



yearly level; the uncertainty bounds in (a) depict the 95% confidence interval for the smoothed trend lines; the servers in (b) represent those with the median dynamic range for each hardware availability year). 

### 4.3. The danger of prospective prediction 

Fig. 7. illustrates the performance of modeling prospective servers on the testing dataset using the timeseries-based data splitting approach (Appendix B), where the 𝑀𝐴𝐴𝑃𝐸 values are generated based on computer experiments with different baseline years and forecast horizons beyond the baseline years. The performance quality of the server models exhibits a rapid reduction with the extension of forecast horizons, resulting in 𝑀𝐴𝐴𝑃𝐸 averages exceeding 20% for forward predictions beyond 2 years. In fact, the 𝑅<sup><</sup> statistics on the testing datasets quickly drop below zero for forward predictions beyond 2 years (refer to OSM), indicating that the prediction results of the server models are less accurate than using the target mean. In this context, it becomes apparent that employing the machine learning approach based on the SPECpower_ssj2008 database for forward-looking server analysis and predictions may not be suitable. This holds particular significance in the dynamically evolving information and communication technology industry, characterized by unpredictable technological breakthroughs [65]. Hence, energy analysts and policymakers should approach forward-looking analysis with caution, recognizing that predictions based on historical information are susceptible to substantial errors and uncertainties. Nonetheless, further scientific investigations and research, coupled with the acquisition of comprehensive data and insights pertaining to server technologies, are imperative to fortify the robustness and reliability of prospective analyses and predictions. 

![Figure](assets/figure_0007_page_0016.svg)Fig. 7. Modeling performance (𝑀𝐴𝐴𝑃𝐸 on testing dataset) of prospective servers across different target variables (Note: The reported results pertain to the performance of the best models. Readers can find performance metric results related to 𝑀𝐴𝑃𝐸, 𝑅𝑀𝑆𝐸 and 𝑅<sup><</sup> in the OSM). 

## 5. Conclusions and future work 

This paper investigated the viability of a machine learning approach for modeling server power consumption, performance-to-power ratio, and maximum throughput using the SPECPower_ssj2008 database. The moderately-sized, detailed, and standardized SPECPower_ssj2008 database provides prerequisite data support for server modeling and analysis. The user-friendly features extracted from the database facilitated the development of a systematic machine learning approach, incorporating data imputation, feature scaling, Bayesian hyperparameter optimization, and model selection. The proposed machine learning approach effectively captures intricate relationships between model targets, server 



configuration, workload level, and hardware availability date, demonstrating consistently low 𝑀𝐴𝑃𝐸 and 𝑀𝐴𝐴𝑃𝐸 values below about 10% when modeling servers with existing technologies. The permutation feature importance method identifies the most predictive parameters in server modeling, emphasizing the significance of technological, operational, and temporal factors. This research further investigated the reliability of prospective server modeling, uncovering significant errors and uncertainties in forward predictions. 

The resulting XGBoost models excel in predicting server power consumption, maximum throughput, and performance-to-power ratio across diverse server component specifications and workload levels. This capability is invaluable for optimizing servers in terms of power consumption, cooling budget, power proportionality, throughput, and energy efficiency. Additionally, it holds significant importance for designing and optimizing data centers, covering aspects such as capacity planning, load predictions, cooling arrangements, workload coordination, and operational controls. Furthermore, these models offer clear insights into the relationship between servers' energy consumption, performance, and efficiency with the hardware availability year. This illuminates opportunities for data center energy conservation and cost reductions through optimized hardware refresh cycles and server procurement strategies. Moreover, this study delved into the predictability of the machine learning approach for prospective server modeling, offering complementary insights to previous research. It illuminates the biases and uncertainties inherent in model predictions based on the SPECPower_ssj2008 database. The findings serve as a cautionary note for energy analysts and policymakers engaging in prospective analyses that lean on historical information. Such analyses should be approached with caution to ensure the reliability of the results. 

This study can be extended by future work from several aspects. First, while this study relies on the SPECpower_ssj2008 database for model training and validation, external datasets can be leveraged to evaluate the applicability of the resulting model to servers that differ significantly from those in the SPECpower_ssj2008 database. Additionally, the features used in this study are limited to general server specifications, such as CPU, memory, storage, hardware availability dates, and workload levels. Future server models could benefit from integrating additional features encompassing aspects such as operating temperature [29,66], the nature of computing workloads (e.g., artificial intelligence (AI), video streaming, social media) [3,67], the type of server accelerator cards (e.g., graphics processing unit, field-programmable gate array, application-specific integrated circuit, AI accelerators) [67,68], and circuit-level technological details [37]. Incorporating such features can provide a more tailored and flexible modeling framework, addressing specific objectives and leveraging available data for diverse applications. Furthermore, the SPECpower_ssj2008 database is created based on servers operating in controlled environments. Future data gathered from real-world operational settings can serve as a critical supplement to explore factors such as workload variability, cooling strategies (e.g., liquid- or air-cooled servers), and power consumption dynamics under fluctuating conditions. Moreover, upcoming research endeavors should focus on exploring new datasets and methodologies to enhance prospective server modeling. This will contribute to refining the accuracy and reliability of predictions in the evolving landscape of server technologies. Finally, future research endeavors should prioritize the modeling of storage and network devices [1,4,11], offering the potential to expand the scope of this study and provide a comprehensive depiction of electricity usage and energy efficiencies across various categories of information technology equipment within data centers. 

## Acknowledgement 

The authors express gratitude for the support from Leslie and Mac McQuown, which contributed to part of the research conducted at Northwestern University. 



The authors gratefully acknowledge support from the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy, Industrial Efficiency and Decarbonization Office. Lawrence Berkeley National Laboratory under Contract No. DE-AC02-05CH11231 with the U.S. Department of Energy. 

## Appendix 

Appendix A. Online supplementary materials (OSM) 

Supplementary data and code related to this article can be accessed via request. 

Appendix B. Time-series-based data splitting 

The time-series-based data splitting scheme is tailored for modeling prospective servers. In this approach, the preprocessed data is organized chronologically based on hardware availability date. Subsequently, the dataset is partitioned into training, validation, and testing sets with respect to a chosen baseline year. For data preceding the baseline year, the initial 80% is designated as the training set, while the remaining 20% is reserved for validation to optimize hyperparameters. The resulting optimized models, generated from candidate machine learning algorithms, undergo evaluation on testing sets, which correspond to different years beyond the selected baseline year. This evaluation aids in selecting candidate machine learning models, where the performance of modeling prospective servers is gauged by the best models derived from the model selection process based on the testing data of specific years beyond the baseline year. Notably, this research conducted computer experiments covering various baseline years (2010 to 2022) and different years (1 to 5) beyond the baseline year to comprehensively quantify the performance of modeling prospective servers. 

![Figure](assets/figure_0008_page_0018.svg)Appendix C. Hyperparameters and search space 

Table. C.1. Hyperparameters and search space for candidate machine learning models. 

|Model|Hyperparameter|Range|Probability<br>distribution|
|---|---|---|---|
|Elastic-net<br>LR|The elastic-net mixing parameter|[0, 1]|Continuous uniform|
|Elastic-net|The elastic-net mixing parameter|[0, 1]|Continuous uniform|
|LR (PFs)|The maximum degree of the<br>polynomial features|[1, 4]|Continuous uniform|
|SVGP|Numberof inducinglocations|[30, 256]|Integeruniform|





||Kernel function|{𝑘<sup>2EF</sup>, 𝑘<sup>4;/<</sup>, 𝑘<sup>4H/<</sup>, 𝑘<sup>4I/<</sup>}<br>1|Multinomial<sup>2</sup>|
|---|---|---|---|
||Learningrate|[10<sup>:I</sup>, 10<sup>8</sup>]|Logarithmic uniform|
|XGBoost|Subsample ratio of columns<br>when constructing each tree|[0, 1]|Continuous uniform|
||Subsample ratio of the training<br>instance|[0, 1]|Continuous uniform|
||Maximumdepthof a tree|[1, 10]|Integeruniform|
||Numberofboostingrounds|[1000, 20000]|Integeruniform|
||L1 regularization term on<br>weights|[0, 10<sup>H</sup>]|Continuous uniform|
||L2 regularization term on<br>weights|[0, 10<sup>H</sup>]|Continuous uniform|
||Learningrate|[10<sup>:I</sup>, 10<sup>8</sup>]|Logarithmic uniform|
|RF|Subsample ratio of columns<br>when constructing each tree|[0, 1]|Continuous uniform|
||Subsample ratio of columns for<br>each level|[0, 1]|Continuous uniform|
||Subsample ratio of columns for<br>each node split|[0, 1]|Continuous uniform|
||Maximumdepthof a tree|[1, 10]|Integeruniform|
||The size of the forest to be<br>trained|[1000, 2000]<br>|Integer uniform|
||L1 regularization term on<br>weights|[0, 10<sup>H</sup>]|Continuous uniform|
||L2 regularization term on<br>weights|[0, 10<sup>H</sup>]|Continuous uniform|
||Learningrate|[10<sup>:I</sup>, 10<sup>8</sup>]|Logarithmic uniform|
|ANN<sup>3</sup>|Hidden layers|<br>[0, 5]|Integeruniform|
||Hidden nodes|[10, 200]|Integeruniform|
||Dropout rate|[0.05, 0.3]|Continuous uniform|
||Learningrate|[10<sup>:I</sup>, 10<sup>8</sup>]<br><br>|Logarithmic uniform<br><br>|
|Note:<sup>1</sup> 𝑘<sup>2E</sup>|<sup>F</sup>, 𝑘<sup>4;/<</sup>, 𝑘<sup>4H/<</sup>, and𝑘<sup>4I/<</sup>denotes|radial-basis, Matérn<br>;<br><<br>, Matérn<br>H<br><|, and Matérn<br>I<br><<br>kernel,|



respectively (each kernel has an additional additive white noise kernel to account for the white noise contribution) [50,69];<sup>2</sup> Each event has an equal probability of occurring;<sup>3</sup> In our ANN implementation, the number of hidden nodes and dropout rate are consistent in each hidden layer, and <u>rectified linear unit (ReLu) activation function [70] was used.</u> 

Appendix D. 𝑅<sup><</sup> and 𝑅𝑀𝑆𝐸 of modeling servers with observed technologies 



![Figure](assets/figure_0009_page_0020.svg)Fig. D.1. 𝑅<sup><</sup> of trained server models by target variables and machine learning algorithms (Note: The best server models are indicated by the Magenta color; The results do not show models with negative 𝑅<sup><</sup> values). 

![Figure](assets/figure_0010_page_0020.svg)References 

- [1] Masanet E, Shehabi A, Lei N, Smith S, Koomey J. Recalibrating global data center energy-use estimates. Science (80- ) 2020;367:984–6. https://doi.org/10.1126/science.aba3758. 

- [2] IEA. Digitalization & Energy. Paris: IEA; 2017. 

- [3] CISCO. Forecast and Methodology, 2016–2021. Cisco Glob Cloud Index 2018. https://virtualization.network/Resources/Whitepapers/0b75cf2e-0c53-4891-918eb542a5d364c5_white-paper-c11-738085.pdf (accessed July 7, 2021). 

- [4] Lei N, Masanet ER. GLOBAL DATA CENTER ENERGY DEMAND AND STRATEGIES TO CONSERVE ENERGY. Data Cent Handb Plan, Des Build, Oper a Smart Data Cent 2021:15–26. https://doi.org/10.1002/9781119597537.CH2. 

- [5] IEA. Electricity 2024 – Analysis and forecast to 2026 2024. https://www.iea.org/reports/electricity-2024 (accessed February 5, 2024). 



- [6] Koomey JG. Worldwide electricity used in data centers. Environ Res Lett 2008;3:034008. https://doi.org/10.1088/1748-9326/3/3/034008. 

- [7] Koomey J. Growth in Data Center Electricity use 2005 to 2010. Anal Press 2011:1–24. 

- [8] Malmodin J, Lundén D. The Energy and Carbon Footprint of the Global ICT and E&M Sectors 2010–2015. Sustain 2018, Vol 10, Page 3027 2018;10:3027. https://doi.org/10.3390/SU10093027. 

- [9] Lei N, Masanet E. Climate- and technology-specific PUE and WUE estimations for U.S. data centers using a hybrid statistical and thermodynamics-based approach. Resour Conserv Recycl 2022;182:106323. https://doi.org/10.1016/j.resconrec.2022.106323. 

- [10] Barroso LA, Clidaras J, Hölzle U. The datacenter as a computer: An introduction to the design of warehouse-scale machines, second edition. 2013. https://doi.org/10.2200/S00516ED2V01Y201306CAC024. 

- [11] Shehabi A, Smith SJ, Sartor DA, Brown RE, Herrlin M, Koomey JG, et al. United States Data Center Energy Usage Report. Berkeley Lab 2016:65. 

- [12] Brown R, Masanet E, Nordman B, Tschudi B, Shehabi A, Stanley J, et al. Report to Congress on Server and Data Center Energy Efficiency: Public Law 109-431. Berkeley, CA: 2007. 

- [13] Geng H. Data Center Handbook: Plan, Design, Build, and Operations of a Smart Data Center. 2nd ed. Palo Alto, California, USA): John Wiley and Sons; 2021. https://doi.org/10.1002/9781119597537. 

- [14] Lei N, Masanet E. Statistical analysis for predicting location-specific data center PUE and its improvement potential. Energy 2020;201:117556. https://doi.org/10.1016/j.energy.2020.117556. 

- [15] Lei N, Lu J, Cheng Z, Cao Z, Shehabi A, Masanet E. Geospatial assessment of water footprints for hyperscale data centers in the United States. J Phys Conf Ser 2023;2600:172003. https://doi.org/10.1088/1742-6596/2600/17/172003. 

- [16] Gozcu O, Ozada B, Carfi MU, Erden HS. Worldwide energy analysis of major free cooling methods for data centers. Proc. 16th Intersoc. Conf. Therm. Thermomechanical Phenom. Electron. Syst. ITherm 2017, 2017. https://doi.org/10.1109/ITHERM.2017.7992592. 

- [17] Lee KP, Chen HL. Analysis of energy saving potential of air-side free cooling for data centers in worldwide climate zones. Energy Build 2013;64:103–12. https://doi.org/10.1016/J.ENBUILD.2013.04.013. 

- [18] Dhiman G, Mihic K, Rosing T. A system for online power prediction in virtualized environments using Gaussian mixture models. Proc - Des Autom Conf 2010:807–12. https://doi.org/10.1145/1837274.1837478. 

- [19] Garraghan P, Thompson H, Al-Anii Y, Kapur N, Summers J, Djemame K. A unified model for holistic power usage in cloud datacenter servers. Proc - 9th IEEE/ACM Int Conf Util Cloud Comput UCC 2016 2016:11–9. https://doi.org/10.1145/2996890.2996896. 

- [20] Cheung H, Wang S, Zhuang C, Gu J. A simplified power consumption model of information technology (IT) equipment in data centers for energy system real-time dynamic simulation. Appl Energy 2018;222:329–42. https://doi.org/10.1016/J.APENERGY.2018.03.138. 

- [21] Ham SW, Kim MH, Choi BN, Jeong JW. Simplified server model to simulate data center cooling energy consumption. Energy Build 2015;86:328–39. https://doi.org/10.1016/J.ENBUILD.2014.10.058. 

- [22] Wu W, Lin W, Peng Z. An intelligent power consumption model for virtual machines under CPUintensive workload in cloud environment. Soft Comput 2017;21:5755–64. https://doi.org/10.1007/S00500-016-2154-6/FIGURES/6. 

- [23] Lien CH, Bai YW, Lin MB. Estimation by software for the power consumption of streamingmedia servers. IEEE Trans Instrum Meas 2007;56:1859–70. https://doi.org/10.1109/TIM.2007.904554. 

- [24] Horyath T, Skadron K. Multi-mode energy management for multi-tier server clusters. Parallel Archit Compil Tech - Conf Proceedings, PACT 2008:270–9. https://doi.org/10.1145/1454115.1454153. 

- [25] Mootaz Elnozahy EN, Kistler M, Rajamony R. Energy-efficient server clusters. Lect. Notes 



Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 2325, Springer Verlag; 2003, p. 179–97. https://doi.org/10.1007/3-540-36612-1_12. 

- [26] Xu X, Teramoto K, Morales A, Huang HH. DUAL: Reliability-aware power management in data centers. Proc - 13th IEEE/ACM Int Symp Clust Cloud, Grid Comput CCGrid 2013 2013:530–7. https://doi.org/10.1109/CCGRID.2013.82. 

- [27] Arroba P, Risco-Martín JL, Zapater M, Moya JM, Ayala JL, Olcoz K. Server Power Modeling for Run-time Energy Optimization of Cloud Computing Facilities. Energy Procedia 2014;62:401–10. https://doi.org/10.1016/J.EGYPRO.2014.12.402. 

- [28] Zapater M, Ayala JL, Moya JM, Vaidyanathan K, Gross K, Coskun AK. Leakage and temperature aware server control for improving energy efficiency in data centers. Proc -Design, Autom Test Eur DATE 2013:266–9. https://doi.org/10.7873/DATE.2013.067. 

- [29] Jin C, Bai X, Zhang X, Xu X, Tang Y, Zeng C. A measurement-based power consumption model of a server by considering inlet air temperature. Energy 2022;261:125126. https://doi.org/10.1016/J.ENERGY.2022.125126. 

- [30] Chen Y, Das A, Qin W, Sivasubramaniam A, Wang Q, Gautam N. Managing server energy and operational costs in hosting centers. Proc. 2005 ACM SIGMETRICS Int. Conf. Meas. Model. Comput. Syst., New York, NY, USA: ACM; 2005, p. 303–14. https://doi.org/10.1145/1064212.1064253. 

- [31] Ge R, Feng X, Cameron KW. Modeling and evaluating energy-performance efficiency of parallel processing on multicore based power aware systems. IPDPS 2009 - Proc 2009 IEEE Int Parallel Distrib Process Symp 2009. https://doi.org/10.1109/IPDPS.2009.5160979. 

- [32] Lin W, Wang W, Wu W, Pang X, Liu B, Zhang Y. A heuristic task scheduling algorithm based on server power efficiency model in cloud environments. Sustain Comput Informatics Syst 2018;20:56–65. https://doi.org/10.1016/J.SUSCOM.2017.10.007. 

- [33] Dayarathna M, Wen Y, Fan R. Data center energy consumption modeling: A survey. IEEE Commun Surv Tutorials 2016;18:732–94. https://doi.org/10.1109/COMST.2015.2481183. 

- [34] Jin C, Bai X, Yang C, Mao W, Xu X. A review of power consumption models of servers in data centers. Appl Energy 2020;265:114806. https://doi.org/10.1016/j.apenergy.2020.114806. 

- [35] Song SL, Barker K, Kerbyson D. Unified performance and power modeling of scientific workloads. Proc E2SC 2013 1st Int Work Energy Effic Supercomput - Held Conjunction with SC 2013 Int Conf High Perform Comput Networking, Storage Anal 2013. https://doi.org/10.1145/2536430.2536435. 

- [36] Basmadjian R, Ali N, Niedermeier F, De Meer H, Giuliani G. A methodology to predict the power consumption of servers in data centres. ACM Int Conf Proceeding Ser 2012:1–10. https://doi.org/10.1145/2318716.2318718. 

- [37] Chandrakasan A, Brodersen R. Low-power CMOS design. 1998. https://doi.org/10.1109/9780470545058. 

- [38] Ng LL, Yeap KH, Goh MWC, Dakulagi V, Ng LL, Yeap KH, et al. Power Consumption in CMOS Circuits. Field-Effect Transistor [Working Title] 2022. https://doi.org/10.5772/INTECHOPEN.105717. 

- [39] Standard Performance Evaluation Corporation. SPECpower_ssj® 2008 2022. https://www.spec.org/power_ssj2008/ (accessed June 22, 2022). 

- [40] Fan X, Weber W-D, Barroso LA. Power provisioning for a warehouse-sized computer. ACM SIGARCH Comput Archit News 2007;35:13–23. https://doi.org/10.1145/1273440.1250665. 

- [41] Zhang X, Lu JJ, Qin X, Zhao XN. A high-level energy consumption model for heterogeneous data centers. Simul Model Pract Theory 2013;39:41–55. https://doi.org/10.1016/J.SIMPAT.2013.05.006. 

- [42] Varsamopoulos G, Gupta SKS. Energy proportionality and the future: Metrics and directions. Proc. Int. Conf. Parallel Process. Work., 2010, p. 461–7. https://doi.org/10.1109/ICPPW.2010.68. 

- [43] Standard Performance Evaluation Corporation. SPECpower_ssj2008 Result File Fields 2012. https://www.spec.org/power/docs/SPECpower_ssj2008-Result_File_Fields.html (accessed June 



22, 2022). 

- [44] Ashenden SK, Bartosik A, Agapow P-M, Semenova E. Introduction to artificial intelligence and machine learning. Era Artif. Intell. Mach. Learn. Data Sci. Pharm. Ind., Elsevier; 2021, p. 15–26. https://doi.org/10.1016/B978-0-12-820045-2.00003-9. 

- [45] Betebenner DW. toOrdinal: Function for Converting Cardinal to Ordinal Numbers by Adding a Language Specific Ordinal Indicator to the Number. 2022. https://cran.rproject.org/web/packages/toOrdinal/citation.html (accessed February 6, 2024). 

- [46] Troyanskaya O, Cantor M, Sherlock G, Brown P, Hastie T, Tibshirani R, et al. Missing value estimation methods for DNA microarrays. Bioinformatics 2001;17:520–5. https://doi.org/10.1093/bioinformatics/17.6.520. 

- [47] Mikolov T, Deoras A, Povey D, Burget L, Černocký J. Strategies for training large scale neural network language models. 2011 IEEE Work. Autom. Speech Recognit. Understanding, ASRU 2011, Proc., 2011, p. 196–201. https://doi.org/10.1109/ASRU.2011.6163930. 

- [48] Eggensperger K, Feurer M, Hutter F, Bergstra J, Snoek J, Hoos HH, et al. Towards an Empirical Foundation for Assessing Bayesian Optimization of Hyperparameters. BayesOpt Work @ NeurIPS 2013:1–5. 

- [49] Ogutu JO, Schulz-Streeck T, Piepho HP. Genomic selection using regularized linear regression models: Ridge regression, lasso, elastic net and their extensions. BMC Proc 2012;6:1–6. https://doi.org/10.1186/1753-6561-6-S2-S10/TABLES/1. 

- [50] Hensman J, Fusi N, Lawrence ND. Gaussian processes for big data. Uncertain. Artif. Intell. - Proc. 29th Conf. UAI 2013, 2013, p. 282–90. https://doi.org/10.48550/arxiv.1309.6835. 

- [51] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., vol. 13-17- Augu, New York, NY, USA: ACM; 2016, p. 785–94. https://doi.org/10.1145/2939672.2939785. 

- [52] Friedman JH. Greedy function approximation: A gradient boosting machine. Ann Stat 2001;29:1189–232. https://doi.org/10.1214/aos/1013203451. 

- [53] Si Z, Yang M, Yu Y, Ding T. Photovoltaic power forecast based on satellite images considering effects of solar position. Appl Energy 2021;302:117514. https://doi.org/10.1016/j.apenergy.2021.117514. 

- [54] Zhao Y, Wang Z, Shen ZJM, Sun F. Data-driven framework for large-scale prediction of charging energy in electric vehicles. Appl Energy 2021;282:116175. https://doi.org/10.1016/j.apenergy.2020.116175. 

- [55] Breiman L. Random forests. Mach Learn 2001;45:5–32. https://doi.org/10.1023/A:1010933404324/METRICS. 

- [56] Wu Y chen, Feng J wen. Development and Application of Artificial Neural Network. Wirel Pers Commun 2018;102:1645–56. https://doi.org/10.1007/S11277-017-5224-X/METRICS. 

- [57] Abiodun OI, Jantan A, Omolara AE, Dada KV, Mohamed NAE, Arshad H. State-of-the-art in artificial neural network applications: A survey. Heliyon 2018;4:e00938. https://doi.org/10.1016/j.heliyon.2018.e00938. 

- [58] Jones DR, Schonlau M, Welch WJ. Efficient Global Optimization of Expensive Black-Box Functions. J Glob Optim 1998;13:455–92. https://doi.org/10.1023/A:1008306431147. 

- [59] de Myttenaere A, Golden B, Le Grand B, Rossi F. Mean Absolute Percentage Error for regression models. Neurocomputing 2016;192:38–48. https://doi.org/10.1016/j.neucom.2015.12.114. 

- [60] Hastie T, Friedman J, Tibshirani R. The Elements of Statistical Learning. New York, NY: Springer New York; 2001. https://doi.org/10.1007/978-0-387-21606-5. 

- [61] Editorial H. Analyzing Hardware Refresh Cycles in the Data Center 2021. Horizontechnology 2021. https://horizontechnology.com/news/data-center-hardware-refresh-cycles/ (accessed July 15, 2021). 

- [62] Esmaeilzadeh H, Blem E, St. Amant R, Sankaralingam K, Burger D. Dark silicon and the end of multicore scaling. IEEE Micro 2012;32:122–34. https://doi.org/10.1109/MM.2012.17. 

- [63] Rodgers DP. IMPROVEMENTS IN MULTIPROCESSOR SYSTEM DESIGN. Conf. Proc. - 



Annu. Symp. Comput. Archit., ACM PUB27 New York, NY, USA; 1985, p. 225–31. https://doi.org/10.1145/327070.327215. 

- [64] John L. Hennessy, Patterson DA. Computer Architecture: A Quantitative Approach. 4th ed. Waltham: 2012. 

- [65] Koomey JG, Calwell C, Laitner S, Thornton J, Brown RE, Eto JH, et al. Sorry, Wrong Number: The Use and Misuse of Numerical Facts in Analysis and Media Reporting of Energy Issues. Annu Rev Energy Environ 2002;27:119–58. https://doi.org/10.1146/annurev.energy.27.122001.083458. 

- [66] ASHRAE Technical Committee 9.9. Thermal Guidelines for Data Processing Environments. 2011. [67] de Vries A. The growing energy footprint of artificial intelligence. Joule 2023;7:2191–4. https://doi.org/10.1016/j.joule.2023.09.004. 

- [68] Lei N, Masanet E, Koomey J. Best practices for analyzing the direct energy use of blockchain technology systems: Review and policy recommendations. Energy Policy 2021;156:112422. https://doi.org/10.1016/J.ENPOL.2021.112422. 

- [69] Seeger M. Gaussian processes for machine learning. Int J Neural Syst 2004;14:69–106. https://doi.org/10.1142/S0129065704001899. 

- [70] Hara K, Saito D, Shouno H. Analysis of function of rectified linear unit used in deep learning. Proc. Int. Jt. Conf. Neural Networks, vol. 2015- Septe, Institute of Electrical and Electronics Engineers Inc.; 2015. https://doi.org/10.1109/IJCNN.2015.7280578. 

