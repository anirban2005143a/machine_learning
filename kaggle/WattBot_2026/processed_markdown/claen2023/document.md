# **Carbon-Awareness in CI/CD** 

Henrik Claßen, Jonas Thierfeldt, Julian Tochman-Szewc, Philipp Wiesner, and Odej Kao 

Technical University Berlin, Berlin, Germany _{_ `h.classen, thierfeldt, tochman-szewc` _}_ `@campus.tu-berlin.de` _{_ `wiesner, odej.kao` _}_ `@tu-berlin.de` 

**Abstract.** While the environmental impact of cloud computing is increasingly evident, the climate crisis has become a major issue for society. For instance, data centers alone account for 2.7% of Europe’s energy consumption today. A considerable part of this load is accounted for by cloud-based services for automated software development, such as continuous integration and delivery (CI/CD) workflows. 

In this paper, we discuss opportunities and challenges for greening CI/CD services by better aligning their execution with the availability of lowcarbon energy. We propose a system architecture for carbon-aware CI/CD services, which uses historical runtime information and, optionally, userprovided information. Our evaluation examines the potential effectiveness of different scheduling strategies using real carbon intensity data and 7,392 workflow executions of Github Actions, a popular CI/CD service. Results show, that user-provided information on workflow deadlines can effectively improve carbon-aware scheduling. 

**Keywords:** continuous integration · carbon-aware scheduling · carbon intensity · sustainability 

## **1 Introduction** 

It is widely accepted that the main cause of anthropogenic climate is the release of greenhouse gases, such as carbon dioxide (CO2), into the atmosphere. The burning of fossil fuels such as coal and gas for energy production is a major contributor to the release of greenhouse gases. Globally, data centers, which are an important part of modern digital infrastructure, consumed about 205 TWh of electricity in 2018, representing about 1% of global energy consumption [1], while in the EU they accounted for 2.7% of energy consumption [2]. Many cloud providers are taking steps to reduce the environmental impact of their data centers by utilizing more renewable energy. However, the production of renewable energy sources such as solar and wind varies greatly over time and different locations, making it difficult to ensure a constant supply of energy. A common metric to quantify the operational carbon emissions of energy consumption is called _carbon intensity_ , which describes the grams of CO2-equivalent greenhouse gases emitted per killowatt-hour of consumed energy (gCO2/kWh). 



A significant portion of the work performed in cloud data centers pertains to services that enable the automated testing, building, and delivery of software, often referred to as continuous integration and delivery (CI/CD) [3]. According to a survey conducted by the CD Foundation [4], the use of CI/CD in the context of development and operation has become a standard practice today and has steadily increased. Out of 70,000 developers, about 47% reported to use CI or CD in their software engineering process, as they increase productivity, software quality, and enable faster release cycles. 

The operation of CI/CD services is a promising target for carbon-aware computing – the alignment of a computing system’s power usage with the availability of energy with low carbon intensity – as they often include recurring or non-time-critical jobs. Moreover, the execution of CI/CD workflows can be resource-intensive and thus energy-intensive, since separate virtual machines or containers are usually used for each individual job. On the other hand, many CI/CD workflows are expected to execute as fast as possible and therefore have no temporal flexibility that can be leveraged. In this paper 

- we discuss challenges and opportunities for carbon-aware CI/CD and propose a system architecture for more sustainable CI/CD services. 

- we quantify potential improvements of carbon-aware scheduling using realworld carbon intensity and CI/CD workflow execution data. 

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 discusses the main opportunities and challenges of carbonawareness in CI/CD followed by a carbon-aware system architecture in Section 4. Section 5 evaluates different scheduling scenarios. Section 6 concludes the paper. 

## **2 Related Work** 

In this section we survey related works on CI/CD job management as well as general carbon-aware scheduling approaches. 

### **2.1 CI/CD Workflow Scheduling** 

Major cloud and CI/CD service providers typically do not publicly disclose the specifics of their workflow scheduling heuristics. Jenkins, an open-source CI/CD automation server, employs a controller-agent architecture where a controller server schedules jobs to available agent nodes based on user-defined criteria, such as labels and priorities. While cloud providers keep their scheduling algorithms under wraps, the research community has been advancing cloud computing scheduling algorithms. For example, in the work of Ibrahim et al. [5] different state of the art scheduling heuristics that aim to optimize resource allocation and load balancing in cloud environments were compared and evaluated. However, to the best of our knowledge, no service provider inherently includes features for carbon-aware job scheduling at this point. 

2 



### **2.2 Carbon-Aware Computing** 

Due to an increasingly critical public perception of unsustainable business practices and the fact that carbon pricing mechanisms, such as emission trading systems or carbon taxes, are being implemented on a global scale [6], the IT industry is actively striving to increase the utilization of low-carbon energy within datacenters. Carbon-aware computing aims to reduce the emissions linked to computing by adjusting flexible workloads in terms of both timing [7–9] and geographic locations [10–12] to align with clean energy sources. 

Like the concept proposed in this paper, the majority of carbon-aware approaches aims at consuming cleaner energy from the public grid [7,8,11,13,14]. For example, Google defers delay-tolerant workloads during periods when power generation is associated with high carbon intensity [8], which describes the amount of greenhouse gas emissions per unit of consumed energy (gCO2/kWh). Although CI/CD workflows are sometimes used as exemplary use cases in evaluation scenarios [7], none of these works explicity adresses the oportunities and challenges arrising when implementing carbon-awareness in CI/CD services. 

Besides optimizing for grid carbon intensity, recent works also try to better exploit renewable energy fluctuations directly. For example, GreenSlot [15] is a scheduler that predicts the near-future availability of solar energy and schedules workload to maximize green energy consumption while also meeting job submission deadlines. Similarly, Cucumber [16] is an admission control policy which accepts low-priority workloads on underutilized infrastructure, only if they can be computed using excess energy. Zheng et al. [10] explore workload migration on underutilized data centers as a measure to reduce curtailment. 

## **3 Opportunities and Challenges** 

In this section, we discuss ways in which existing CI/CD services can be made more carbon-aware and debate what challenges arise in doing so. After that, we look into the possibilities of utilizing additional user-supplied information to further improve carbon-aware scheduling. 

### **3.1 Leveraging Temporal Flexibility vs.** **_Fail Fast_** 

The _fail fast_ concept is a fundamental concept of CI/CD. It involves providing fast feedback to developers on whether their code changes meet the requirements or not. This helps to identify issues early in the development process, which can save time and resources in the long run. 

Many carbon-aware scheduling strategies involve shifting workloads to times when there is a greener energy mix or when there is renewable excess energy available. However, as the _fail fast_ concept demands workflows to be executed as soon and as fast as possible, it strongly conflicts with one of the most important properties of jobs suitable for carbon-aware computing: Temporal flexibility. This conflict highlights the need for a careful analysis of which jobs are eligible and to 

3 



what extent carbon-aware scheduling can be implemented without compromising the speed and efficiency of the development process. Thus, a balance must be struck between these two concepts to achieve both fast feedback for developers and reduced carbon emissions. 

### **3.2 Promising CI/CD Workflows for Carbon-Aware Scheduling** 

As stated above, not every CI/CD job is viable for full emission reducing scheduling, i. e. scheduling in time and location. Identifying this subset of jobs is the important first step in making CI/CD more carbon friendly. 

- The first and most important category is made up of periodically running jobs. These are triggered by a date and time and not by a code change, such as nightly builds, integration test pipelines, or database backups. They are often not subject to the fail fast constraint: For example, the very requirement to perform a workflow _nightly_ is often to perform it _outside of business hours_ , which is usually more than half a day. The size of flexibility windows significantly impacts the potential for carbon footprint reduction [7]. 

- Second, CI/CD workflow often comprise of a multitude of interdependent jobs that are sometimes executed in parallel. Some of these steps may take multiple hours to complete, while other finish in a few minutes. Through the use of historical runtime data, carbon-aware schedulers can estimate flexibility windows for shorter tasks and leverage this information for carbonaware scheduling. 

- The third and last category is made up of jobs which are unnecessarily executed. Some code changes to a repository, e. g. changes to documentation, do not warrant the execution of, for example, unit tests. Some research has already been done in this direction [17]. Detecting these cases and aborting the workflow yields the greatest benefit because their carbon emissions can be reduced by 100%. However, identifying unnecessary CI/CD workflow executions is not the focus of this paper. 

If there are multiple datacenters available to the CI/CD service to schedule workflows on, even workflows that cannot be shifted in time have a large potential in carbon reduction through the use of carbon-aware scheduling. This potential, however, highly depends on data locality and data protection regulations of the specific workflow. 

### **3.3 Leveraging Historical Runtime Data** 

Integrating carbon awareness into existing CI/CD solutions is the most sensible way to achieve a high level of adoption. Therefore, we will now examine which information could be extracted from existing services without external input for carbon-aware scheduling. 

The first bit of information needed is to identify a job’s category (as defined above). Since a key concept of CI/CD services is automation, workflow definitions are seldom created on the fly but instead are defined in some machine 

4 



readable format. These are then passed to the CI/CD framework used to execute. These configuration files, typically in YAML format, can also be fed into the scheduler to extract information about the parallelism, dependencies as well as periodicity of the jobs. This information can then be used to try to determine the category and make decisions on the scheduling. Additionaly, configuration files may contain further execution requirements such as build system, mockup, or test data. As data transmission can also generate carbon emissions [18], it is furthermore important to inform the scheduler about where data is stored. 

Data points produced by one or more actual executions of a job can also be used to influence scheduling decisions. For example, knowing the average and maximum runtime allows to better choose a time frame for execution. Moreover, start and finish times of jobs can be used to guess dependencies between jobs which might have not been included in their definitions. The time frame between the end of one job and the start of the next earliest depending job offers flexibility for scheduling. Even if a job is not dependent on another job, the start and end times can give an indication of a possible deadline: Consider a job that is not triggered by some user activity, starting after office hours or in the night and finishing before the start of the next days office hours. In this case, it can be assumed that this job is some kind of over night build or test and the time frame deducted from this can then be used for scheduling. This is especially useful for jobs which run periodically. 

The last source of information is the current state of the CI/CD system. As service providers try to optimize their computing resources they cache container/VM images as well as required data at certain locations. Furthermore, providers have to keep spare computing resources available at different sites to guarantee fast response times (fail fast) under high system load. A scheduler can make use of this information and the learned workflow execution requirements to influence where (machine, data center, or region) a job will be executed. For example, starting a new server creates higher associated carbon costs than consolidating work on running machines. As outlined before, moving data to a desired location also consumes energy and therefore produces carbon emissions. Following, if some or all data requirements for a job execution are present at a certain data center it can be more efficient to run the job there even if the carbon footprint of the electricity is higher than elsewhere. 

### **3.4 Improving Scheduling with User-Supplied Information** 

To further increase the effect of carbon-aware scheduling, we now investigate the potential effect of user-supplied information on the scheduling. To implement this, a user interface must be provided by the CI/CD service provider. An example is provided in Listing 1.1. Furthermore, users should be incentivized to provide additional information on their workflow runs and to accept potential delays in workflow executions. 

One example of useful information that users can provide to the scheduler is runtime estimates. Especially for new workflows where historical information is not available, the cold-start problem significantly hinders the scheduler’s ability 

5 



**Listing 1.1.** Exemplary interface which allows users to provide additional information on a workflow’s duration, deadline, and allowed regions for execution. 

<mark>---</mark> **<mark>name</mark>** <mark>: CI /CD j o b s</mark> **<mark>on</mark>** <mark>: [ push ]</mark> **<mark>j o b s</mark>** <mark>:</mark> **<mark>job −a</mark>** <mark>:</mark> **<mark>runs −on</mark>** <mark>: ubuntu − l a t e s t</mark> **<mark>carbon −aware</mark>** <mark>: y e s</mark> **<mark>s t e p s</mark>** <mark>: -</mark> **<mark>name</mark>** <mark>: My f i r s t s t e p</mark> **<mark>uses</mark>** <mark>: a c t i o n s / h e l l o w o r l d @ m a i n</mark> **<mark>with</mark>** <mark>:</mark> **<mark>d u r a t i o n</mark>** <mark>: 1h</mark> **<mark>d e a d l i n e</mark>** <mark>: 3h</mark> **<mark>allowed −r e g i o n s</mark>** <mark>: [ eu− c e n t r a l −1] ---</mark> 

to perform sensible carbon-aware decisions. If users are able to provide estimates of the duration of the job we can incorporating this information into the planning process to enable adjustments to the schedule, as needed. However, it is important to consider the validity and accuracy of user information. This especially applies to the estimated runtime. It is known from high-performance computing that users more often than not tend to overestimate the runtime [19–22]. We expect the same to be true for CI/CD jobs. One possibility to counteract that is to gradually move to using historical data as it becomes available. The user should be informed of this change. 

Another highly valuable input for carbon-aware schedulers are deadlines. As described above, if this information is not provided we can only try to guess a job’s deadline based on interdependecies with other workflows. Since estimates must be conservative to avoid introducing delays or even errors, the potential carbon reductions can not be fully exploited. With a user-specified deadline, this ambiguity is removed, jobs are still guaranteed to finish on time, and the full potential can be realized. As described above, periodic jobs benefit the most from this. 

Lastly, users should also be able to specify allowed regions for scheduling, for example, to avoid the scheduling in regions that require the migration of large volumes of data or that contradict with data protection regulations. This information can is used to influence the selection of the assigned region, as shorter transfer times are taken into account. 

## **4 A Carbon-Aware CI/CD Service Architecture** 

In the following, we present a high-level systems architecture for carbon-aware CI/CD services in a distributed computing environment. We intentionally de- 

6 



scribe the concept in a a general manner and do not specify whether it is an internal part of a CI/CD framework or provided by a third party to ease the integration into existing solutions such as GitHub Actions, Jenkins, or proprietary services. We assume that the scheduler is fed on a job-by-job basis and that it can query the information outlined in Section 3 in an effective manner. 

To reduce the carbon footprint of CI/CD services, we first filter out promising jobs for carbon-aware scheduling and then plan their execution using information about previous runs, the state of the CI/CD system, and carbon intensity forecasts at different locations. The overall concept is depicted in Figure 1. When a CI/CD workflow starts, a request is made to the scheduler. The result then contains when the job should be run, the region where to run and optionally the estimated duration that was used in the scheduling process. If the job should run at a later point in time we need to store meta information about the job. 

A preprocessor filters all incoming requests, to sort out jobs which cannot or should not be processed further. Jobs cannot be processed further if it is not possible to apply carbon-aware scheduling (see Section 3) and should not be processed further if the expected carbon savings are smaller than the costs of the scheduling. Calculating this difference depends on many factors such as resource consumption of the scheduling process or length of the job. Therefore, we can not give a general answer when to apply the scheduling to a job and when not. If a user provides an estimate for the runtime of a job, the duration used in the scheduling process is a combination of this estimate and historical data. The weight of the user input decreases with increasing historical data. Additionally, 

![Figure](assets/figure_0001_page_0007.svg)**Fig. 1.** High-level systems architecture 

7 



a time buffer is added to account for unforeseen increases in the duration, e. g. a sudden increase in the code base or tests. With an increasing number of previous runs the buffer is reduced. 

The second step is the actual scheduling, which is the most important step, since the carbon reduction happens here. To estimate the carbon footprint, forecast data predicting the future carbon intensity per time and region is used. This data is fetched from an API for the time frame spanning the arrival time of the request to the (estimated) deadline of a job. An algorithm then uses this and available data of the CI/CD framework to estimate the total carbon footprint for each possible start time and region and selects the best result in terms of carbon intensity. For jobs that were previously filtered out, the currently best region in terms of carbon intensity is selected. To compare the prediction with the actual carbon emissions and provide feedback to the user, the runtime and energy consumption of a finished job is used to estimate the actual carbon intensity. 

The interface for the human user to input information about a job (e. g. estimated runtime) can be implemented by a extending existing specification formats, see Listing 1.1. This also applies to reporting results back to the user, by adding this data to CI/CD framework’s reporting mechanisms. 

## **5 Evaluation of Carbon-Aware Scheduling Strategies** 

We implemented a first prototype of our proposed architecture using Node.js and evaluate different carbon-aware scheduling strategies against it. After describing the experimental setup, we present our results, and discuss the limitations of this analysis. 

### **5.1 Experimental Setup** 

To model varying carbon intensity at different datacenters, we utilized marginal carbon intensity data provided by WattTime<sup>1</sup> over a four day period (October 12-16, 2022). Marginal carbon intensity describes the carbon intensity of the power source that would meet any new demand – also called marginal power plant. For the example, the grid’s marginal carbon intensity is very low if an additional kilowatt of demand would be produced by an otherwise shut-off wind turbine, but high, if it is generated by a gas turbine. This makes marginal carbon intensity a very meaningful metric for scheduling decisions. 

The data by WattTime contains actual and forecasted carbon intensity for 12 different regions across Europe, North America, and Australia. Each region was assumed to have unlimited resource capacity for jobs to be executed. 

To model the CI/CD service, we collected 7,392 historic workflow executions from the same time period by crawling ten popular GitHub repositories<sup>2</sup> that 

> 1 `https://www.watttime.org` 

> 2 alibaba/arthas, apache/dubbo, apache/flink, apache/spark, k3s-io/k3s, kubernetes/minikube _,_ microsoft/typescript _,_ microsoft/vscode _,_ netdata/netdata _,_ Tencent/ncnn 

8 



perform their CI/CD using Github Actions. The resulting dataset included the workflow name, start date, and runtime for each workflow execution. 

We analyzed five experiments using different scheduling strategies. For each experiment, we iterate over all jobs in the dataset in temporal order. The five scheduling strategies are 

- **Round-robin.** This naive strategy uses round-robin scheduling establishing a baseline to assess the improvements introduced by carbon-awareness. 

- **Location Shifting.** This strategy demonstrates the effectiveness of scheduling CI/CD jobs at locations of low carbon intensity. Yet unknown jobs — those for which we lack historical data to estimate the runtime — are scheduled using the round-robin method. 

- **Location** + **Time Shifting (** **_{_ 1, 3, 6** **_}_ h).** The remaining strategies show the full potential of carbon-aware scheduling by additionally enabling the scheduler to exploit temporal flexibility of jobs. This approach is tested across three cases with varying deadline buffers (1h, 3h, 6h) – information which, in practice, could be provided by users. The job’s deadline was set to the sum of the duration and the buffer, ensuring each job had the same flexibility window for rescheduling. We will abbreviate this strategy in the following section as LTS- _{_ 1, 3, 6 _}_ . 

### **5.2 Results** 

For each scenario, we report the distribution of jobs and accumulated carbon emissions over time relative to the round-robin baseline. We opted for relative results, as our dataset did not contain any energy consumption values and therefore we could not calculate carbon emissions in grams of CO2eq or similar. This means that our findings provide an indication of the relative carbon intensity of different scheduling strategies, but not an absolute measure of carbon emissions. 

In Figure 2, it can be observed that in the round-robin baseline, the number of jobs is relatively evenly distributed, except for October 15th (Saturday), when the number of jobs is reduced over the weekend. Carbon emissions show some spikes on the afternoon of October 13th, as well as the early morning of October 12th and 14th. However, there are only a few periods where there are no or very low carbon emissions. 

Since the _Location Shifting_ strategy involves no temporal scheduling, the distribution of jobs over time remained the same, but the carbon emissions in Figure 2 are noticeable lower overall. As a result, the overall carbon emissions could already be reduced by 25.31%. 

In the _Location_ + _Time Shifting_ strategy, the effect of additionally leveraging temporal flexibility becomes apparent. For LTS-1, there was a noticeable reduction in carbon emissions, with some of the spikes shifting in time, which led to an improvement of 28.59% compared to the baseline. In the cases of LTS-3 and LTS-6, the impact was even more pronounced: jobs were significantly shifted to periods with lower carbon intensity. This redistribution resulted in more concentrated job clusters, along with intermittent gaps that featured fewer or no jobs. The overall carbon emissions were reduced by 31.20%. 

9 



![Figure](assets/figure_0002_page_0010.svg)**Fig. 2.** Amount of currently running jobs and accumulated carbon emissions for all five strategies. 

### **5.3 Limitations** 

Although our data set used real-world data, we do not have the necessary information about data locality, the carbon cost of moving data, or resource utility. Furthermore, power consumption was assumed to be steady, constant, and equal for each job – meaning it only scales with its duration. This assumption simplifies the evaluation but may not reflect real-world scenarios accurately. Lastly, the data set also lacks information about sequential jobs and dependencies. Since our scheduler does not account for these cases, it would have been useful to evaluate dependency violations as well. This would provide insights into how our scheduler handles complex job dependencies and potential improvements needed in this area. Therefore, our results are to be considered rather indicative, and results in real environments may be different. 

## **6 Conclusion and Future Work** 

In this paper, we presented the idea of integrating approaches from carbon-aware computing into CI/CD services. We discussed the opportunities and challenges when doing so and evaluated different carbon-aware scheduling strategies to demonstrate their possible effectiveness in carbon footprint reduction. 

Solely relying on historical runtime data for estimating the job duration and immediately allocating jobs to the region with the best window already resulted in a 25.31 % reduction in overall carbon emissions. A key observation was that the effective distribution of jobs across multiple regions with varying carbon intensities was a significant factor in reducing emissions. Building on this, our complete 

10 



scheduling approach introduces user input for estimated durations and deadlines, giving jobs the flexibility to be rescheduled in more carbon-efficient time windows. When we employed this strategy, the improvement further increased by an additional 3.28 – 5.89 %, bringing the total improvement to approximately 31.2 %. It is important to clarify that our test data did not accurately represent real-world user input; the estimated durations were closely aligned with actual runtimes. Thus, we anticipate even greater improvements when accounting for the inaccuracies commonly found in real user input. 

As a next step, we will investigate the practical implications of different carbon-aware CI/CD scheduling by testing them on the internal CI/CD service of a large industrial company. We anticipate that real-life deployments with global knowledge about the number and types of jobs offer additional opportunities as, for example, workflow duration and deadline estimates can be performed in a more sophisticated manner. On the other hand, we expect new challenges in terms of data locality, resource constraints, and interdependencies between workflows which limit scheduling flexibility. Lastly, we plan to simulate different seasons and datacenter locations through the use of a carbon-aware computing testbed [23] to give us a better picture of when and where there is substantial potential for savings. 

## **Acknowledgments** 

We sincerely thank WattTime for providing us with access to their marginal carbon intensity data. This research was supported by the German Ministry for Education and Research (BMBF) as Software Campus (grant 01IS17050). 

## **References** 

1. E. Masanet, A. Shehabi, N. Lei, S. Smith, and J. Koomey, “Recalibrating global data center energy-use estimates,” _Science_ , vol. 367, no. 6481, pp. 984–986, 2020. 

2. F. Montevecchi, T. Stickler, R. Hintemann, and S. Hinterholzer, _Energy-efficient Cloud Computing Technologies and Policies for an Eco-friendly Cloud Market. Final Study Report_ . LU: Publications Office of the European Union, 2020. 

3. M. Meyer, “Continuous Integration and Its Tools,” _IEEE Software_ , vol. 31, no. 3, pp. 14–16, 2014. 

4. CD Foundation, “State of Continuous Delivery Report: The Evolution of Software Delivery Performance,” 2022. 

5. M. Ibrahim, S. Nabi, A. Baz, H. Alhakami, M. S. Raza, A. Hussain, K. Salah, and K. Djemame, “An in-depth empirical investigation of state-of-the-art scheduling approaches for cloud computing,” _IEEE Access_ , vol. 8, pp. 128282–128294, 2020. 

6. W. Bank, “State and trends of carbon pricing 2022,” tech. rep., Washington, DC: World Bank., 2022. 

7. P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, and L. Thamsen, “Let’s wait awhile: How temporal workload shifting can reduce carbon emissions in the cloud,” in _ACM Middleware_ , 2021. 

11 



8. A. Radovanovic, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy, D. Xiao, M. Haridasan, P. Hung, N. Care, S. Talukdar, E. Mullen, K. Smith, M. Cottman, and W. Cirne, “Carbon-aware computing for datacenters,” _IEEE Transactions on Power Systems_ , 2022. 

9. G. Fridgen, M.-F. K¨orner, S. Walters, and M. Weibelzahl, “Not all doom and gloom: How energy-intensive and temporally flexible data center applications may actually promote renewable energy sources,” _Business & Information Systems Engineering_ , vol. 63, no. 3, 2021. 

10. J. Zheng, A. A. Chien, and S. Suh, “Mitigating curtailment and carbon emissions through load migration between data centers,” _Joule_ , vol. 4, no. 10, 2020. 

11. Z. Zhou, F. Liu, Y. Xu, R. Zou, H. Xu, J. C. Lui, and H. Jin, “Carbon-aware load balancing for geo-distributed cloud services,” in _International Symposium on Modelling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS)_ , 2013. 

12. F. Moghaddam, R. Farrahi Moghaddam, and M. Cheriet, “Carbon-aware distributed cloud: multi-level grouping genetic algorithm,” _Cluster Computing_ , vol. 18, pp. 477–491, 2015. 

13. W. A. Hanafy, Q. Liang, N. Bashir, D. Irwin, and P. Shenoy, “CarbonScaler: Leveraging cloud workload elasticity for optimizing carbon-efficiency,” in _ACM SIGMETRICS / IFIP Performance_ , 2024. 

14. L. Lin, V. M. Zavala, and A. Chien, “Evaluating coupling models for cloud datacenters and power grids,” in _ACM e-Energy_ , 2021. 

15. I. Goiri, M. E. Haque, K. Le, R. Beauchea, T. D. Nguyen, J. Guitart, J. Torres, and R. Bianchini, “Matching renewable energy supply and demand in green datacenters,” _Ad Hoc Networks_ , vol. 25, pp. 520–534, 2015. 

16. P. Wiesner, D. Scheinert, T. Wittkopp, L. Thamsen, and O. Kao, “Cucumber: Renewable-aware admission control for delay-tolerant cloud and edge workloads,” in _International European Conference on Parallel and Distributed Computing_ , 2022. 

17. R. Abdalkareem, S. Mujahid, E. Shihab, and J. Rilling, “Which Commits Can Be CI Skipped?,” _IEEE Transactions on Software Engineering_ , vol. 47, no. 3, pp. 448– 463, 2021. 

18. M. Ficher, F. Berthoud, A.-L. Ligozat, P. Sigonneau, M. Wissl´e, and B. Tebbani, “Assessing the carbon footprint of the data transmission on a backbone network,” in _Conference on Innovation in Clouds, Internet and Networks (ICIN)_ , 2021. 

19. W. Cirne and F. Berman, “A comprehensive model of the supercomputer workload,” in _4th IEEE International Workshop on Workload Characterization_ , 2001. 

20. D. Tsafrir, Y. Etsion, and D. G. Feitelson, “Backfilling Using System-Generated Predictions Rather than User Runtime Estimates,” _IEEE Transactions on Parallel and Distributed Systems_ , vol. 18, no. 6, pp. 789–803, 2007. 

21. W. Tang, Z. Lan, N. Desai, and D. Buettner, “Fault-aware, utility-based job scheduling on Blue, Gene/P systems,” in _IEEE CLUSTER_ , 2009. 

22. W. A. Ward, C. L. Mahood, and J. E. West, “Scheduling Jobs on Parallel Systems Using a Relaxed Backfill Strategy,” in _Job Scheduling Strategies for Parallel Processing_ , Springer, 2002. 

23. P. Wiesner, I. Behnke, and O. Kao, “A testbed for carbon-aware applications and systems,” _arXiv:2306.09774 [cs.DC]_ , 2023. 

12 

