# **A Water Efficiency Dataset for African Data Centers** 

Noah Shumba 

Opelo Tshekiso Carnegie Mellon University Africa Kigali, Rwanda 

Pengfei Li 

Carnegie Mellon University Africa Kigali, Rwanda 

Rochester Institute of Technology Rochester, New York, USA 

Giulia Fanti 

Shaolei Ren 

Carnegie Mellon University Pittsburgh, Pennsylvania, USA 

University of California, Riverside Riverside, California, USA 

## **Abstract** 

### **ACM Reference Format:** 

Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren. 2025. A Water Efficiency Dataset for African Data Centers. In _ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS ’25), July 22–25, 2025, Toronto, ON, Canada._ ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3715335.3735483 

Artificial intelligence (AI) computing and data centers consume large amounts of freshwater, both directly for cooling and indirectly for electricity generation. While most attention has been paid to developed countries such as the U.S., this paper presents the first-of-its-kind dataset that combines nation-level weather and electricity generation data to estimate water usage effectiveness for data centers in 41 African countries across five different climate regions. We also use our dataset to evaluate and estimate the water consumption of inference on two large language models (i.e., Llama-3-70B and GPT-4) in 11 selected African countries. Our estimates suggest that writing a 10-page report using Llama-3-70B could consume as much as **0.66 liters** of water, while the water consumption by GPT-4 for the same task may go up to about **59 liters** . For writing a medium-length email of 120-200 words, Llama3-70B and GPT-4 could consume about **0.13 liters** and **2.9 liters** of water, respectively. All the numbers for generative model inference tasks are based on public information available in 2024, when we initially prepared the analysis. Since then, AI inference systems have improved substantially. For example, recent disclosures suggest that energy efficiency improved by more than 30x between May 2024 and May 2025 [8]. Accordingly, our 2024 estimates should be interpreted as historical reference values rather than as representative of current performance. Interestingly, given the same AI model, 9 of the 11 selected African countries consume less water than the global average, mainly because of lower water intensities for electricity generation. In particular, Libya and Tunisia have an even lower water consumption than the U.S. average, largely due to their reliance on natural gas with low water intensity. However, Ethiopia and the Republic of the Congo, whose electricity grids rely almost entirely on hydroelectric generation, exhibit substantially higher water consumption than the global average due to high reservoir evaporation. Our dataset is publicly available.<sup>1</sup> 

## **1 Introduction** 

With the rapid growth of artificial intelligence (AI) and digital services, the demand for data centers has increased substantially [14]. While data center infrastructure was historically lacking in Africa, the continent’s burgeoning digital economy has recently led to a surge in data center constructions, with a projected market growth of 50% by 2026 compared to 2021 [4]. 

Data centers are notorious for their massive energy usage and water consumption, which have raised significant concerns even in developed countries such as the U.S. [5, 23, 40]. More critically, the added pressure on local water resources is particularly acute in Africa, where many countries are already grappling with extended droughts and water scarcity challenges [46, 48]. Therefore, it is important to assess data centers’ water consumption in Africa, supporting healthy development of the data center industry for essential economic growth while ensuring responsible utilization of limited freshwater resources. While recent studies have begun to address the growing water consumption of data centers and AI computing [12, 15, 20, 23], they have predominantly focused on regions with large data center concentrations, such as the U.S. and Europe, while leaving out Africa—despite its rapid expansion of data centers and pressing challenges of water scarcity [44, 50]. 

In this paper, we address the critical gap in the literature and present a first-of-its-kind water efficiency dataset for data centers in 41 African countries across five distinct climate regions. The dataset includes hourly estimates of water usage effectiveness (WUE) for both direct and indirect water consumption over one year. We obtain these estimates by combining weather data from across Africa with the corresponding fuel mix data (i.e., the composition of energy sources in each country). Unlike prior work [41], we incorporate water stress levels and infrastrcuture inefficiences such as leakage. To the best of our knowledge, our work is the first 

## **CCS Concepts** 

• **Social and professional topics** → _Computing / technology policy_ ; 

- **Applied computing** → _Enterprise computing infrastructures_ . 

## **Keywords** 

Water Efficiency, Data Centers,Resource Usage 

1https://huggingface.co/datasets/PengfeiLi/WaterEfficientDatasetForAfricanDataCenters 

_COMPASS ’25, Toronto, ON, Canada_ 2025. ACM ISBN 979-8-4007-1484-9/2025/07 https://doi.org/10.1145/3715335.3735483 

This version of the manuscript corrects an earlier minor data processing error that affected some of our onsite water usage effectiveness (WUE) numbers. Because total water use is dominated by offsite water consumption, our overall findings remain similar, although the rankings of some countries have shifted. All other data, including the energy estimates described in Section 4, remain unchanged and were based on public sources available as of mid-2024, when the original manuscript was written. 



Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren 

COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

study on AI sustainability to analyze these factors, which strongly influence water consumption. 

To demonstrate the utility of this dataset, we estimate the water consumption of two recent large language models (LLMs), i.e., Llama-3-70B and GPT-4, across 11 selected African countries and compare their AI inference water consumption with that in the U.S. and globally. We also illustrate how these estimates compare to water scarcity in various countries. 

These insights can be important for policy development, data center site selection, and infrastructure planning. By making our dataset publicly available, we aim to inform sustainable AI deployment strategies that align with Africa’s unique environmental and resource constraints. 

## **2 Background and Methodology** 

While our dataset is partly based on the methodology and modeling of [15, 23], which study WUE and AI model water consumption with a heavy emphasis on the U.S. data centers, we also incorporate regional water stresses and infrastructure inefficiencies due to leakages in our model. 

## **2.1 Water Usage Effectiveness (WUE)** 

WUE is measured in units of L/kWh: liters of water consumed per kWh of energy used. Like [15], we do not model supply chain manufacturing because this aspect often relies on generalized, less accurate data that may not reflect the unique operational practices of individual data centers or computing workloads. The methodology in [15] provides equations for modeling _onsite WUE_ , which refers to water directly consumed/evaporated to cool down the facility for each unit of server energy consumption, and _offsite WUE_ , which is also called the electric water intensity factor and refers to indirect water consumption by the generation of electricity that supplies each unit of data center energy. Note that water _consumption_ is defined as “water withdrawal minus water discharge”, i.e., the evaporated portion of water withdrawal that may not be immediately available for reuse [36]. Data centers commonly consume 80% of their direct freshwater withdrawal (in many cases, potable water), while only about 10% of the water withdrawal is consumed by typical households and offices [13]. 

_2.1.1 Onsite Water Usage Effectiveness._ To assess onsite WUE, [15] presents an empirical model created from a commercial cooling tower by considering two configurations. The first configuration is called “ _fixed approach_ ”, which fixes the differential between wetbulb and cold water temperatures, and the second one “ _fixed cold water temperature_ ” sets a constant cold water temperature. The WUE formulas for these two configurations are as follows: 



$$
𝛾Approach = � -0.0001896 \cdot{}{} 𝑇2 𝑤+ 0.03095 \cdot{}{} 𝑇𝑤+ 0.4442 �+ , (1)
$$



$$
𝛾ColdWater = � 0.0005112 \cdot{}{} 𝑇2 𝑤-0.04982 \cdot{}{} 𝑇𝑤+ 2.387 �+ , (2)
$$

where _𝑇𝑤_ is the wet-bulb temperature in Fahrenheit and [ _𝑥_ ]<sup>+</sup> = max{0 _,𝑥_ }. Unless otherwise noted, we will focus on Equation (2) and simply refer it to as onsite WUE (i.e., _𝛾_ off = _𝛾_ ColdWater), because it is typically easier to set a fixed cold water temperature without adjustment in real systems. While the onsite WUE for a cooling 

tower can differ from other cooling methods such as air economization with water evaporation, we note that cooling towers are one of the most commonly adopted and efficient heat rejection mechanisms for data centers [6, 13], especially in hot regions like Africa. 

_2.1.2 Offsite Water Usage Effectiveness._ Electricity generation is water-intensive and must respond to demand in real-time to maintain grid stability. Thus, similar to carbon emissions associated with electricity usage, data centers are also accountable for their electricity water consumption. Technology companies (e.g., Meta) have recently begun to include indirect water consumption for electricity generation in their sustainability reports [28]. This is critical for holistically understanding the true water impact of data centers, especially in regions where the energy mix includes significant hydroelectric and/or thermal power generation with high water intensities [36, 42]. Based on [15, 36], we use the following offsite WUE formula: 



$$
𝛾off(𝑡) = � 𝑘𝑒𝑘(𝑡) \cdot{}{} 𝑤𝑘 � 𝑘𝑒𝑘(𝑡) , (3)
$$

where _𝑒𝑘_ ( _𝑡_ ) is the amount of electricity produced by energy fuel type _𝑘_ (e.g., hydroelectric, geothermal, coal) at time _𝑡_ and _𝑤𝑘_ is the corresponding water consumption or intensity factor in L/kWh. 

## **2.2 Water Leakage** 

Leakage refers to water lost during transmission to its destination [30]; in our case, the destination of interest is data centers. Existing studies quantifying data center water usage in the U.S. and Western regions typically do not model the impact of water infrastructure damage [15, 19, 24]. 

Incorporating leakage into water usage estimates is crucial for regions with aging or poorly maintained infrastructure. In many African countries, deteriorating water infrastructure contributes to substantial losses. According to [30], global water leakage rates average around 39%, while Africa-specific leakage is estimated at 46%. Some countries, such as Nigeria, experience extreme losses exceeding 61%, significantly affecting the effective water availability for data center operations. 

To account for this issue, we account for water leakage in our estimates, modeling it as a proportional loss of transmitted water for onsite cooling. Using country-level average leakage estimates from [16], we adjust onsite water consumption estimates. Specifically, to estimate L, the rate of water lost in transmission per kWh of server energy used, we multiply the baseline onsite WUE by the leakage rate _ℓ_ for a given country: 



$$
L = ℓ\cdot{}{} 𝛾on, (4)
$$

where the leakage rate _ℓ_ is taken directly from [16]. This lost volume is then added to the baseline WUE to obtain the adjusted onsite water consumption: 



$$
˜𝛾on = 𝛾on + L, (5)
$$

and our subsequent analysis uses these leakage-adjusted rates _𝛾_ ˜on. 

## **2.3 Water Stress** 

The impact of data centers’ water usage varies heavily with the level of _water stress_ in a country, which roughly measures how 



A Water Efficiency Dataset for African Data Centers 

COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

much water is required for a nation’s needs, as a function of water supply. Higher water stress levels indicate greater competition for limited water supplies, making it more challenging to justify waterintensive applications, including AI-driven data centers. 

To assess this impact, we compared our water consumption estimates with Baseline Water Stress (BWS) values across different countries [25]. BWS is defined as the ratio of total annual water withdrawals to available renewable water supply, providing a standardized measure of water scarcity. The BWS scale ranges from 0 to 5, where: 0–1 represents low stress, 1–2 low to medium stress, 2–3 medium to high stress, 3–4 high stress, and 4–5 extremely high stress. 

This study utilizes total water stress values [25], which aggregate withdrawals across domestic, industrial, and agricultural sectors. This approach captures both direct water consumption and indirect water usage from electricity generation. 

## **3 Data Collection** 

We describe how we collect the necessary data to compute onsite and offsite WUEs, respectively. 

**Table 1: Climate regions and representative countries** 

|Climate Region|Representative Countries|
|---|---|
|Rainforest|Republic of the Congo, Gabon, Rwanda|
|Savanna|Morocco, Tunisia|
|Desert|Egypt, Libya|
|Steppe|Namibia, Ethiopia|
|Mediterranean|Algeria, South Africa|



- **Weather data.** For weather data, we first identify five distinct climate regions in Africa: _Rainforest, Savanna, Desert, Steppe, and Mediterranean regions_ [22]. We then collect weather data from the countries for each climate region, consisting of hourly wet-bulb temperature, humidity, and precipitation over one year from August 23, 2023 to August 22, 2024. All the weather data are obtained from WeatherAPI [1], which collects data via ground-based weather stations and satellite imagery. We then pick the high and low extremes in terms of the average wet-bulb temperature for each region to obtain a representative range. The selected 11 representative countries are summarized in Table 1. 

- **Energy fuel mix.** We collect the energy fuel mix (i.e., the composition of energy fuels) for electricity generation in each selected country sourced from OurWorldInData [37]. Due to the lack of access to fine-grained data, we use annual granularity for estimating the offsite WUE as done in the prior literature [36]. Additionally, we need the water intensity of each fuel type in each selected country to compute offsite WUE in (3). While direct data on the water consumption of various energy fuel types for African countries is lacking, [39] studies water withdrawal and consumption throughout different stages of energy production in Africa. Thus, we use [39] to derive the average water intensity for each energy fuel type in Africa. 

![Figure](assets/figure_0001_page_0003.svg)**Figure 1: Monthly average baseline onsite WUE for desert (red) and rainforest (blue) regions.** 

Table 2 lists the African and global water intensity estimates adopted in this work; the global estimates are taken from [36, 39]. 

Since [39] reports water consumption factors for multiple cooling systems and technology types per fuel (Table 3 therein), we must select a representative cooling technology for each fuel. For most countries, we use cooling tower (CT)-based consumption factors as a conservative baseline. We select the water consumption factors ( _𝑤𝑘_ ) in Table 2 that best align with African deployments of power technologies [39], including combined-cycle gas turbine (CCGT) for gas and subcritical coal (0.042 L/kWh) [39]. For natural gas specifically, we use the combined-cycle rather than subcritical steam value because Africa’s major gas-generating countries, including Egypt [43], Algeria [9, 34], Tunisia [29], and Libya [11] operate predominantly modern CCGT plants. Similarly, we use the subcritical coal value because Africa’s coal fleet is predominantly subcritical [10, 35]. The exception to CT-based cooling is solar, where we use the photovoltaic (PV) factor of 0.023 L/kWh [39], since PV panels involve no thermal cycle or cooling system and the vast majority of Africa’s solar capacity is PV, with concentrated solar power (CSP) limited to Morocco and South Africa [7, 18]. Wind (0.001 L/kWh) similarly requires no cooling [39]. Hydroelectric intensity (5.32 L/kWh) follows the reservoir evaporation methodology in [39] and [27]. 

- **Leakage.** We obtained national leakage data from [16, 30], which contains information about global water and sanitation utilities (IBNET). By providing a standardized framework for collecting and sharing core cost and performance indicators, IBNET enables water utilities to identify leakage hotspots, for instance. 

- **Water Stress Levels.** To account for water scarcity and competition, we use baseline water stress (BWS) values from [25], a platform for assessing global water risks. 

## **4 Dataset Evaluation** 

Our final dataset provides onsite and offsite (hourly) WUE for capital cities in 41 African countries, as well as leakage. For clarity, Figure 1 illustrates the monthly averages for a few selected countries in the rainforest and desert regions. The plot clearly illustrates seasonal trends, as well as onsite WUE differences of about 12% between climate regions. Specifically, rainforest regions generally exhibit higher onsite water consumption than desert regions, as 



COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren 

**Table 2: Water consumption factors (** _𝑤𝑘_ **) by energy source, technology, and cooling system.** 

|**Energy Source**|**Technology**|**Cooling**|_𝑤𝑘_**(L/kWh)**|_𝑤𝑘_**(L/kWh)**|
|---|---|---|---|---|
||**(Africa)**|**(Africa)**|**(Africa)**|**(Global)**|
|Coal|Subcritical steam|CT|2.008|2.008|
|Natural Gas|CCGT|CT|0.795|1.580|
|Oil|Subcritical steam|CT|2.765|2.765|
|Biomass|Subcritical steam|CT|2.095|2.095|
|Nuclear|Subcritical|OT|1.515|2.120|
|Geothermal|Flash/Binary|CT|0.042|0.042|
|Solar|Photovoltaic (PV)|None|0.023|0.400|
|Wind|Onshore turbines|None|0.001|0.001|
|Hydropower|Reservoir-based|Evap.|5.320|22.000|



their warmer wet-bulb temperatures require more cooling tower evaporation to reach a target temperature. 

One downstream use for our dataset is to estimate the cost of LLM inference in different countries. We first describe our methodology for computing these estimates, followed by our results. 

## **4.1 Methodology: Estimating Water Consumption of LLM Inference** 

We also present the details of estimating water consumption by two LLMs, i.e., Meta’s Llama-3-70 B and OpenAI’s GPT-4. We use the following equations to calculate the onsite and offsite water consumption, respectively: 



$$
𝑊on = 𝛾on \cdot{}{} 𝐸 and 𝑊off = 𝛾off \cdot{}{} 𝜌\cdot{}{} 𝐸,
$$

where _𝑊_ is the water consumption, _𝛾_ is the WUE, _𝜌_ is the power usage effectiveness (PUE), _𝐸_ is the server energy consumption for AI models, and the subscript “on” and “off” denote “onsite” and “offsite” wherever applicable, respectively. To incorporate leakage, we also compute the adjusted onsite WUE: 

provided by [38] assumes a default PUE of 1.2. The PUE overhead is not needed for calculating the onsite water consumption, but should be considered when assessing the offsite water consumption. For different African countries, we consider an average country/region-wise PUE 

provided by [3]. By taking the lowest when multiple values are presented in [3], the PUE values for the 11 selected African countries are as follows: 

|**Country**|**PUE**|**Country**|**PUE**|
|---|---|---|---|
|Algeria|2.3|Namibia|2.1|
|Egypt|2.3|Republic of the Congo|2|
|Ethiopia|1.5|South Africa|1.4|
|Gabon|1.9|Tunisia|2.3|
|Libya|2.3|Rwanda|1.7|
|Morocco|2.3|||



**Table 3: Power Usage Effectiveness (PUE) by country for our 11 selected countries.** 



$$
˜𝑊on = ˜𝛾on \cdot{}{} 𝐸.
$$

Thus, to estimate the LLMs’ water consumption, we need their onsite and offsite WUEs (including the onsite WUE adjusted for leakage), energy consumption, and PUE. 

_4.1.1 WUE._ For African countries, we use the average onsite and offsite WUEs from our dataset. For the U.S. and global references, their average onsite WUEs are obtained from publicly accessible reports based on Microsoft’s U.S. average (0.55 L/kWh) and Equinix’s global average (1.07 L/kWh), which represent efficient hyperscale and colocation data centers, respectively [2, 49]. Their average offsite WUEs are acquired from the World Resource Institute report [36]. 

_4.1.2 Power Usage Effectiveness._ Power Usage Effectiveness (PUE) is a metric that assesses the energy efficiency of a data center by comparing the total energy consumed by the facility to the energy used by the computing equipment. The ideal PUE is 1.0, indicating 100% energy efficiency in computing. The inference energy estimate 

We consider Microsoft’s U.S. average PUE of 1.17 [49] and Equinix’s global average PUE of 1.42 [2] for the U.S. and global averages, respectively. 

_4.1.3 Energy consumption._ Exact LLM inference energy costs are often lacking in the public domain, especially for powerful proprietary LLMs such as GPT-4 deployed in real-world inference systems. To estimate LLM inference energy, some studies resort to a commonly cited claim that each request of the GPT model family underlying ChatGPT consumes about 10x the energy as a Google search [17], while others use GPUs’ processing capability in tera operations per second (TOPS) and the power consumption reported by manufacturers [47]. In practice, however, the actual LLM inference energy consumption depends on a variety of factors, including the hardware, service level objectives (SLOs), and system optimization [33, 45]. 



A Water Efficiency Dataset for African Data Centers 

COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

To estimate LLM inference energy consumption for a user-facing application (used by, e.g., ChatGPT), we resort to an online calculator [38] and a recent study [45]. These two sources use different methods to calculate the LLM inference energy consumption, which we describe as follows. 

_LLM inference energy estimates by [38, 45]._ The online calculator [38] estimates the inference energy consumption by various LLMs based on a transparent methodology. It first uses the energy measurements from a set of open-sourced models (mostly on Nvidia A100 GPUs) to fit an energy consumption curve in terms of the number of model parameters. For mixture-of-expert model architectures such as the one commonly believed to be used by GPT-4, a range of active model parameters are considered. The energy measurement takes into account a server’s non-GPU power attributed to the model depending on the fraction of GPU resources the model utilizes. Nonetheless, [38] only considers the token generation phase, while neglecting the prompt processing phase (i.e., processing user prompts to generate the first output token) which is also energy-intensive [32, 33, 45]. In addition, it does not consider batching and essentially models a lightly-loaded system without request contention. This can be viewed as a reference system used by industries, e.g., [32, 33] measure LLM inference energy and power consumption without batching as a reference value for energy and power provisioning, while [33, 45] use the latency measurement in such a reference system to set real SLO targets. 

On the other hand, [45] measures the actual GPU energy consumption for LLMs on enterprise-grade Nvidia DGX H100 servers. Its measurement also considers “state-of-the-practice” optimization techniques commonly used in real systems, including batching. Importantly, it considers the prompt processing phase and representative SLO targets, which are both crucial for real-world LLM deployment. 

Energy estimates assuming a fully utilized system without accounting for SLOs may not reflect the industry practice, since a fully-utilized system can lead to significant SLO violations, which are not tolerable in real-world LLM deployment, especially for commercial LLM applications such as real-time conversations that have strict SLO targets to deliver good quality of experiences [33, 45]. As a result, server resources for LLM inference are typically provisioned based on the peak demand to ensure SLOs are met at all times. In other words, the LLM inference servers may not be highly utilized under non-peak loads, resulting in a high energy consumption per request. For example, the Llama-2-70B inference for a medium-length request on H100 GPUs consumes 9.4 Wh energy without batching [33], while the inference energy consumption is still over 4.0 Wh when batching is applied under various system loads using “state-of-the-practice” optimizations (the last column in Table II) [45]. 

The measurement in [45] only includes the GPU energy consumption for a small set of open LLMs. To account for the non-GPU server energy consumption, we need to multiply the energy consumption in [45] by a factor of 1 _._ 5 ∼ 2 _._ 0 based on the server power provisioning breakdown [32]. 

While [38] and [45] use different methodologies, we note that the server-level inference energy consumption estimated by [38] (as of November 20, 2024) is generally lower than that measured 

by [45] for the same model size, assuming the LLM inference system is optimized using state-of-the-practice techniques in [45]. For example, for Llama-3-70B to write a medium-length email with 250 tokens (or about 120-200 words), [38] estimates the inference energy consumption as 2.62 Wh (after removing the PUE of 1.2 for data center overheads), whereas [45] shows the server-level energy consumption is about 10 Wh (after multiplying the value in the last column of Table III by 1.6 to account for the non-GPU server energy). 

This result might be surprising, as [38] does not consider system optimization or batching whereas [45] uses reasonable “state-ofthe-practice” optimizations including batching. Nonetheless, [38] mostly uses A100 GPUs and does not consider prompt processing energy consumption, whereas [45] uses H100 GPUs (which may be more energy-consuming than A100 GPUs for LLM inference as shown by [33]) and considers both prompt processing and token generation energy consumption. Additionally, the strict SLOs in real-world deployment prohibits the LLM inference system from being fully utilized. Thus, the LLM inference energy consumption estimated by [38] without system optimization could be even lower and still serves as a good reference point. 

_4.1.4 Energy consumption for writing a 10-page report and a mediumlength email._ For the task of writing a 10-page report,<sup>2</sup> we assume the output is 5,000 tokens and use the estimates by [38], since the energy measurement results in [45] do not include generating such long outputs using Llama-3-70B. After removing the PUE of 1.2 for data center overheads, we estimate that the energy consumption to write a 5,000-token text by Llama-3-70B and GPT-4 are 52.25 Wh and 4.66 kWh, respectively, based on the results in [38] as of November 20, 2024. Note that, due to the proprietary nature of GPT-4, [38] assumes 1,760 billion parameters for GPT-4 with a mixture-of-expert architecture based on the best-known information from various public sources. Additionally, the energy consumption estimates for models with such large model sizes are based on extrapolation. As a result, without detailed information from model owners, the energy estimates for large proprietary models may have less accuracy than for small/medium open models. 

For the task of writing a medium-length email, we assume the output is 250 tokens or about 120-200 words. For Llama-3-70B, by considering a medium-length prompt and a medium system load, we estimate the inference energy consumption as ∼ 10 Wh after multiplying the value in the last column of Table III by 1.6 to account for the non-GPU server energy [45]. For GPT-4, we estimate the inference energy consumption as ∼ 232 Wh [38]. While [45] provides more realistic estimates for shorter tasks by incorporating batching and prompt processing, its applicability is currently limited to one of the models we evaluate—specifically Llama-3-70B—and to shorter output tasks. For longer tasks, such as generating a 10-page report, we rely on [38], which offers broader model coverage, including GPT family, albeit under simplified assumptions. Using [45] where available and [38] otherwise, our approach balances methodological rigor with practical applicability, allowing consistent estimation across diverse tasks and model architectures. 

2The calculator [38] assumes 5,000 tokens for a 5-page report. Based on the token-toword ratio [31], we consider 5,000 tokens as roughly a 10-page report. 



COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren 

![Figure](assets/figure_0002_page_0006.svg)**Figure 2: Water consumption across 11 selected countries for writing a medium-length email (250 tokens) using Llama-3-70B and GPT-4, respectively.** 

Finally, we emphasize some limitations of our methodology. First, our models are themselves simplifications of the actual water usage, which has been boiled down to national levels; in reality there are substantial variations by region and time. Second, we only estimate the water usage during the model inference time, excluding the water footprint associated with idle power. Even under our modeling assumptions, it is challenging to obtain precise data on the energy fuel mix and the electricity water intensity in Africa. Moreover, the actual energy consumption of LLM inference may vary depending on the (possibly customized) optimization techniques used by real systems, particularly for the proprietary GPT-4 model. As such, our results should be regarded as first-order estimates rather than precise representations. We encourage AI model developers and data center operators to enhance transparency regarding their most recent water usage, especially in African countries facing water scarcity. 

## **4.2 Results** 

To demonstrate the utility of our dataset, we use it to estimate the water consumption of two LLMs, i.e., Meta’s Llama-3-70B and OpenAI’s GPT-4, following the method in [23]. The tasks we evaluate are to write a comprehensive 10-page report and a medium-length email, as shown by Figures 2 and 3.<sup>3</sup> Our results indicated that writing a 10-page report using Llama-3-70B and GPT-4 in Africa could consume approximately as much as **0.66 liters** and **59 liters** of water, respectively. 

When comparing the water usage of these models between various African countries, the United States and the global averages, we observed the following: First, 9 of the 11 selected African countries have a lower water consumption than the global average. In addition, Algeria, South Africa, and Tunisia even have a lower water consumption than the U.S. average. This may be surprising, as Africa is commonly viewed as a water-scarce and dry continent. The underlying reason is that the per-fuel water intensity factors 

> 3We did not evaluate additional tasks because the numbers would simply scale linearly as a function of the energy consumption of each task for each model. However, the methodology would remain the same for other tasks. 

( _𝑤𝑘_ ) applicable to Africa’s generation fleet are lower than the global averages underlying the Water Resource Institute benchmark of 4.807 L/kWh [36]. As shown in Table 2, Africa-specific _𝑤𝑘_ values differ from global averages for three key fuels: (i) natural gas, where Africa’s predominantly CCGT fleet [9, 43] yields _𝑤𝑘_ = 0 _._ 795 L/kWh versus a global average of ∼1.58 L/kWh that includes simple-cycle plants; (ii) hydropower, where Africa’s deep, high-head reservoirs yield _𝑤𝑘_ = 5 _._ 32 L/kWh [39] versus global estimates of ∼22 L/kWh [27]; and (iii) nuclear, where cooling technology differences yield 1.515 versus ∼2.12 L/kWh globally. 

Four of our selected countries (Algeria, Tunisia, Egypt, and Libya) rely primarily on CCGT gas, which has the lowest _𝑤𝑘_ among thermal fuels. Two countries (South Africa and Morocco) rely on coal ( _𝑤𝑘_ = 2 _._ 008), which is higher than gas but still below the global weighted average. Even with higher onsite WUE and PUE values typical of African data centers [3], the offsite advantage keeps total water consumption below the global reference for 9 of the 11 countries. 

Second, the results in Figs. 2 and 3 indicate a strong relationship between the electricity fuel mix and water consumption. Ethiopia and Republic of Congo are the only countries exceeding the global average, driven by their heavy reliance on hydroelectric power (95.7% and 99.6% respectively [37]), which has the highest water intensity due to reservoir evaporation [27, 39]. Countries with moderate hydro shares, such as Namibia, Rwanda, and Gabon, also approach the global average, highlighting the disproportionate impact of hydro on water usage. In contrast, gas-dominated countries (e.g., Algeria, Tunisia, and Egypt) exhibit the lowest water consumption, while coal-based systems fall in between. Overall, variation in water consumption is more strongly linked to electricity fuel mix than to climate, with hydroelectric share emerging as the primary driver. 

Regarding leakage, countries in desert regions (e.g., Libya and Egypt) exhibit higher leakage rates despite having lower overall water consumption than the global average. This highlights the role of water infrastructure inefficiencies in exacerbating water losses. Regions such as the Steppe and Savanna have elevated leakage 



A Water Efficiency Dataset for African Data Centers 

COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

![Figure](assets/figure_0003_page_0007.svg)**Figure 3: Water consumption across 11 selected countries for writing a 10 page report (5000 tokens) using Llama-3-70B and GPT-4, respectively.** 

of placing data centers in these regions.Ethiopia and the Republic of the Congo—the two countries above the global average—both exhibit low water stress levels (Baseline Water Stress (BWS) ∼<1), suggesting that their relatively high water consumption from hydroelectric generation is unlikely to pose a significant threat to local water availability. Rwanda and Gabon, which sit just below the global average due to moderate hydro shares, also have low water stress. On the other hand, several North African countries (Egypt, Libya, Tunisia) face extremely high water stress (BWS _>_ 4) despite having relatively low AI water consumption due to their gas-dominated grids. A notable case is Namibia, where water consumption is near the global average and water stress is very high (BWS 4.18); deploying water-intensive AI models there could place additional strain on an already scarce resource. 

**Figure 4: Water consumption and (scope-2) carbon emission across various African countries for writing a 10-page report using the Llama-3-70B model. “Congo” indicates Republic of the Congo.** 

To specifically examine the impact of data centers on local water resources, we then narrowed our focus to the countries in Africa with the most existing data centers [26]. Figure 6 shows a trend where countries such as South Africa and Nigeria, which have numerous data centers, also exhibit high water consumption per model inference . This could be indicative of the greater strain these data centers place on local water resources, especially in regions with high levels of water stress. Our data suggests that without strategic intervention, the expansion of data centers could exacerbate local water scarcity issues, placing additional strain on already limited water resources. Strategic placement should prioritize areas with lower water stress levels or where advanced cooling technologies can be implemented effectively. For future placement, the ideal "sweet spot" for situating data centers in Africa would be regions that demonstrate a lower water stress level coupled with moderate water consumption. This criterion suggests that data centers could be strategically placed in areas where they would exert a minimal impact on local water resources while still fulfilling operational demands. 

rates relative to their nominal onsite water consumption, suggesting that infrastructure improvements in these areas could lead to substantial water savings. More broadly, the comparison between African countries, global averages, and developed regions like the United States reveals stark disparities in leakage levels. 

_4.2.1 Correlation between Water Consumption and Carbon Emissions._ Building on our findings, we explored the relationship between water consumption and carbon emissions for the Llama-370B model for the task of writing a 10-page report. From Figure 4, we observe a tradeoff between water consumption and carbon emission, which is consistent with the findings in prior studies [20]. This prompts further attention to strike a balance between water consumption and carbon emissions to enable truly sustainable AI in African countries. 

## **5 Conclusion** 

_4.2.2 Incorporating Water Stress Levels._ Figure 5 presents a comparison of total water consumption against the existing water stress levels in the 11 selected countries, analyzing the potential impact 

This study presents the first-of-its kind large-scale evaluation of water efficiency for African data centers, providing a crucial dataset 



COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren 

![Figure](assets/figure_0004_page_0008.svg)**Figure 5: Water consumption and Water Stress levels across 11 selected countries for writing a 10 page report (5000 tokens) using Llama-3-70B and GPT-4, respectively. “Africa" indicates the African average, “Congo” indicates Republic of the Congo and “Global" indicates the global average.** 

![Figure](assets/figure_0005_page_0008.svg)**Figure 6: Water Consumption and Stress Levels in African Countries with high data center density (number of data centers in parenthesis), evaluated for writing a 10 page report (5000 tokens) using Llama-3-70B and GPT-4, respectively.** 

that accounts for both direct and indirect water consumption across diverse climatic regions. Our findings highlight significant regional variations, demonstrating that most African countries (9 of 11 selected) exhibit lower AI-related water consumption than global averages, primarily because they generate electricity from fuels with low water intensities. However, countries with hydro-dominated electricity grids—such as Ethiopia and Republic of Congo—face substantially higher water consumption due to the water-intensive nature of hydroelectric generation. Our analysis suggests that the electricity fuel mix is a stronger predictor of cross-country variation than climate region alone, which has practical implications for data center site selection and infrastructure planning. This study 

underscores the need for site-specific infrastructure planning and adaptive cooling technologies to mitigate these challenges. 

However, limitations such as the aggregated nature of water stress data and variations in cooling technologies indicate that further refinements are needed to enhance the accuracy of AIrelated water impact assessments. Moving forward, policymakers, researchers, and data center operators can prioritize sustainable water management strategies by improving infrastructure to minimize leakage, adopting water-efficient cooling systems, and ensuring AI deployment aligns with local water availability. 

## **Acknowledgments** 



COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

A Water Efficiency Dataset for African Data Centers 

This work was made possible in part by funding from the Gates Foundation. The views and opinions expressed in this study are those of the authors and do not necessarily reflect the views or positions of the sponsors. 



COMPASS ’25, July 22–25, 2025, Toronto, ON, Canada 

Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren 

## **References** 

- [1] [n. d.]. WeatherAPI. https://www.weatherapi.com/. Accessed: 30 August 2024. 

- [2] 2023. 2023 Equinix Sustainability Report. https://sustainability.equinix.com/wpcontent/uploads/2024/07/Equinix-Inc_2023-Sustainability-Report.pdf. [Accessed 29-08-2024]. 

- [3] Africa Data Centres Association. 2021. African Climate & Datacenter PUE 2021. https://africadca.org/wp-content/uploads/2022/05/PUE-ClimateHydrogen-in-African-DCs-White-paper.pdf. 

- [4] Africa Data Centres Association. 2024. Data Centres in Africa Focus Report 2024. https://africadca.org/en/data-centres-in-africa-focus-report-2024. 

- [5] Kishwar Ahmed, Mohammad A Islam, Shaolei Ren, and Gang Quan. 2014. Can Data Center Become Water {Self-Sufficient}?. In _6th Workshop on Power-Aware Computing and Systems (HotPower 14)_ . 

- [6] Equinix. 2024. Sustainability Report. https://sustainability.equinix.com/wpcontent/uploads/2024/07/Equinix-Inc_2023-Sustainability-Report.pdf. 

- [7] ESI-Africa. 2023. The global value of CSP plants in Africa. https://www.esiafrica.com/renewable-energy/solar/the-global-value-of-csp-plants-in-africa/. 

- [8] Cooper Elsworth et al. 2025. Measuring the environmental impact of delivering AI at Google Scale. _CoRR_ abs/2508.15734 (2025). https://doi.org/10.48550/arXiv. 2508.15734 

- [9] GE Vernova. 2024. Boosting Power in Algeria with the TM2500. https://www. gevernova.com/gas-power/resources/case-studies/tm2500-algeria. 38 open-cycle gas turbines at 18 Saharan sites. 

- [10] Global Energy Monitor. 2024. Medupi power station. https://www.gem.wiki/ Medupi_power_station. Accessed: 2025-03-21. 

- [11] GlobalData. 2021. Power Plant Profile: Benghazi Combined Cycle Power Plant, Libya. https://www.power-technology.com/marketdata/benghazi-combinedcycle-power-plant-libya/. CCGT owned by General Electricity Company of Libya, commissioned 2010. 

- [12] Wedan Emmanuel Gnibga, Andrew A. Chien, Anne Blavette, and Anne Cécile Orgerie. 2024. FlexCoolDC: Datacenter Cooling Flexibility for Harmonizing Water, Energy, Carbon, and Cost Trade-offs. In _Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy ’24)_ . Association for Computing Machinery, New York, NY, USA, 108–122. doi:10. 1145/3632775.3661936 

- [13] Google. 2024. Environmental Report. https://sustainability.google/reports/. 

- [14] H. C. Granade, J. Creyts, A. Derkach, P. Farese, S. Nyquist, and K. Ostrowski. 2009. McKinsey Global Energy and Materials: Unlocking Energy Efficiency in the U.S. Economy. 

- [15] Pranjol Sen Gupta, Md Rajib Hossen, Pengfei Li, Shaolei Ren, and Mohammad A Islam. 2024. A dataset for research on water sustainability. In _Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems_ . 442–446. 

- [16] IBNET. 2025. Non-Revenue Water — IBNET Toolkit. https://www.ib-net.org/ toolkit/ibnet-indicators/non-revenue-water/. Accessed: 2025-01-27. 

- [17] International Energy Agency. 2024. Electricity 2024. https://www.iea.org/reports/ electricity-2024. 

- [18] IRENA. 2023. Renewable Capacity Statistics 2023. https://www.irena.org/ Publications/2023/Mar/Renewable-capacity-statistics-2023. 

- [19] Mohammad A. Islam, Kishwar Ahmed, Shaolei Ren, and Gang Quan. 2014. Making Data Center Less “Thirsty” via Online Batch Job Scheduling. _ICAC_ (2014). 

- [20] Mohammad A. Islam, Kishwar Ahmed, Hong Xu, Nguyen H. Tran, Gang Quan, and Shaolei Ren. 2018. Exploiting Spatio-Temporal Diversity for Water Saving in Geo-Distributed Data Centers. _IEEE Transactions on Cloud Computing_ 6, 3 (2018), 734–746. doi:10.1109/TCC.2016.2535201 

- [21] Leila Karimi, Leeann Yacuel, Joseph Degraft-Johnson, Jamie Ashby, Michael Green, Matt Renner, Aryn Bergman, Robert Norwood, and Kerri L. Hickenbottom. 2022. Water-Energy Tradeoffs in Data Centers: A Case Study in Hot-arid Climates. _Resources, Conservation and Recycling_ 181 (2022), 106194. doi:10.1016/j.resconrec. 2022.106194 

- [22] Alfred Kröner, John Innes Clarke, Robert Walter Steel, David N. McMaster, Robert K.A. Gardiner, Kwamina Busumafi Dickson, Akinlawon Ladipo Mabogunje, Audrey Smedley, John F.M. Middleton, and Davidson S.H.W. Nicol. 2024. Africa. Encyclopedia Britannica. https://www.britannica.com/place/Africa Accessed: 30 August 2024. 

- [23] Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. 2024 (accepted). Making AI less “thirsty": Uncovering and addressing the secret water footprint of AI models. _Commun. ACM_ (2024 (accepted)). 

- [24] Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. 2024 (accepted). Making AI Less “Thirsty”: Uncovering and Addressing the Secret Water Footprint of AI Models. _Commun. ACM_ (2024 (accepted)). 

_Sciences_ 19, 5 (2015), 2091–2101. doi:10.5194/hess-19-2091-2015 

   - [28] Meta. 2024. Sustainability Report. https://sustainability.atmeta.com/2024sustainability-report/. 

   - [29] Mitsubishi Power. 2022. Mitsubishi Power Commences Operation of the Rades C CCGT Power Plant in Tunisia. https://power.mhi.com/regions/emea/news/ 20220906. 450 MW CCGT owned by STEG, commissioned September 2022. 

   - [30] Karel Josy Ngueyim Nono, Victor Dang Mvongo, and Celestin Defo. 2024. Assessment of non-revenue water in the urban water distribution system network in Cameroon (Central Africa). _Water Supply_ 24, 5 (04 2024), 1755–1767. doi:10.2166/ws.2024.071 

   - [31] OpenAI. [n. d.]. OpenAI API Pricing. https://openai.com/api/pricing/. 

   - [32] Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Brijesh Warrier, Nithish Mahalingam, and Ricardo Bianchini. 2024. Characterizing Power Management Opportunities for LLMs in the Cloud. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS ’24)_ . Association for Computing Machinery, New York, NY, USA, 207–222. doi:10.1145/3620666.3651329 

   - [33] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . 118–132. doi:10.1109/ISCA59077.2024. 00019 

   - [34] Power Technology. 2019. Biskra Combined-Cycle Power Plant, Algeria. https://www.power-technology.com/projects/biskra-combined-cycle-powerplant/. 1,338 MW CCGT. 

   - [35] Power Technology. 2023. The Kusile Power Station Project, South Africa. https: //www.power-technology.com/projects/kusilepowerstation/. Accessed: 2025-0321. 

   - [36] Paul Reig, Tianyi Luo, Eric Christensen, and Julie Sinistore. 2020. Guidance for calculating water use embedded in purchased electricity. _World Resources Institute_ (2020). 

   - [37] Hannah Ritchie and Pablo Rosado. 2020. Energy Mix. _Our World in Data_ (2020). https://ourworldindata.org/energy-mix. 

   - [38] Adrien Banse Samuel Rincé and Valentin Defour. 2024. EcoLogits Calculator. https://huggingface.co/spaces/genai-impact/ecologits-calculator. 

   - [39] Rocio Gonzalez Sanchez, Roman Seliger, Fernando Fahl, Luca De Felice, Taha BMJ Ouarda, and Fabio Farinosi. 2020. Freshwater use of the energy sector in Africa. _Applied Energy_ 270 (2020), 115171. 

   - [40] A. Shehabi, S. J. Smith, N. Horner, I. Azevedo, R. Brown, J. Koomey, E. Masanet, D. Sartor, M. Herrlin, and W. Lintner. 2016. United States Data Center Energy Usage Report. _Lawrence Berkeley National Laboratory, Berkeley, California. LBNL-1005775_ (2016). 

   - [41] Noah Shumba, Opelo Tshekiso, Pengfei Li, Giulia Fanti, and Shaolei Ren. 2024. A Water Efficiency Dataset for African Data Centers. https://arxiv.org/abs/2412. 03716 

   - [42] Md Abu Bakar Siddik, Arman Shehabi, and Landon Marston. 2021. The Environmental Footprint of Data Centers in the United States. _Environmental Research Letters_ 16, 6 (2021), 064017. 

   - [43] Siemens. 2018. Completion of world’s largest combined cycle power plants in record time. https://press.siemens.com/global/en/feature/completion-worldslargest-combined-cycle-power-plants-record-time. Egypt 14.4 GW CCGT megaproject. 

   - [44] Statista. n.d.. Data Center - Africa. https://www.statista.com/outlook/tmo/datacenter/africa Accessed: 2025-01-27. 

   - [45] Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, and Esha Choukse. 2025. DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency. In _IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . 

   - [46] The Water Project. [n. d.]. The Water Crisis: Poverty and Water Scarcity in Africa. https://thewaterproject.org/why-water/poverty. (Accessed on 11/25/2024). 

   - [47] Bill Tomlinson, Rebecca W. Black, Donald J. Patterson, and Andrew W. Torrance. 2024. The Carbon Emissions of Writing and Illustrating Are Lower for AI than for Humans. _Scientific Reports_ 14, 3732 (February 2024). 

   - [48] UNICEF. [n. d.]. Water crisis in the Horn of Africa. https://www.unicef.org/ documents/water-crisis-horn-africa. (Accessed on 11/25/2024). 

   - [49] Noelle Walsh. 2022. How Microsoft measures datacenter water and energy use to improve Azure Cloud sustainability. _Microsoft Azure Blog_ (April 2022). 

   - [50] World Health Organization (WHO) Regional Office for Africa. n.d.. Water. https: //www.afro.who.int/health-topics/water Accessed: 2025-01-27. 

- [25] Tianyi Luo, Robert Young, and Paul Reig. 2015. Aqueduct projected water stress country rankings. _Technical Note_ 16 (2015), 1–16. 

- [26] Data Center Map. 2024. Data Centers in Africa. https://www.datacentermap. com/africa/ 

- [27] Mesfin M Mekonnen, PW Gerbens-Leenes, and Arjen Y Hoekstra. 2015. The water footprint of electricity from hydropower. _Hydrology and Earth System_ 

