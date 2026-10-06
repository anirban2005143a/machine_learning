# **Small Bottle, Big Pipe: Quantifying and Addressing the Impact of Data Centers on Public Water Systems** 

|Yuelin Han|Pengfei Li|Adam Wierman|Shaolei Ren<sup>1</sup>|
|---|---|---|---|
|_UC Riverside_|<br>_RIT_|_Caltech_|_UC Riverside_|



### **Abstract** 

Water is a critical resource for data centers and an efficient means of cooling. However, meeting the growing water demand of data centers requires substantial peak water withdrawals, which many communities in the United States do not have the available capacity to supply, especially during the hottest days of the year. This largely overlooked peak water capacity constraint is emerging as a bottleneck for data centers and can force operators to rely on waterless, less efficient dry cooling, thereby increasing electricity demand and further stressing the power grid during summer peaks. In this paper, we focus on the direct water withdrawal of U.S. data centers for cooling and examine their impacts on public water systems. Based on public sources including government records and water utility data, our analysis indicates that, if the 2024 water use intensity persists, U.S. data centers could collectively require 697–1,451 million gallons per day (MGD) of new water capacity through 2030, comparable to New York City’s average daily supply of approximately 1,000 MGD. Under an optimistic scenario with an industry-wide compound annual water use intensity reduction by 10%, the water capacity demand decreases to 227–604 MGD, although the case of high-growth IT loads could still require enough capacity to hypothetically supply about half of New York City’s daily water demand for most of the year, excluding peak hot days. The total valuation of the new water capacity need is on the order of $10 billion, reaching up to $58 billion in the high-growth case. These impacts are highly concentrated on individual communities hosting data centers, placing additional stress on already capacity-constrained public water systems without substantial infrastructure upgrades. Finally, we provide recommendations to address the growing water capacity demand of U.S. data centers, including reporting peak water use, developing corporate-community partnerships to benefit community well-being, adopting a Water Capacity Neutral approach (colloquially “Pipe Neutral”) to allow host communities to retain limited water capacity resources, and implementing coordinated water-power planning to responsibly leverage water for peak power reduction and opportunistically utilize surplus power to mitigate peak water impacts on public water systems. 

# **1 Introduction** 

The advancement of artificial intelligence (AI) and computing technologies holds the promise of reshaping industries, accelerating scientific discovery, and offering powerful tools to address pressing societal challenges, from controlling plasma for nuclear fusion to predicting extreme river-related events for watershed resilience [1]. Yet this rapid progress is being powered by an unprecedented buildout of warehouse-scale data centers filled with power-hungry servers [2]. As these data center facilities proliferate, they may pose significant local infrastructure challenges in the communities where they operate, raising public concerns that residents may shoulder disproportionate infrastructure costs and face sustained obligations to address associated financial, environmental, and health risks [3,4]. 

A well-recognized challenge of data centers’ rapid expansion stems from their increasing electricity usage, with a single AI data center campus capable of drawing hundreds of megawatts, enough to power a mid-sized city. In the United States, the surging demand for AI is projected to drive the data center electricity use to 6.7 to 12.0% of the national total in 2028, up from 4.4% in 2023 [2]. As a result of the electricity consumption, the climate impacts of data centers’ carbon footprint [5–9] and the public health burden from their associated air pollution [4, 10, 11] have received heightened attention. Additionally, an immediate physical constraint often lies in the already strained local power infrastructure [12,13]. 

Alongside electricity, water is not just a basic necessity for human-being but also “a critical input supporting the data center and AI revolution,” as noted in a recent roundtable discussion on data centers convened 

> 1Corresponding authors: Adam Wierman (adamw@caltech.edu) and Shaolei Ren (shaolei@ucr.edu) 

1 



by the U.S. Environmental Protection Agency (EPA) [14]. Indeed, even setting aside off-site water use for electricity generation and chip manufacturing (Section 2.1), on-site evaporative cooling for data centers is widely recognized as a power-efficient approach [15, 16] and even “the most efficient means of cooling in many places” [17]. For example, compared to waterless dry cooling, evaporative cooling for a large technology company’s new data center campuses in Louisiana “reduces electricity demand by 25–35% at the same time when the grid experiences peak summer loads and regional power demand is at its highest” [16]. Notably, multiple leading technology companies have signed agreements with local water utilities to secure substantial water allocation for cooling their future data center campuses across various states [16,18–21]. 

However, despite the significant benefits of water and the often relatively modest _annual_ total, data center water use is highly concentrated during the hottest days of the year, resulting in substantial peak daily water withdrawals. For example, peak daily water demand for a large state-of-the-art data center using evaporative cooling during the summer can often exceed 1 million gallons per day (MGD) and, for some planned facilities, may even reach as high as 8 MGD [18,19,21]. 

There is no “national reservoir” that data centers can tap into. Instead, data centers predominantly rely on treated water—primarily potable—supplied by local public water systems.<sup>2</sup> Unfortunately, many public water systems in the United States are aging, fragmented, and/or financially constrained [23], thus lacking sufficient available capacity to meet the additional peak water demands of large data centers [24]. For example, when interviewed about a new data center requesting 6 million gallons per day (MGD) of water capacity—an amount that could potentially exhaust the available surplus supply in Newton County, Georgia—a representative of the county’s water authority stated that “we just don’t have the water” [25]. Without substantial upgrades, even large water utilities, such as Loudoun Water that serves over 300,000 residents and many tens of data centers in Northern Virginia [26], could be unable to meet the total water capacity needs if all data centers were to employ evaporative cooling during the summer (Appendix B). 

The high capital expenditure required to expand water capacity to meet the additional demand from data centers—as reflected by three recent water infrastructure upgrades for large technology companies costing a total of nearly $1 billion [16,18,27]—is frequently beyond the capability of many public water systems (Table 1), particularly small and medium-sized systems which are often under-resourced and together constitute nearly 99% of community water infrastructure in the United States [28]. Like power grids, public water systems are designed to safely and reliably meet the maximum demand across all pressure zones, with margins to handle extreme conditions such as prolonged heatwaves and droughts. However, fulfilling this mission is becoming increasingly challenging due to longstanding underinvestment and limited technical resources. According to the EPA surveys on drinking water and wastewater systems [28,29] and a recent third-party analysis [30], U.S. public water and wastewater systems now face a total infrastructure funding need of $1.3–3.4 trillion over the next 20 years to improve service reliability, safety, and resilience. Moreover, as highlighted in the EPA’s report to Congress in 2024 [31], water affordability has become a significant challenge nationwide, with an estimate of 12.1 to 19.2 million U.S. households lacking affordable access to water services. These factors can further constrain the ability of public water systems to expand capacity for new industry-scale demands. 

To address water capacity shortages without increasing the burden on local ratepayers, state laws often require large water users to make “pro rata” fair contributions to fund infrastructure upgrades and system expansion [32,33], sometimes in combination with corporate-community partnerships. For example, a large technology company has recently committed up to $400 million to invest in local public water infrastructure to support its new Louisiana data centers hosting AI and cloud services [16]. In another case, a water capacity request of 1.2 MGD from a data center developer (first reportedly for cooling [34] and later allegedly for fire suppression [35]) triggered major water infrastructure upgrades costing the developer over $100 million [36]. Without these upgrades, the host town in Wisconsin, which had less than 2 MGD of remaining water capacity according to the water utility’s 2024 annual report [37], would likely see operational safety margin significantly reduced for its water system. 

Even when financial resources are available, however, public water infrastructure projects can take multiple years to complete and may face further delays due to additional constraints, including engineering water rights allocation and multi-agency approvals [24,38,39]. In a recent instance, an Indiana water project, 

> 2As detailed in Section 2.3, we use the term “ _public water system_ ” to refer broadly to community water systems and related water infrastructure [22], including treated reclaimed water and wastewater treatment systems. 

2 



partly financed by the state, to add 25 MGD of new water supply capacity and expand wastewater treatment capacity by 15 MGD for an economic development district is expected to take approximately six years to complete and cost more than $1 billion, including $560 million for new water supply [40], $225 million for upgrades to existing infrastructure, and $350 million for wastewater treatment expansion [18]. As a result, a leading technology company’ data center located in that district to host AI and core products may not be able to operate at full capacity until its allocated water capacity of 8 MGD becomes available in 2031 [18,19]. 

In some cases, the availability of local and regional water sources can become a key and complex bottleneck. For example, expanding the capacity of a reservoir located on federal land for an Oregon town would require federal-level review and approval beyond the town’s jurisdiction [39], which can significantly constrain the water supply available to new large users or expansions of existing demand. 

In another instance involving a data center project in Virginia, the county record [33] indicates that the future expansion of water capacity from 2 MGD to 8 MGD to serve the data center “will limit the water supply available [to the water authority’s service area] and will impair economic and other development opportunities in those localities that are dependent on water supply.” The county further recognizes that “new water supply sources must be identified, accessed, and developed” for the water and wastewater system [33]. This requirement effectively accelerates the water authority’s long-range water source planning by decades [41]. As a result, future expansion of the data center project may face substantial delays while new water supply sources are developed, a process which itself is a “lengthy and costly project” [33]. 

Compared with power grid expansion, public water system upgrades are often equally complex and time-consuming, and in some cases even more so, as they are heavily constrained by source availability, local hydrology, water rights, regulatory oversight, and/or environmental considerations [42, 43]. For example, it can take roughly 20 years or more to search for a new water source and have it in service [32]. Moreover, many communities rely on reservoirs naturally replenished by rainfall or snowpack, which are highly sensitive to drought conditions and further constrain the development of additional reliable water supplies [24]. In addition, unlike cross-state electricity transmission, transporting water over long distances is considerably more challenging and generally not a viable option. 

Although data centers have increasingly constructed on-site (typically gas-fired) power plants for electricity to address grid capacity constraints [44,45], sourcing water from on-site groundwater wells or nearby rivers for cooling is rare (Table 2) and requires comprehensive environmental reviews, constrained site selection, and water rights approvals, thereby entailing significant business risks [24]. 

Consequently, the limited available capacity of public water systems is emerging as a critical, yet frequently underestimated, bottleneck to the rapid growth of data centers. Insufficient water capacity often compels data centers to rely on waterless, less-efficient dry cooling, which increases electricity demand and can further strain the power grid during summer peaks [15,16,46,47]. Moreover, such “unmet” water demands are often treated as “zero” water demands in public discourse, effectively overlooking or deferring the challenges of limited available capacity in many public water systems rather than proactively addressing them. Indeed, an Energy and Environmental Economics (E3) study released in December 2025 projects that water availability will become an even more crucial factor than the well-recognized power access for data center siting within the next five years [48]. 

**Contributions.** In this paper, we focus on data centers’ direct water withdrawal (also referred to as water _use_ for clarity) in the United States, which hosts the largest number of data centers in the world [49], and examine the impacts of data centers on public water systems. 

We first show that data centers use water differently than other users in two key aspects: compared to typical public water users such as residential buildings and offices, data centers have higher _consumptive_ ratios (i.e., the ratio of evaporated or consumed water to total water withdrawn) and exhibit substantially higher daily _peaking_ factors (i.e., the ratio of maximum daily use to average daily use). While the high consumptive ratio could affect regional water availability [50], the high peaking factor, often reaching 6 and sometimes exceeding 10 (Appendix A.6), poses an even greater challenge for public water systems, many of which already struggle to accommodate new water demands during the summer. Thus, even though some data centers use evaporative cooling only for a limited number of days and have low annual water usage, their capacity impacts on public water systems remain significant [18, 21, 51]. Importantly, this challenge is largely hidden from both total annual water figures and the commonly-reported annualized water usage effectiveness (WUE, defined as the ratio of total direct/on-site water consumption to total IT energy). 

3 



Next, we study U.S. data center water use for the period of 2024-2030, including total annual water withdrawal, water capacity need, and the valuation of water capacity. To substantiate our projections, we examine the 2024 sustainability reports of five leading hyperscale data center operators and seven large colocation providers (Table 2), collectively representing approximately 86% and 22% of U.S. hyperscale and colocation IT loads in 2024, respectively. We use this data to establish industry-wide baseline values for water use and consumptive ratios. While data center operators rarely disclose their peak water requirements, we review public reporting, including government records, water utility data, and planning documents, to conservatively set peaking factors for peak water demand and to inform the valuation of water capacity. In addition to the baseline scenario, we incorporate annual industry-wide WUE reductions by assuming compound annual rates of 5% and 10% under moderate and optimistic scenarios, respectively, reflecting both efficiency improvement and tightening water capacity constraints in many U.S. public water systems. 

Our analysis reveals the following key findings. 

- **Annual water withdrawal and consumption:** In 2030, under the baseline scenario, data centers could withdraw 80–150 billion gallons and consume 60–110 billion gallons annually, representing roughly 0.6–1.1% of total annual withdrawals and 4–7% of total annual consumptive use in U.S. public water systems. The higher share of consumption reflects that data centers evaporate approximately 75% of the water they withdraw, compared to roughly 12% for other public water users on average [52]. These figures could decrease by half under an optimistic scenario with substantial water use reductions. While aggregated annual totals are relatively modest, a pressing challenge arises from data centers’ large and rapidly growing peak daily water demand, which many public water systems are unable to accommodate given longstanding resource constraints and the difficulty of developing new water supplies. 

- **New water capacity demand:** Meeting the growing demand of U.S. data centers requires substantial new peak water capacity. From 2024 to 2030, U.S. data centers are projected to require 697–1,451 MGD of new water capacity under the baseline scenario, comparable in scale to New York City’s average daily supply of roughly 1,000 MGD [53]. Under an optimistic scenario with substantial water reductions, the total water capacity need drops to 227–604 MGD, although high-growth IT loads could still require enough capacity to hypothetically supply about half of New York City’s daily water demand for most of the year, except during the hottest peak days. Further, the water capacity demand is highly concentrated on individual communities hosting data centers, placing additional stress on already capacity-constrained public water systems in the absence of substantial infrastructure upgrades. 

- **Valuation of new water capacity demand:** The total valuation of new water capacity needed to support U.S. data centers through 2030 is generally on the order of $10 billion. Under the baseline scenario, valuations range from $7–28 billion in the low-growth case and $15–58 billion in the high-growth case, while the optimistic scenario reduces these figures to $2–9 billion and $6–24 billion, respectively. Although modest relative to the overall investment in the AI industry, such new water capacity expansion can be challenging for financially constrained public water systems [28, 29, 31], highlighting the critical importance of corporate–community partnerships to expand infrastructure without raising local water rates [3,16]. 

Finally, to address the growing water demand of U.S. data centers, water capacity resources must be planned proactively through multi-stakeholder collaboration rather than treated as an afterthought. Our key recommendations include: reporting peak water use to inform public water systems and decisionmakers; developing corporate-community partnerships to aligning corporate responsibility with community well-being; adopting a Water Capacity Neutral approach, or colloquially “Pipe Neutral,” that allows host communities to retain limited available water capacity for future development; and implementing coordinated water-power planning to responsibly use water as a “water battery” for peak power reduction during peak hours, while opportunistically utilizing surplus power to alleviate peak water stress on public water systems during heatwaves. Together, these strategies can enhance operational efficiency, protect local water systems, and ensure long-term community resilience while supporting data center growth and technological advancement. 

_Disclaimer: The goal of this study is to quantify and assess the potential impact of data centers on public water systems_ 

4 



_in the United States. This study is not intended to advocate for or against the construction of data centers. We do not take a position on decisions pertaining to any specific data center, including decisions that may be directly or indirectly related to water use. In addition, our estimates of water capacity valuation are intended to illustrate the scale of new water infrastructure needed to meet data center demand and should not be interpreted as the actual financial obligations of data center operators, which may vary depending on contractual arrangements, accounting practices, local rate structures, and other context specific factors._ 

# **2 Preliminaries on U.S. Data Centers and Public Water Systems** 

This section provides background on data centers and their water use, followed by a brief overview of public water systems in the United States and discussion on the emerging water capacity bottleneck for data centers. 

## **2.1 Overview of U.S. Data Centers and Water Use** 

We broadly classify data centers into the following three types according to a recent report from LBNL [2]. 

- _Hyperscaler_ : Hyperscaler data centers typically refer to large-scale facilities owned and operated by major technology companies to support massive computing workloads, often spanning multiple buildings or campuses. 

- _Colocation_ : Colocation data centers are operated by third-party providers that lease space, power, and cooling to multiple tenants (or a single large tenant), enabling organizations to outsource the physical infrastructure while retaining full control over their IT hardware. Some colocation data centers have IT loads comparable to, or even exceeding, those of hyperscale facilities. 

- _Others_ : Other types of data centers include enterprise or in-house data centers, telecommunications data centers, and small server rooms, which are often owned and operated by individual organizations to meet internal computing and networking needs. They are typically smaller and more geographically dispersed than hyperscalers. 

In the United States, as of 2024, colocation and hyperscaler data centers constitute the dominant total data center energy consumption [2]. Specifically, colocation data centers account for approximately 48%, hyperscalers represent 36-40%, and the other data centers make up the remaining small share. Fueled by accelerating AI demand, the total U.S. data center energy consumption is expected to grow at an annual rate of 13–27%, reaching 325–580 TWh in 2028 [2]. Colocation and hyperscaler data centers continue to constitute the two largest segments, with colocation accounting for a slightly higher share. In contrast, other types of data centers, including enterprise and in-house facilities, are expected to comprise a declining share of total data center energy consumption, falling to approximately 4-6% by 2028 [2]. 

**Data center water use.** There are three related concepts in water-related studies: water _withdrawal_ , water _consumption_ , and water _discharge_ . Water withdrawal refers to freshwater taken from surface or groundwater sources for use, regardless of whether it is later returned, and reflects competition for shared water resources [54]. Water consumption denotes the portion of withdrawn water that is not returned because it is evaporated or otherwise removed, thereby directly affecting downstream water availability [55]. Water discharge is defined as the difference between withdrawal and consumption and refers to the portion of withdrawn water that is immediately returned. Data centers use water both directly (Scope 1) and indirectly (Scopes 2 and 3) [56]. We discuss each in the remainder of this section. 

**Scope 1.** The direct water use of data centers is primarily associated with cooling, although humidification control can also require water. For a large data center, humidification demand alone may reach on the order of millions of gallons per year [57]. Critically, IT equipment, including servers, storage systems, and networking devices, generates heat that must be dissipated to the surrounding environment to ensure reliable operation. The cooling process typically occurs in two stages [58,59]. 

The first stage is _server_ -level cooling, which transfers heat from the IT equipment to an intermediate facility-level heat exchanger through either air-based or closed-loop liquid-based cooling, without direct water consumption. Notably, the high power density of GPU-powered AI servers often necessitates liquid cooling (e.g., direct-to-chip and immersion cooling) because of its superior heat removal capability [58,60], whereas liquid-cooled AI racks still have approximately 20% or more loads cooled by air [15,61]. 

5 



The second stage is _facility_ -level cooling, which rejects heat from the facility to the outside environment and may involve water use depending on the cooling technology employed. Common heat rejection mechanisms include: (1) wet cooling towers that primarily rely on water evaporation; and (2) air-cooled systems (dry coolers), often supplemented by direct evaporative or adiabatic cooling to reduce the peak power demand during the hottest days of the year if applicable. 

There is a fundamental tradeoff between power and water use for facility-level cooling [58], which we discuss in detail in Appendix E. Briefly speaking, evaporative cooling uses substantially more on-site water than waterless dry cooling, “but it also consumes much less energy” as noted by a large colocation provider [15]. Likewise, a leading technology company acknowledges in its sustainability report that “water is the most efficient means of cooling in many places” [17]. Notably, multiple large technology companies have signed agreements with local water utilities to secure substantial water capacity (e.g., up to 8 MGD at full buildout) for cooling their newest AI and cloud data center campuses across various states [16,18–21], with some water capacity expected to be delivered in 2031 due to required infrastructure upgrades [18]. 

More concretely, industrial disclosures from major technology companies indicate that water-cooled data centers can use 25 to 35% less electricity than air-cooled data centers [47,62], especially during the summer “when the grid experiences peak summer loads and regional power demand is at its highest” [16]. The annual energy savings from evaporative cooling are also substantial [63]. For example, in Nevada, a data center that primarily relies on non-evaporative cooling exhibits a 6.4 to 10.2% higher _annual_ average power usage effectiveness (PUE, defined as the ratio of total data center energy to IT energy) than another facility operated by the same company that relies heavily on evaporative cooling [17]. More energy savings from evaporative cooling are also observed in the monthly PUE and WUE measurements from real-world data centers, which are not often publicly available (Figure 1b) [64]. These benefits of water evaporative cooling can further translate into reductions in carbon emissions [65], public health risks [10], and indirect water use associated with electricity generation [56,66], as well as, critically, reductions in peak power demand to mitigate the stress on the grid during the summer months. 

In general, for server-level cooling, liquid-cooled high-density servers (e.g., AI servers) can typically operate at higher allowable inlet temperatures than air-cooled servers (e.g., general-purpose compute, storage, or networking equipment). This higher thermal tolerance expands the range of ambient conditions under which facility-level “free” air cooling can be utilized, thereby reducing reliance on power-intensive mechanical chillers or evaporative cooling systems for a greater portion of the year. Nonetheless, even for a leading technology company’s state-of-the-art data center under construction which uses a “water-efficient closedloop, liquid-cooled system that recirculates the same water” [19], a substantial water capacity of 8 MGD is still secured to support facility-level (evaporative) cooling during peak heat conditions [18]. 

A portion of the water withdrawn by data centers is discharged as wastewater, most commonly into municipal wastewater treatment facilities. For example, a large data center currently under construction has been allocated a water capacity of 8 MGD and a wastewater capacity of 4 MGD [18], while the first phase of another planned data center receives an initial capacity of 2 MGD of water and 0.57 MGD of wastewater [21]. Depending on the quality of the supplied water and operational settings, the required wastewater capacity generally scales linearly with the water withdrawal. Therefore, we use data center water withdrawal as a primary metric for both water and wastewater capacity requirements. 

In addition to the use of water for cooling, some data centers may directly manage fire water systems to comply with safety codes and ensure sufficient flow (typically measured in gallons per minute, GPM) during fire emergencies. For instance, one data center states that it has requested a peak capacity of 1.2 million gallons per day for fire suppression [35] in response to inquiries regarding its large peak need, which is reportedly utilized (for cooling) during the hottest days of the year [34]. Domestic water use includes needs such as restrooms and kitchens, and is typically minimal. Unless explicitly stated, we exclude both fire and domestic water from our quantitative analysis. 

**Scopes 2 and 3.** The indirect water use of data centers is associated with electricity generation (Scope 2) as well as upstream supply chain activities (Scope 3), including rare earth mining and semiconductor manufacturing. While the Scope 2 indirect water use for electricity generation is typically larger than direct water use, it is non-potable in most cases and therefore not directly relevant to public water supply [56]. The U.S. Energy Information Administration (EIA) routinely reports water withdrawal for the electric power sector, which is one of the largest freshwater-withdrawing sectors in the United States and comparable to 

6 



agriculture in total withdrawal volumes (42.5% for thermoelectric power vs. 43% for crop irrigation relative to the total water withdrawal in the contiguous United States) [52,67]. It has also recently begun publishing monthly data on water withdrawal and consumption for individual power plants [68]. Additionally, as water withdrawals can affect immediate water availability and downstream water quality, water permits are issued based on total withdrawal volumes and applicable consumptive use limits [38, 69]. Therefore, it is important to quantify Scope 2 water withdrawal and consumption in order to assess the _operational_ impacts of data centers on regional watersheds. A recent study indicates that a large technology company’s Scope 3 water accounts for over 99% of its total corporate water footprint [70]. In practice, however, information on Scope 3 water use is often incomplete and limited [71], and is therefore typically excluded from full-scope analyses of data center water use in most studies. 

_In this paper, we focus on the impact of data centers on public water systems. Therefore, while Scope 2 and 3 water remains crucial, we consider only direct Scope 1 water withdrawal and refer to it as water use or demand, unless otherwise noted._ 

## **2.2 U.S. Public Water Systems** 

In the United States, a public water system is defined as any infrastructure, publicly or privately owned, that supplies water for human drinking through pipes or other constructed conveyances to at least 15 service connections or to an average of at least 25 people for at least 60 days per year [22]. Nationwide, there are approximately 150,000 public water systems [22] and 16,000 municipal wastewater treatment facilities [72]. 

A public water system typically draws water from surface sources such as lakes, rivers, or reservoirs, though some rely on groundwater from aquifers. The water is treated to meet the EPA safety standards under the Safe Drinking Water Act [73], using methods like filtration and disinfection. After treatment, water may be stored in tanks and is then distributed to communities through a network of mains and smaller service lines. In addition to potable water, some water systems also provide reclaimed or recycled water for non-potable purposes, such as irrigation or industrial use. While wastewater is collected and processed separately subject to the Clean Water Act [74], drinking water and wastewater systems are increasingly considered as an integrated network, allowing coordinated planning and management of water resources. 

For the convenience of presentation, in this paper, we simply use the term “Public Water System” to refer to the entire system cycle, including sourcing, treatment, distribution, wastewater processing, and water recycling. 

**Community water systems.** A subset of public water systems that supply water to the same population year-round are known as _community water systems_ . There are roughly 50,000 community water systems in the United States, of which approximately 40,000 are small systems each serving no more than 3,300 people, about 9,000 are medium systems each serving between 3,301 and 100,000 people, and only 708 are large systems serving more than 100,000 people [28]. Nearly all hyperscale and colocation data centers in the United States are supplied by community water systems (mostly from potable water sources), with only a few exceptions drawing water from private groundwater sources (Table 2) that often require comprehensive environmental reviews and can impose additional availability risks [24]. 

Compared to roughly 3,000 power utilities nationwide that are interconnected through regional transmission networks [75], the 50,000 community water systems are significantly more fragmented, with many lacking sufficient financial and technical capacity. According to the most recent EPA surveys conducted prior to the AI boom, total funding needs for U.S. water infrastructure are projected to reach approximately $1.3 trillion over a 20-year period.<sup>3</sup> This includes $625 billion (in 2021 dollars) for drinking water systems between 2021 and 2040 [28] and $630 billion (in 2022 dollars) for wastewater treatment and stormwater control between 2022 and 2042 [29]. More recent third-party analyses that incorporate updated stormwater requirements and PFAS compliance costs estimate the total funding needs at $3.4 trillion (in 2025 dollars) over the period from 2025 to 2044 [30]. 

As highlighted in the EPA’s 2024 report to Congress [31], water affordability (defined as water service costs not exceeding 3.0% to 4.5% of household income) has emerged as a significant challenge nationwide, with an estimated 12.1 to 19.2 million U.S. households lacking access to affordable water services. Combined 

> 3All monetary values are in U.S. dollars. According to the U.S. Bureau of Labor Statistics [76], $1 in January 2021 has the same purchasing power as $1.24 in December 2025. For cost estimates using values between 2021 and 2025, inflation adjustments are not applied unless noted. 

7 



**_Table 1:_** _U.S. water treatment surplus capacity (in MGD) by system service population category for all water sources and systems based on the EPA’s most recent nationwide community water system survey (2006) [78]. The average surplus is defined as the design capacity minus the peak daily flow (Table 19 in [79]). *As the median surplus is not reported, we use the median design capacity minus the median peak daily treatment production (Table 9 in [80], which does not further disaggregate systems serving “over 10,000” people by size)._ 

|**Category**|**100 or**<br>**Less**|**101 -**<br>**500**|**501 -**<br>**3,300**|**3,301 -**<br>**10,000**|**10,001 -**<br>**50,000**|**50,001 -**<br>**100,000**|**100,001-**<br>**500,000**|**Over**<br>**500,000**|**All Sizes**|
|---|---|---|---|---|---|---|---|---|---|
|**Average Surplus**|0.08|0.14|0.34|0.68|0.84|2.48|3.86|9.41|0.72|
|**Median Surplus***|0.03|0.10|0.23|0.89||3.0|0||0.20|



with increasing climate risks [77], the affordability challenge has further intensified the financial pressures faced by many community water systems. Mitigating these risks requires improvements in capacity, planning, and investment. 

**Capacity expansion.** Public water systems are designed to safely and reliably meet maximum demand at all times and across all pressure zones, with additional margins and dedicated fire flow capacity to address extreme conditions such as prolonged heatwaves and droughts. Accordingly, drinking water treatment plants, pump stations, storage tanks and reservoirs, transmission mains, distribution networks, and wastewater treatment facilities must all be appropriately sized and carefully planned to accommodate peak scenarios. As a result, capacity expansion and upgrades to water infrastructure often require multiple years to complete, particularly for complex projects such as treatment plant expansions. These projects typically involve environmental review, water rights permitting and allocation, multi-agency approvals, funding authorization, detailed engineering design, and construction, all of which can substantially extend implementation timelines [24, 38]. For example, the process of identifying, developing, and bringing a new water source into service can take roughly 20 years or longer [32]. 

Moreover, many communities rely on reservoirs naturally replenished by rainfall or snowpack, which are highly sensitive to drought conditions and further constrain the development of additional reliable water supplies [24]. In addition, unlike cross-state electricity transmission, transporting water over long distances is considerably more challenging and generally not a viable option. Therefore, compared with power grid expansion, public water system upgrades are often equally challenging and time-consuming, and in some cases even more so due to strong local hydrology constraints. 

## **2.3 Water Capacity Bottleneck for Data Centers** 

A large data center that relies on evaporative cooling can consume millions of gallons of water per day during the hottest periods of the year. For example, a planned data center in Virginia is projected to use up to 8 MGD at full build-out [21], or reportedly as much as 30 times the town’s current largest water user (a beverage bottling company) [27]. 

On the other hand, a public water system’s short-term ability to accommodate new service connection requests largely depends on its available capacity, as decided by the more restrictive of the remaining physical capacity and the uncommitted capacity (i.e., total physical capacity minus already allocated capacity). The system’s full design capacity is therefore not the primary determinant. A public water system that serves, or is already committed to serving, existing users may still deny a new service connection if its available capacity is limited, even when the requested capacity is substantially smaller than that of some existing users. For example, a large technology company recently stated that its new data centers in Louisiana will draw water only from the host community’s “verified surplus water” capacity, ensuring “no strain on local water supplies” [16]. More details of capacity planning and management can be found in Appendix J. 

Unfortunately, many U.S. public water systems lack sufficient available capacity to accommodate new, large-scale industrial peak demands [24, 28]. Table 1 summarizes the surplus water treatment capacity of U.S. community water systems based on the EPA’s most recent nationwide survey conducted in 2006 [78]. The results indicate that the available surplus capacity is limited across all system sizes when compared with the water capacity demand of large data centers. Systems relying on surface water generally have larger total and surplus capacities than those relying on groundwater sources [79]. Although this survey has not been updated, the current situation is likely similar or potentially more constrained, as reflected by the rapidly 

8 



growing water infrastructure funding needs (after inflation adjustment) shown in the EPA’s 2023 Report to Congress [28]. Meanwhile, water capacity expansions are typically implemented incrementally in line with population growth, making it unlikely that the current nationwide statistics of water treatment surplus capacity differs substantially from the values reported in Table 1. For example, under California’s Drinking Water State Revolving Fund (DWSRF) policies [81], funding for capacity expansions is generally limited to 10% above the existing maximum daily demand. This rule is intended to ensure engineering reliability, maintain service to existing customers, and reduce financial and operational risk. 

Moreover, according to California’s 2025 report on water resources [82], more than 50% of urban water suppliers in California report zero annual surplus capacity, while an additional 4.5% report annual shortages that require supplier response actions. Among suppliers reporting monthly data, 20% project shortages during certain months of the coming year that require shortage response measures. Although the data on daily supply and demand is not available, the fraction of suppliers experiencing shortages of daily capacity for some days of the year is likely substantially higher than 20%. This suggests that accommodating new large water users, such as data centers that rely on evaporative cooling during the summer, can be challenging for many urban water suppliers under existing capacity constraints. 

Table 1 reports only the treatment plant capacity, while other critical components such as distribution pipes, wastewater processing, storage, and water source availability can also be bottlenecks. In addition, after accounting for operational safety margins, reliability headroom, fire suppression requirements, usage categories, and distribution losses, the practically _allocable_ surplus capacity becomes even smaller. The allocable capacity is the most relevant factor for connecting large water users, even when the system’s total capacity is large. For example, in Norwalk, Iowa, a city with approximately 15,000 residents, only 0.65 MGD of water capacity is designated as industrial reserve for new large water users as of March 2026, even though other existing water uses may be more substantial [83]. 

Consequently, depending on local climate conditions and cooling system design, supporting a 100 MW IT load using evaporative cooling—which is modest compared with emerging gigawatt-scale AI facilities and generally requires a peak water capacity allocation of approximately 0.5–2.5 MGD (Appendix C)—can frequently exceed the available surplus capacity of public water systems. This estimate does not include deployments in hot climates, where elevated ambient temperatures can substantially increase water demand, potentially pushing the required peak withdrawal capacity beyond 5 MGD. In fact, many data center projects have already required substantial upgrades to local water infrastructure, even when their water capacity demand is as low as approximately 0.1 MGD [16,18,21,36,84–86]. 

Importantly, this constraint is not limited to small or medium systems. For example, without substantial capacity expansion, even a large water utility such as Loudoun Water, which serves more than 300,000 residents and many data centers in Northern Virginia [26], could face difficulty accommodating the total water capacity requests of data centers within its service territory if all the data centers were to rely on evaporative cooling (Appendix B). 

To balance grid stress and water use, some data centers have increasingly adopted hybrid cooling strategies, such as applying evaporative cooling to a portion of IT loads and/or supplementing dry coolers with direct evaporative assistance only during summer periods. Nonetheless, even if only 10% of IT loads are cooled with direct evaporative assistance during the summer in a cold climate, the growing scale of AI data centers can still lead to substantial peak water demand. For example, when fully built, an AI-dedicated data center is planned for approximately 0.7 MGD of peak water demand, which can be substantial for the available surplus capacity of a small or medium public water system in the host community in Wisconsin [87]. In another case of a host town with less than 2 MGD available remaining physical capacity (according to the water utility’s 2024 annual report) [37], a peak water capacity request of 1.2 MGD reportedly triggers a major water infrastructure upgrade exceeding $100 million [36]. More recently, a leading technology company’s state-of-the-art data center under construction employs a “closed-loop, liquid-cooled system that recirculates the same water” [19] and relies on water evaporation for facility-level cooling only during the hottest days of the year. However, the data center may not be able to operate at full capacity until its allocated water capacity of 8 MGD becomes available, as anticipated in 2031 [18, 19]. In another instance involving a new data center project in Virginia, the county record [33] indicates that the future expansion of water capacity from 2 MGD to 8 MGD to serve the data center “will limit the water supply available [to the water authority’s service area] and will impair economic and other development opportunities in those 

9 



**_Table 2:_** _2024 U.S. data center water and energy use of selected companies. An asterisk “_<sup>_∗_</sup> _” denotes values estimated or assigned based on third-party sources. A dagger “_<sup>_†_</sup> _” denotes ratios derived from reported water consumption values when withdrawal-based ratios are not explicitly disclosed. A dash “–” indicates that the corresponding value is not reported and cannot be plausibly estimated. Approximately 94% of Hyperscale-1’s water use is potable, although the source is not explicitly identified as “municipal water.” The details of data sources are available in Appendix A.4._ 

|**Company**|**IT Energy**<br>**(MWh)**|**Total Energy**<br>**(MWh)**|**PUE**|**Water Consumption**<br>**(ML)**|**Water Withdrawal**<br>**(ML)**|**WUE**<br>**(L/kWh)**|**Consumptive**<br>**Ratio**|**Municipal**<br>**Ratio**|
|---|---|---|---|---|---|---|---|---|
|Hyperscale-1|20,417,908|22,255,520|1.09|23,120|29,510|1.13|0.78|–|
|Hyperscale-2|7,764,474|9,006,789|1.16|2,951|3,934|0.38|0.75<sup>_∗_</sup>|99.45%<sup>_†_</sup>|
|Hyperscale-3|11,828,810|12,775,115|1.08|1,698|2,365|0.14|0.72|99.79%|
|Hyperscale-4|1,564,220|1,705,000|1.09<sup>_∗_</sup>|2,132|2,842|1.36|0.75<sup>_∗_</sup>|93.54%|
|Hyperscale-5|16,004,477<sup>_∗_</sup>|18,245,104|1.14|1,560|2,081|0.10|0.75<sup>_∗_</sup>|–|
|Colocation-1|4,533,626|5,667,033|1.25<sup>_∗_</sup>|2,793|3,724|0.62|0.75<sup>_∗_</sup>|99.79%<sup>_†_</sup>|
|Colocation-2|1,741,150|2,420,198|1.39|1,654|2,153|0.95|0.77|94.49%|
|Colocation-3|2,584,146|3,617,805|1.40|2,119|2,323|0.82|0.91|100.00%|
|Colocation-4|3,520,345|5,139,703|1.46|1,021|1,312|0.29|0.78|–|
|Colocation-5|1,237,664|1,522,327|1.23|1,584|2,112|1.28|0.75<sup>_∗_</sup>|–|
|Colocation-6|598,551|826,000|1.38|431|575|0.72|0.75<sup>_∗_</sup>|–|
|Colocation-7|623,706|829,529|1.33|36|48|0.06|0.75<sup>_∗_</sup>|–|
|**Hyperscale**|**57,579,889**|**63,987,528**|**1.11**|**31,460**|**40,732**|**0.55**|**0.77**|–|
|**Colocation**|**14,839,188**|**20,022,595**|**1.35**|**9,638**|**12,246**|**0.65**|**0.79**|–|



localities that are dependent on water supply.” As a result, future expansion of the data center project may face substantial delays while new water supply sources are developed, a process which itself is a “lengthy and costly project” [33]. 

Therefore, choosing between air-cooled and evaporative systems for facility-level cooling to manage the tradeoff between water and electricity use may no longer be solely a design decision. Instead, the availability of public water capacity is increasingly emerging as a binding, yet often underestimated, constraint and projected to surpass power access as a limiting factor for data center siting within the next five years [48]. 

# **3 Understanding the Characteristics of Data Center Water Use** 

This section characterizes data center water usage, emphasizing its highly variable and spiky nature, which can pose significant challenges for public water systems. We then examine the impact of data centers on public water systems and the limitations of existing volumetric-based water footprint studies, which often overlook temporal variability and peak water demand that are critical constraints for data centers. 

## **3.1 How Data Centers Use Water Differently from Other Users** 

To understand the characteristics of data center water use, we review public reporting sources, including data center sustainability reports and water utility data, and identify two key features: a high consumptive ratio and a high peaking factor. 

### **3.1.1 High consumptive ratio** 

The consumptive ratio quantifies the fraction of withdrawn water that is actually consumed, providing a measure of the impact on downstream water availability. We see from Table 2 that data centers have a much higher consumptive ratio than typical industrial or residential users served by public water systems, because a large fraction of water withdrawal is lost to evaporation in facility-level cooling systems. Specifically, data centers typically exhibit high consumptive water ratios, ranging from 70 to 90%, in contrast to urban residential homes that consume only 5–15% of their total water withdrawals [88]. Manufacturing facilities generally have a consumptive rate of around 20% [70], and office buildings are often estimated to consume about 10% of their water use [17]. For comparison, the average consumptive rate for public water supply in the contiguous United States is approximately 12% [52]. 

Unlike other industrial processes where water can often be recycled or returned to the system, evaporative cooling converts water into vapor to remove heat, making these losses generally unavoidable. In fact, some data centers may add chemicals to treat public water on-site, allowing for more cycles of concentration in cooling towers. This reduces blow-down water and further increases the consumptive ratio. Even 

10 



![Figure](assets/figure_0001_page_0011.svg)**_Figure 1:_** _(a) Monthly WUE of a data center with 34.5 MW IT capacity in Phoenix, Arizona, in 2019 [64]; (b) Monthly PUE and WUE comparisons of two data centers in Phoenix, Arizona, in 2019 (one with 54 MW IT capacity using dry coolers, and another one with 34.5 MW IT capacity using cooling towers) [64]; (c) Consumptive ratios of selected companies and data center sites in Table 2. The site-level consumptive ratios are for a large technology company’s U.S. data center fleet, excluding air-cooled sites._ 

with liquid-cooled servers that allow higher server-level temperature setpoints, water evaporation are still preferred for facility-level cooling during summer peak periods in many places [16], keeping consumptive ratios high. 

For the same amount of water withdrawal, a high consumptive ratio can pose additional water challenges for public water systems, particularly in drought-prone regions with limited supply. These challenges are further amplified by the complex water rights and applicable consumptive constraints legally imposed on the host community’s public water systems [38]. As a result, data centers’ higher water consumptive ratio may have a greater impact on water availability, a potential concern discussed in a recent study on the Washington Metropolitan Area water supply [50]. 

### **3.1.2 High peaking factor** 

Similar to power grids, public water systems are engineered to reliably meet maximum demand at all times and across all pressure zones, with additional safety margins [43]. Accordingly, a user’s peak water demand is a critical factor for water infrastructure planning, system resilience, and overall operational reliability. 

Nonetheless, most data center operators disclose only annual total water use, often aggregated across an entire corporate fleet, providing limited visibility into facility-level peak demand [17,43,89,90]. To address this limitation, we review publicly available records to characterize data centers’ peak water demand. In particular, we focus on the daily peaking factor whenever applicable, defined as the ratio of maximum daily water use to average daily water use. This metric is widely used in water infrastructure planning because it directly informs capacity sizing for treatment, storage, and distribution systems [43, 50, 91, 92]. Other key design parameters, such as peak hourly flow rate and instantaneous flow expressed in gallons per minute, are typically derived from the daily peaking factor using additional multiplicative adjustment factors [91,92]. 

**Evaporative cooling towers.** We first examine the monthly water usage effectiveness (WUE, defined as the ratio of total water consumption to IT energy use) for a large colocation data center in Phoenix, Arizona, serving a nearly constant IT load in 2019 [64]. Cooling towers are a widely-used conventional cooling system, offering stable performance and broad applicability across diverse climates [93]. The ratio of the peak monthly WUE to the average monthly WUE is approximately 1.46, indicating seasonal variation in water intensity despite relatively stable computing demand. Assuming a constant consumptive ratio (i.e., the fraction of withdrawn water that is consumed), seasonal variation in WUE can serve as a proxy for variation in withdrawal. In the absence of the facility-specific daily peaking factor, regulatory guidance commonly applies a minimum adjustment factor of 1.5 [92], yielding an estimated daily peaking factor of at least 2.19. 

In a more recent case of The Dalles, Oregon, the water pressure zone (Zone 310) in which a hyperscale data center using cooling towers is located and accounts for the majority of zonal water demand, exhibits a measured peaking factor of 2.21 [17, 39, 91]. Consequently, given that non-data center water demands are typically less “spiky,” the data center’s daily peaking factor may exceed 2.21. 

11 



![Figure](assets/figure_0002_page_0012.svg)**_Figure 2:_** _Water use data collected from the monthly financial reports available at [51]. (a) Monthly water use of a hyperscale data center in Iowa; (b) The ratio of the data center’s water use to the total of 20 largest water users served by same water works; (c) Monthly peaking factors (i.e., ratio of the maximum monthly water use to the average use between 2022 and 2025) of all the large water users counted by the water works._ 

**Dry cooling with evaporative assistance.** An increasingly adopted configuration is air-based dry cooling supplemented with evaporative assistance during periods of high ambient temperature. Under typical server temperature setpoints and local climatic conditions, evaporative assistance may operate for only 5 to 15% of the year, and possibly up to 40% in hotter climates [94]. As a result, water use is highly concentrated in a limited number of hot days, leading to a relatively high peaking factor even when total annual water consumption is substantially lower compared to conventional cooling towers. As evaporative operation is triggered primarily on the hottest days, the high water demand is temporally aligned with periods of peak thermal stress, further amplifying the maximum hourly and daily water use. Consequently, although annual water use may be substantially reduced, the infrastructure implications associated with peak demand of evaporative assistance systems can remain significant. 

Figure 2c presents the monthly water withdrawal of a hyperscale data center in Iowa, indicating a ratio of peak monthly withdrawal to average monthly withdrawal of 4.30. It is important to note that the disclosed withdrawal volume includes a small portion of domestic water use, which typically exhibits a peaking factor in the range of 1.5 to 2.5. Consequently, the monthly peak-to-average water withdrawal ratio attributable solely to the data center cooling load is likely higher than 4.30. Applying a minimum adjustment factor of 1.5 to translate average demand to maximum day demand [92] yields an estimated daily peaking factor of at least 6.45. 

We further present in Figure 2b the percentage contribution of the data center’s monthly water use relative to the total water use of the top 20 largest users served by the same water works. This analysis further highlights the pronounced seasonal variability of data centers that rely on evaporative assistance during the summer. Although the data center’s water use is relatively low during cooler months, its elevated summer water demand is sufficiently large that it becomes the largest annual water user in both 2024 and 2025 served by the water works [51]. 

By comparison, we also present in Figure 2c the monthly peaking factors for other large water users reported in the financial disclosures from 2022 to 2025 [51]. Some water users appear in the reports for fewer than four years; for these users, we compute the ratio of their maximum monthly water use to their average monthly use over the period in which they are reported. We can see that all other large water users exhibit substantially lower peaking factors than the data center. Note that, water user T, a large corporate campus, exhibits an abnormally high water use of approximately 15 million gallons in August 2023, exceeding more than ten times its average monthly use between 2022 and 2025. This spike is likely associated with campus expansion and infrastructure upgrades during that period, as indicated by the city’s building permit reports (May to July 2023) [95], resulting in a one time increase in water use. It does not exhibit the recurring annual pattern observed for the data center and other users. We therefore exclude this anomalous month when calculating the monthly peaking factor for user T. 

**Peaking factor for planning.** The total water capacity allocated to a data center is often not publicly disclosed, frequently on the grounds of business confidentiality. In one rare case with disclosed water demand [87], a large AI-dedicated hyperscale data center in Wisconsin requests a total water capacity of 

12 



0.7 MGD to cool its non-AI servers, while its average daily water demand is only approximately 23,000 gallons, including relatively stable domestic use but excluding fire protection water. This implies a peaking factor exceeding 30. Due to the cold climate in Wisconsin, it is estimated that up to only 480 hours of cooling per year will require water [87], which reduces the total annual water use while increasing the peaking factor. In addition, the corresponding peaking factor for wastewater discharge at the same facility is approximately 15, which is substantially higher than that of non-data center users. 

In another case regarding a hyperscale data center in Leesburg, Virginia, government records on water allocation and diesel generator capacity indicate that the planned daily peak WUE is estimated at 1.15 L/kWh as of March 2025 (Appendix B), whereas the reported annual WUE for the operator’s data center fleet across Northern Virginia is 0.14 L/kWh as of 2023 (which may be lower due to efficiency improvements in 2025) [56]. This comparison suggests that the peaking factor for the Leesburg data center campus is approximately 8 or higher. 

Similarly, in an innovation district under development in Indiana, a leading technology company’s data center hosting AI and core products is estimated to have a planned daily peaking factor of 6.3 or higher, whereas other users in the district typically exhibit peaking factors below 2.0, except for one hospital at 3.0, based on water allocation agreements or pre-agreements (Appendix D). When the full water capacity of 8 MGD is delivered in 2031, the data center’s allocation would account for 32% of the total 25 MGD of new water capacity, making it likely the largest single user in the district based on allocated capacity. 

The elevated peaking factor is also evident at an aggregated level when multiple data centers are geographically clustered, such as in Northern Virginia. For example, a recent report analyzing water utility data indicates that the measured daily peaking factor of data centers within the Prince William Water service area reaches 10 in 2024 [50]. However, when a large number of data centers with diverse cooling system designs and operational configurations are aggregated, the overall peaking factor is moderated by multiplexing effects. Nevertheless, the weighted estimate for the actual peaking factor (not for infrastructure planning purposes) in Northern Virginia remains in the range of 3.5 to 3.7 under different scenarios (Table 6-5 of [50]), which is still substantially higher than the typical _aggregated_ peaking factor of 1.5 to 2.5 for non-data center users [96]. 

**Summary.** The water use of data centers exhibits pronounced seasonal and even daily spikes, resulting in peaking factors estimated to range from 3 to 10, depending on ambient temperature and cooling system configuration. Cooling systems that rely on evaporative assistance during the summer generally exhibit substantially higher peaking factors than conventional evaporative cooling towers. Additionally, in cooler climates and/or when server-level cooling operates at higher temperature setpoints that only need evaporative assistance for a limited number of days, the peaking factor can be even higher. For planning purposes, similar to power capacity redundancy [50,97], data centers often request more water than their actual demand to hedge against extreme heat events and/or rapid future load increases. Finally, we note that while water storage tanks can buffer short-term fluctuations in hourly demand, they are often less effective for prolonged events such as extended or month-long heatwaves that data centers typically prepare for. As a result, data centers employing evaporative cooling generally reserve substantial daily peak water capacity allocations to hedge against these risks. Nonetheless, as water capacity is increasingly becoming a constrained societal resource (e.g., in some cases, it can take 20 years or longer to identify, develop, and bring new water supply sources into service [32]), the combination of water tanks and other water-aware computing-based solutions (Appendix F) presents a promising approach to lowering the peak water demand and strengthening public water system resilience. 

## **3.2 Impacts of Data Centers’ Peak Water Demand on Public Water Systems** 

Similar to power systems that are designed to provide a reliable electricity supply with stable voltage maintenance, public water systems need to reliably meet maximum demand at all times and across all pressure zones, with additional safety margins and dedicated fire flow capacity to address extreme conditions such as prolonged heatwaves. Nonetheless, within the broader context of underfunded infrastructure and growing water affordability concerns [28,29,31], many community water systems have limited _surplus_ capacity available for large data centers, and developing new water sources can require a long lead time [24]. 

**Water utility responses to data center peak demands.** Some water utilities have begun to express concerns regarding data centers’ peak water demands. For example, West Des Moines Water Works stated in 2022 

13 



that it “will only consider future data center projects” if such projects can “demonstrate and implement technology to significantly reduce peak water usage from the current levels,” in order to preserve water availability for other users [98]. 

When asked about the water capacity requested by new data centers that would exceed the county’s available supply, a representative of the Newton County, Georgia, water authority stated that “they [data centers] are taking up the community wealth” and “we just don’t have the water” [25]. In Newton County, upgrades and capacity expansions needed to meet the new industrial-scale water demands and future growth are reportedly projected to increase local water rates by 33% over the next two years, compared with a typical annual rise of approximately 2%, unless additional external funding is secured [25]. 

In the United Kingdom, Thames Water, the country’s largest water utility, has reportedly discussed options for “restricting or reducing or objecting to” data centers’ peak water use during the hottest periods of the year [99]. 

**Corporate-funded infrastructure upgrades.** In response to concerns such as those above, communitycorporate partnerships have increasingly been established by data center projects, paralleling investments in power infrastructure [3]. For example, a planned data center in Virginia will receive up to 8 MGD of water capacity at full build-out [21], which would reportedly be approximately 30 times greater than the town’s current largest water user (a beverage bottling facility) and require roughly $300 million in water infrastructure investments, anticipated to be borne by the data center operator [27]. In another case, a data center operator has funded water storage infrastructure to support the host town and enhance its water security [100]. More recently, a major technology company pledged up to $400 million to upgrade local public water infrastructure to support evaporative cooling at its data center campus, “[reducing] electricity demand by 25–35% during periods of peak summer loads when regional power demand is highest” [16]. 

In a public water system, even though the total system-wide capacity is available, any component, including drinking water treatment plants, pump stations, storage tanks and reservoirs, transmission mains, distribution networks, and wastewater treatment facilities, can become a bottleneck for meeting the water demand of data centers. For example, despite sufficient aggregate system capacity in Leesburg, Virginia, a technology company agreed to pay $25 million to fund water and wastewater improvements to secure approximately 0.64 MGD of water capacity and “ensure the cost of serving our facilities [i.e., the data center campus] does not fall on local ratepayers” [3, 84]. Similarly, another data center with a relatively modest water capacity need of 0.1 MGD and wastewater need of 0.08 MGD requries utility upgrades, including the construction of a new pump station, reportedly costing $5.4 million paid by the data center [85]. In another case, a small host community with less than 2 MGD available remaining capacity (according to the water utility’s 2024 annual report) [37], faced a peak water capacity request of 1.2 MGD, which the data center indicates is for fire protection [35], reportedly triggering a major water infrastructure upgrade exceeding $100 million [36]. 

In a more complex scenario, a city’s water reservoir situated on federal lands needs capacity expansion to support future growth, which would necessitate comprehensive environmental review and federal approval. To expedite the process, the city is considering alternative approaches, such as legislative measures to transfer federal land ownership [39]. This example highlights the non-financial and regulatory constraints that can limit the effective capacity of public water systems. 

**“Unmet” water demand.** Importantly, while power capacity shortages for data centers are widely recognized [2] and have even been described as a “crisis” [101], the increasingly binding constraint of Scope 1 water availability has received much less attention. Treating water constraints in isolation can undermine the viability of water evaporative cooling, which is recognized as “the most efficient means of cooling in many places” [17]. In the absence of available water capacity, data centers may have to rely on dry cooling technologies, which can increase electricity demand, even for state-of-the-art AI facilities that operate with relatively high server-level cooling temperature setpoints [46]. For example, in Newton County, Georgia, a recent data center project in 2025 reportedly requested approximately 6 MGD of water capacity [25], which could not be accommodated under the county’s existing water infrastructure [102]. Moreover, although many data centers served by Loudoun Water in Northern Virginia use dry cooling as facilitated by higher server-level cooling temperature setpoints, this also partially reflects Loudoun Water’s potential capacity constraints at the time those data centers were connected—if all the data centers had employed water-based 

14 



evaporative cooling, Loudoun Water’s available capacity could have been depleted or even exceeded on the hottest days (Appendix B). 

Consequently, the “ _unmet_ water demand” of data centers—the water that would have been used if it were available—is often effectively viewed as “zero water demand” in public discourse. This silent shift may further exacerbate existing power capacity shortages, in addition to contributing to other externalities, including increased public health risks [10]. 

## **3.3 Limitations of Total Volumetric Water Footprints** 

We conclude this section by emphasizing the contrast between the discussion above and prior work on data centers’ overall water footprint. Existing studies predominantly focus on quantifying and reducing the total water consumption of data centers [2, 5, 9, 56, 103, 104], often including indirect water consumption for electricity generation [49, 103] and sometimes incorporating regional water stress adjustments [66, 105]. Moreover, amid broader global challenges of water imbalances described as “water bankruptcy” [106], several technology companies have pledged substantial financial resources to become “Water Positive,” aiming to manage their water consumption responsibly and mitigate associated ecosystem impacts through initiatives such as water replenishment and restoration projects [107–110]. Nonetheless, these studies and efforts exhibit the following limitations and may lead to unintended consequences. 

**Water withdrawal.** While the existing studies are valuable for understanding overall impacts of data centers on global and regional long-term water resources and ecosystems, most of them, except for a few cases [50,56], tend to overlook the effects of water withdrawal, which is critical for short-term water availability, water quality, and allocation [38, 55]. For example, the U.S. Energy Information Administration (EIA) routinely reports water withdrawal for the electric power sector, which is one of the largest freshwaterwithdrawing sectors in the United States and comparable to agriculture in total withdrawal volumes (42.5% for thermoelectric power vs. 43% for crop irrigation relative to the total water withdrawal in the contiguous United States) [52, 67]. It has also recently begun publishing monthly data on water withdrawal and consumption for individual power plants [68]. Importantly, public water systems almost always use water withdrawal as the primary metric for planning and allocation, and technology companies include Scope 1 water withdrawals in their annual sustainability disclosures [17,89]. 

**Cross-sector comparisons.** The distinction between municipal (typically potable) water for Scope 1 and non-potable water for Scope 2 is often missing from aggregate water consumption numbers, while Scope 3 water is rarely included due to the lack of reliable data in the public domain. However, the water footprint reported for other sectors often reflects the full lifecycle [111]. This can unintentionally lead to misinformed cross-sector comparisons. For example, the full-scope lifecycle water use for certain animal-derived products, such as hamburgers—including rainwater stored in the soil (i.e., green water) used to grow feed crops for patty production, whereas only less than 2% of the meat’s overall water footprint may come from public water systems (Appendix H) [112]—is sometimes directly compared with the operational or Scope 1 water use of data centers or AI model inference. Similarly, water-intensive golf courses, whose potable water use is increasingly regulated and accounts for only around 10% or less of total water applied (Appendix I), are sometimes compared against a data center’s Scope 1 municipal (potable) water use. Fundamental inconsistencies in accounting scopes and/or water types render such comparisons uninformative, obscuring the challenges faced by public water systems and misinforming public discourse. 

**Peak Scope 1 water use.** Most importantly, while existing studies have examined the spatial impacts of data center water use [5, 66, 104], they do not account for the high peak demand of data centers—particularly daily peak Scope 1 water use—which is a key consideration for public water system management. This omission neglects the higher peaking factor associated with data centers compared to typical public water users (Figure 2c), an increasing challenge given that many U.S. public water systems are aging and underresourced [23,24]. 

# **4 Quantifying and Forecasting Data Centers’ Peak Water Demand** 

This section presents a quantitative analysis of U.S. data centers’ total peak water demand, with projections through 2030. The results reveal that in the high-growth case and without substantial water usage reduction, U.S. data center expansion could require up to 1,451 million gallons per day (MGD) in new water capacity 

15 



from 2024 to 2030, with infrastructure valuations reaching as high as $15-58 billion. If hypothetically pooled, this capacity would be sufficient to supply New York City for much of the year, except during a limited number of peak demand days. When significant water reductions are realized at a compound annual rate of 10%, the new water capacity need could decrease to 227–604 MGD, with a corresponding valuation range of $2–9 billion and $6–24 billion, depending on the IT load growth rates. Importantly, these impacts are highly concentrated on individual communities hosting data centers, highlighting the need of sustained corporate-community partnerships to address them effectively. 

## **4.1 Growth Rates, Scenarios, and Data Sources** 

We first present a general methodology to quantify data centers’ peak water demand. Analogous to using IT power load as the basis for data center design (e.g., cooling systems, battery sizing, and backup generators), we use peak daily water use as the primary metric. This metric is widely employed in water infrastructure management and directly informs capacity sizing for treatment, storage, distribution, and wastewater handling [50,91,102]. While the actual daily peak water demand of a data center can be directly measured for post-hoc reporting and analysis, estimating it in advance is valuable for guiding infrastructure planning. For planning purposes, the peak daily water demand can therefore be calculated as: 



$$
PeakWaterDemand = β \cdot{}{} TotalWaterConsumption T , (1)
$$

where _TotalWaterConsumption_ is the total Scope 1 water use planned over _T_ days, and _β ∈_ [1 _, ∞_ ) is the estimated or planned peaking factor. 

### **4.1.1 Growth rates** 

We consider three data center growth cases informed by the recent U.S. data center energy report published by researchers at Lawrence Berkeley National Laboratory (LBNL) [2], complemented by industry market analyses [113, 114]. These scenarios capture a range of plausible medium-term expansion trajectories for U.S. data center capacity under accelerating AI-driven demand. 

- **Low growth:** The total U.S. data center energy increases at a compound annual growth rate (CAGR) of 13% as considered in [2]. For IT loads of hyperscale and colocation data centers, the CAGRs are 21% and 23%, respectively. 

- **Mid growth:** This represents the algorithmic mean of the low- and high-growth cases for IT loads. 

- **High growth:** The total U.S. data center energy increases at a CAGR of 27% as considered in [2]. For IT loads of hyperscale and colocation data centers, the CAGRs are 29% and 32%, respectively. 

While the LBNL report projects growth through 2028, we extend the same growth rates through 2030, to align with the study period used in a recent analysis of U.S. data centers’ carbon and water footprints [5] and with projected growth rates from other market analyses (e.g., CAGR of approximately 20% through 2030 [113] and approximately 22.4% over 2023–2030 under a medium-growth scenario [114]). 

### **4.1.2 Scenarios** 

Previous studies [2,50] consider constant or increasing average water usage effectiveness (WUE, defined as the ratio of total water consumption to total IT energy). In our analysis, we incorporate annual industry-wide WUE reductions by assuming compound annual rates of 5% and 10% under moderate and optimistic scenarios, respectively. These adjustment rates reflect continued technological and operational improvements in data center water efficiency, as well as tightening capacity constraints in many U.S. public water systems, which may render evaporative cooling infeasible or restrict its deployment in certain regions. Specifically, we consider the following three scenarios. 

- **Baseline:** Industry-wide WUEs for hyperscale and colocation data centers remain constant at 2024 levels through 2030. 

- **Moderate:** Industry-wide WUEs for hyperscale and colocation data centers decrease at a compound annual rate of 5% between 2024 and 2030. 

16 



- **Optimistic:** Industry-wide WUEs for hyperscale and colocation data centers decrease at a compound annual rate of 10% between 2024 and 2030. 

While individual companies may achieve greater WUE reductions, the industry-wide adjustments applied in our study are consistent with previous observations and align with common reduction targets reported by data center operators [115, 116]. For example, a major technology company’s goal of reducing water intensity by 40% by 2030 [116], relative to its 2020 level, lies between our Moderate and Optimistic scenarios. Importantly, under the Moderate and Optimistic scenarios, the projected U.S.-wide average WUE in 2030 (Table 8) is approximately 0.43 L/kWh and 0.31 L/kWh, respectively. These values are lower than the modeled WUE projections reported in [2] and are consistent with prevailing industry trends, thereby representing plausible forward-looking estimates. In addition to the Baseline, Moderate, and Optimistic scenarios, we include two references—Reference (LBNL) and Reference (NS)—which apply WUE values based on the modeled results in [2] and [5], respectively.<sup>4</sup> 

Currently, the majority of data center water use is supplied by public potable sources (e.g., more than 94% for a large technology company [17]), despite the adoption of reclaimed or recycled water in some cases, which is often also provided by municipal utilities [50]. In the following analysis, we do not further differentiate between potable and non-potable water. Instead, we discuss the role of non-potable water in Section 5 as a potential approach to mitigating the impacts of data centers on public water systems while preserving the peak power reduction benefits of evaporative cooling. 

### **4.1.3 Data Sources** 

We briefly describe the data sources and settings for our analysis, with details available in Appendix A. For each IT load growth case (low, mid, and high growth), we quantify the annual water consumption, withdrawal, and peak water withdrawal for hyperscale and colocation data centers separately, while excluding potential water use from the smaller and declining segment of other data center types (Section 2.1). 

Rather than relying on simulated WUE models [2, 5] that may exhibit substantial discrepancies from actual water use, we base our water efficiency calculations on industrial disclosures. Specifically, we review the 2024 sustainability reports from five leading hyperscale data center operators and seven large colocation data center providers (Table 2), which collectively represent about 86% and 22% of the hyperscale and colocation IT loads (relative to the 2024 mid values in Table 7) in the United States, respectively. Although these industrial reports have varying levels of reporting quality, they are among the most reliable and comprehensive data currently available in the public domain. 

Additionally, we review multiple public sources, including government records, water utility data, and planning documents, to conservatively determine peaking factors to estimate the peak water demand. Finally, public reports are used to estimate the capacity valuation of U.S. public water system infrastructure needed to support the new data centers’ peak water demand from 2024 to 2030. 

While we have collected public data sources to the best of our ability and aimed to cover most plausible scenarios, projection uncertainties inherently remain, especially given the rapid growth of the data center industry and the challenge of obtaining detailed industrial data. Our parameter choices are generally conservative to avoid overestimation. As more comprehensive and preferably audited data in standardized formats become available, refinements to our analysis may be necessary. 

## **4.2 Results** 

We now present the analysis of U.S. data centers’ Scope 1 water use from 2024 to 2030, including the annual water withdrawal, consumption, total water capacity need, and valuation of the capacity need. 

### **4.2.1 Annual water withdrawal** 

We first present the total annual water withdrawal by U.S. data centers from 2024 to 2030, which serves as the basis for our water capacity analysis. The results are shown in Figure 3, with detailed values provided in Table 10. 

We observe that our Baseline scenario, in which water efficiency remains constant through 2030, generally yields a higher water withdrawal number than Reference (LBNL). In contrast, the Moderate and 

> 4For our analysis, we re-calculate the total water _consumption_ for Reference (LBNL) rather than directly using the 2024–2028 values reported in [2]. Specifically, we follow the standard annual on-site WUE definition, resulting in lower water consumption estimates than those originally reported. See Table 9 in the appendix for details. 

17 



![Figure](assets/figure_0003_page_0018.svg)**_Figure 3:_** _Annual water withdrawal in billion gallons (BG) under Reference, Baseline, Moderate, and Optimistic scenarios, 2024–2030. For Baseline, Moderate, and Optimistic scenarios, dark and light shades represent hyperscale and colocation data centers, respectively. The dashed line indicates total water withdrawal for Reference (LBNL)._ 

Optimistic scenarios, reflecting industry-wide WUE reductions, yield lower projected withdrawals than Reference (LBNL), highlighting the potential benefits of adopting more water-efficient cooling technologies for U.S. data centers in the future. 

Specifically, Figure 3 illustrates an upward trend in annual water withdrawals by data centers from 2024 to 2030 across all growth scenarios. In the low-growth case, total withdrawals generally remain below 80 billion gallons even under the Baseline scenario, whereas the high-growth case projects a substantial increase that could exceed 140 billion gallons by 2030 under the Baseline scenario. Comparing the Baseline and Optimistic scenarios highlights that substantial improvements in cooling efficiency and operational practices could potentially reduce the projected total water demand by approximately half. Our results also distinguish between hyperscale and colocation data centers, showing that colocation data centers account for the majority of usage, although both segments scale substantially as overall demand rises. The growing trajectory in the mid and high growth cases underscores that, without a shift toward moderate or optimistic water efficiency improvements, the continued data center expansion can exert significant and accelerating pressure on public water resources. This finding is consistent with a recent internal projection by a leading technology company, which estimates that its Scope 1 water usage could increase by approximately 75% relative to the 2024 level [117], despite substantial improvements in water efficiency. 

Based on the most recent study [52], public water supply systems withdraw an average of 35,400 million gallons per day (MGD), including 4,219 MGD of consumptive use. This represents approximately 14.5% of the total water withdrawn annually, on average, within the contiguous United States during water years 2010–2020 for major categories including crop irrigation, public supply, and thermoelectric power. Although public supply accounts for a smaller share of total withdrawals, it provides drinking water and directly supports essential human uses, making it a distinct category of water use that is comparatively more critical and constrained. 

When benchmarking against total public water supply, we use the baseline values in the most recent study [52] without adjusting for future system-wide changes. Nationally, under the Baseline scenario, total U.S. data center water withdrawals in 2030 are projected to represent approximately 0.6% to 1.1% of total public water withdrawals in the low- and high-growth cases, respectively. Specifically, without significant improvements in water efficiency and in the high-growth case, total U.S. data center water withdrawals in 2030 could rival the current annual water supply of Los Angeles (from 2019 to 2024), which serves approximately 4 million residents [118]. 

Under the Optimistic scenario, which assumes substantial water efficiency gains, the national share of U.S. data center water withdrawals decreases further to approximately 0.3% to 0.6%, depending on the growth rates. This highlights the potential benefits of implementing water-efficient technologies. 

Although the national share of data center withdrawals relative to total public supply remains modest, public water is fundamentally a local and seasonal resource. It is considerably more difficult to transfer across regions than electricity and is constrained by local water infrastructure, hydraulic conditions, water rights, and permitting requirements. Thus, the impacts of data centers’ Scope 1 water use should be interpreted in the context of local public water infrastructure, particularly the _available_ surplus capacity of public water systems, which is often constrained due to longstanding underinvestment (Section 2.3). 

18 



![Figure](assets/figure_0004_page_0019.svg)**_Figure 4:_** _Annual water consumption in billion gallons (BG) under the Reference, Baseline, Moderate, and Optimistic scenarios from 2024 to 2030. For the Baseline, Moderate, and Optimistic scenarios, dark and light shades represent hyperscale and colocation data centers, respectively. The dashed line indicates the result for Reference (LBNL)._ 

**_Table 3:_** _Average daily demand and water capacity need (million gallons per day), and capacity valuation from 2024 to 2030 under Reference, Baseline, Moderate, and Optimistic scenarios. The valuations are expressed as “(low, high)” cost estimates._ 

|**Year**|**Metric**|**Growth**|**Refe**|**rence**||**Baseline**|||**Moderate**|||**Optimistic**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||**LBNL**|**NS**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|
|||Low|50|206|68|31|37|68|31|37|68|31|37|
||ADD|**Mid**|**55**|**221**|**74**|**34**|**40**|**74**|**34**|**40**|**74**|**34**|**40**|
|2024||High|59|236|79|38|42|79|38|42|79|38|42|
|||Low|225|927|306|138|168|306|138|168|306|138|168|
||Capacity|**Mid**|**245**|**995**|**332**|**153**|**178**|**332**|**153**|**178**|**332**|**153**|**178**|
|||High|265|1064|357|169|189|357|169|189|357|169|189|
|||Low|179|599|223|96|127|164|71|93|118|51|67|
||ADD|**Mid**|**259**|**832**|**312**|**136**|**176**|**230**|**100**|**129**|**166**|**72**|**94**|
|||High|339|1065|402|176|226|295|130|166|214|94|120|
|2030||Low|807|2694|1003|433|570|737|318|419|533|230|303|
||Capacity|**Mid**|**1167**|**3744**|**1406**|**613**|**793**|**1033**|**451**|**583**|**747**|**326**|**421**|
|||High|1528|4793|1809|794|1015|1330|583|746|961|422|539|
|||Low|582|1767|697|295|402|431|180|251|227|92|135|
||**Capacity**<br>|**Mid**|**922**|**2749**|**1074**|**460**|**614**|**702**|**297**|**404**|**415**|**172**|**243**|
||**Increase**|High|1262|3730|1451|625|826|972|415|558|604|253|351|
|**2024**_→_**2030**|**Capacity**|Low|(6, 23)|(18, 71)|(7, 28)|(3, 12)|(4, 16)|(4, 17)|(2, 7)|(3, 10)|(2, 9)|(1, 4)|(1, 5)|
||**Valuation**|**Mid**|**(9, 37)**|**(27, 110)**|**(11, 43)**|**(5, 18)**|**(6, 25)**|**(7, 28)**|**(3, 12)**|**(4, 16)**|**(4, 17)**|**(2, 7)**|**(2, 10)**|
||($ Billion)|High|(13, 50)|(37, 149)|(15, 58)|(6, 25)|(8, 33)|(10, 39)|(4, 17)|(6, 22)|(6, 24)|(3, 10)|(4, 14)|



### **4.2.2 Annual water consumption** 

Water consumption is an important, and sometimes regulated [38,69], performance metric for public water systems, as it directly affects regional water availability, particularly during drought conditions [50]. Thus, while our primary focus is on data centers’ Scope 1 water withdrawals, we also present Scope 1 water consumption in Figure 4 for completeness, with detailed values provided in Table 9. As shown, total U.S. data center water consumption follows a trend similar to total water withdrawals in Figure 3. 

Compared with the contiguous U.S. public water supply’s total consumptive use of 4,219 MGD [52], data centers’ water consumption in 2030 is projected to represent approximately 4% to 7% under the Baseline scenario and 2% to 4% under the Optimistic scenario, depending on the IT load growth rates. These shares are notably higher than the corresponding withdrawal shares, reflecting the higher consumptive ratio of data center water use compared with other users served by public water systems. 

While these figures represent national aggregates and already indicate a non-negligible impact, the implications of data centers’ consumptive water use are more appropriately assessed at local and seasonal scales, where certain public water systems may experience disproportionately greater impacts. 

### **4.2.3 New water capacity demand** 

A data center’s total water capacity requirement, in terms of maximum daily water demand, is a critical planning consideration and may not be fully supported by existing public water systems. To quantify this water capacity requirement, the average daily demand (ADD) needs to be adjusted using a peaking factor. In this study, we review multiple public records and _conservatively_ adopt an average peaking factor of 4.5, reflecting the practice that data centers often provision water capacity with worst-case considerations and 

19 



![Figure](assets/figure_0005_page_0020.svg)**_Figure 5:_** _Valuation of the new water infrastructure capacity to meet U.S. data centers’ new peak water demand from 2024 to 2030 under different scenarios for low, mid, and high growth projections._ 

redundancy, analogous to approaches used in power infrastructure planning. As evaporative-assisted cooling becomes more common, it can improve the annual water efficiency while simultaneously increasing the peaking factor, since such systems typically operate only 5 to 15% of the year. For example, it is common for leading technology companies’ planned peaking factors to range from 6 to 8 or even higher for their stateof-the-art data centers under construction. Additional details regarding the peaking factor are available in Appendix A.6. 

The results are summarized in Table 3. While we aggregate U.S. data centers’ water capacity needs, it is important to note that an individual data center’s actual peak daily water use can be lower than its total capacity requirement (which typically accounts for redundancy) and may not align nationally on the same day, as the hottest days driving the peak water demand generally occur asynchronously across regions. 

From 2024 to 2030, the estimated new water capacity needed to support U.S. data centers’ peak water demand ranges from 697 to 1,451 MGD under the Baseline scenario. For comparison, New York City delivers approximately 1 billion gallons of water per day to nearly 9 million residents in 2024 [53]. Hypothetically, if all the new water capacity needed by U.S. data centers from 2024 to 2030 in the high-growth case were pooled, it could supply New York City for most of the year, except for the few peak days in the summer. While supplying water at the scale of New York City is far more complex than simple aggregation, this comparison highlights the magnitude of U.S. data centers’ total water capacity needs due to their high peaking factors. 

Under the Optimistic scenario, which assumes substantial water reductions, the new total water capacity need decreases to between 227 and 604 MGD, depending on IT load growth rates between 2024 and 2030. Even with continuous efficiency gains corresponding to a 10% compound annual reduction in WUE, highgrowth IT loads can still drive the total water capacity need to a level that, if aggregated hypothetically, could supply roughly half of New York City’s daily public water demand (2024 level). 

### **4.2.4 Valuation of new water capacity** 

We now quantify the valuation of the total new water capacity needed for U.S. data centers from 2024 to 2030. To establish a plausible range of cost estimates, we review public reports on 17 water infrastructure projects of varying complexity, type, and geography, including 6 projects specifically related to data centers. Table 11 indicates that a reasonable valuation (in 2021–2025 dollars, without inflation adjustment) for water infrastructure is $10–$40 million per MGD, depending on factors such as project complexity. 

Using this cost range, Figure 5 and Table 3 present the total valuation of the new water capacity needed to support U.S. data centers through 2030. The overall valuation is generally on the order of $10 billion. Specifically, under the Baseline scenario, the valuation ranges from $7–28 billion in the low-growth case and $15–58 billion in the high-growth case. Under the Optimistic scenario, the valuations decrease to $2–9 billion and $6–24 billion in the low-growth and high-growth cases, respectively. 

Although these billion-dollar-level investments are modest compared to the overall AI and data center industry, they may not be affordable for many public water systems that are already grappling with financial challenges and large funding needs [28, 29]. Thus, corporate–community partnerships to expand water infrastructure are particularly important, meeting data centers’ water needs while avoiding increased water rates for local ratepayers amid rising water affordability challenges [31]. 

20 



Importantly, our valuation may differ from the actual new funding requirement, as some of this capacity may already be available and (partially) funded by existing sources such as state–federal support, bonds, or current ratepayer contributions. Nonetheless, similar to “Water Positive” initiatives that replenish water to offset Scope 1 consumption [107, 108], even if the water capacity is already available, data centers can add new capacity to proactively offset their capacity needs, allowing the host community to retain limited available infrastructure capacity for future growth. Further details are provided in Section 5. 

# **5 Recommendations** 

In this section, we provide a set of key recommendations to address the growing water capacity demand of U.S. data centers and its impact on public water systems, while additional recommendations to address the overall water efficiency are available in Appendix F. 

**Recommendation 1: Report peak water usage.** While total annual water withdrawal and consumption are important metrics, data centers’ peak water demand is critical for public water system planning and management. However, data center operators rarely disclose their peak water demand. We therefore recommend that data center annual reports include per-site peak water metrics, including withdrawal, consumption, and discharge, to provide a comprehensive view of operational water impacts on public water systems. Additionally, to assess peak water efficiency, we recommend extending the standard annualized water usage effectiveness (WUE) metric to define the _peak_ WUE (pWUE) as follows. **Definition 1 (pWUE)** _The peak water usage effectiveness, pWUE, has a unit of L/kWh and is defined as_ 



$$
pWUE = max t\in{}{}T WaterConsumption(t) ITEnergy(t) , (2)
$$

_where WaterConsumption_ ( _t_ ) _and ITEnergy_ ( _t_ ) _denote the total Scope 1 water consumption and IT energy consumption on day t, respectively, and T represents the period considered, such as a calendar or fiscal year._ 

The pWUE metric can be determined from either actual measurements for post-hoc reporting and analysis, or from projected values for planning purposes. Additionally, when the IT load remains relatively flat, pWUE can be simply expressed as _pWUE_ = _β · WUE_ , where _β ∈_ [1 _, ∞_ ) is the daily peaking factor (either measured or planned), and _WUE ≥_ 0 is the annual WUE commonly reported in the literature [2] and in industry disclosures [89]. 

**Recommendation 2: Develop corporate-community partnerships.** The lack of technical and financial capacity remains a critical challenge for many U.S. public water systems [28]. While some state laws require large water users to make “pro rata” fair contributions to fund water infrastructure expansion [32,33], additional corporate-community partnerships that go beyond these legal requirements may further strengthen the resilience of water infrastructure in host communities. By providing funding or technical assistance for local water system upgrades, data centers can help mitigate the added strain their operations place on public water supplies. Additionally, sourcing non-potable water for industrial use, such as recycled or reclaimed wastewater, can help reduce demand on municipal drinking water supplies for both data center operations and community needs [119]. We therefore recommend that data centers actively participate in these partnerships, aligning corporate responsibility with community well-being and long-term water resilience. 

**Recommendation 3: “Pipe Neutral” commitments.** Technology companies have increasingly committed to “Water Positive” initiatives by replenishing more water than their Scope 1 consumption [107,108]. However, these replenishment efforts focus on total volumetric benefits and may not address the impacts of data centers’ peak water use on public water systems. As a complement to the recent community-first AI approaches [3], we recommend that data centers add new water capacity to meet their needs, avoiding water rate increases for local ratepayers while allowing the host community to retain limited water infrastructure capacity for future growth. We refer to this approach as “Water Capacity Neutral,” or colloquially, “Pipe Neutral.” 

To quantify the net contribution of a data center to public water systems, we formally define a metric called the Water Capacity Impact ( `WCI` ) score as follows. **Definition 2 (** `WCI` **)** _The Water Capacity Impact (_ _`WCI` ) score is defined as_ 



$$
WCI = CAdded -CAllocated CAvailable , (3)
$$

21 



_where CAdded, CAllocated, and CAvailable denote the water capacity added by the data center, allocated to the data center, and currently available for allocation in the host community, respectively._ 

A `WCI` score less than _−_ 1 indicates that the available water capacity is insufficient to meet the data center’s needs; a score between _−_ 1 and 0 indicates a net resource usage of the water capacity; and a score greater than 0 indicates that the data center contributes more water capacity to the public water system than it needs, representing a positive contribution. A concrete example is a large technology company’s recent commitment of up to $400 million to local public water infrastructure to support its new Louisiana data centers [16], effectively reversing the otherwise negative `WCI` score. 

**Recommendation 4: Coordinated water-power planning.** Water-based evaporative cooling uses water, but can reduce peak power demand by 10–35% compared to waterless dry cooling [47, 63]. Thus, water used for evaporative assistance—even if applied only on the hottest days—functions like a “water battery,” alleviating peak-hour demand on the electrical grid during the summer. Indeed, while regional water availability remains important, evaporative cooling, if used responsibly, can be an economically competitive approach to shaving peak power demand compared to traditional generation methods (Appendix G). This aligns with a leading technology company’s statement that “water is the most efficient means of cooling in many places” [17]. In addition, some large data centers have recently secured up to 8 MGD of peak water capacity to reduce the electricity demand during the summer [18,21], with associated water infrastructure upgrades costing hundreds of millions of dollars [27]. 

On the other hand, opportunistically utilizing available surplus power, such as solar power during summer midday when water evaporative rates are highest, can reduce peak water demand without overloading the grid. This strategy is particularly beneficial when public water systems are stressed during heatwaves. 

We therefore recommend a coordinated water-power strategy to optimally manage the tradeoff: (1) responsibly leveraging water to alleviate grid stresses while minimizing impacts on local water resources; and (2) opportunistically utilizing power to reduce the burden on public water systems without overloading the grid. 

# **6 Conclusion** 

In this paper, we examine data centers’ direct water withdrawal in the United States and their impacts on public water systems. We first show that, compared to typical water users such as residential buildings, data centers have a higher consumptive ratio and a higher peaking factor, even when their annual water use remains modest. Using public sources including industrial disclosures and government records, we quantify the U.S. data center water use from 2024 to 2030. Our analysis shows that, under the Baseline scenario, data centers’ total water withdrawal in 2030 accounts for 0.6%–1.1% of total public water withdrawals in the contiguous United States, while their water consumption reaches 4%–7% of total public water consumption due to the high consumptive ratio. Most importantly, under the Baseline scenario, U.S. data centers are projected to require 697–1,451 MGD of new water capacity, valued at up to $58 billion and comparable to New York City’s average daily supply of roughly 1,000 MGD. Under an Optimistic scenario with a 10% compound annual reduction in water use intensity, these figures decrease to approximately 30–50% of the Baseline scenario values. While these numbers are national aggregates, local and seasonal impacts on specific public water systems may be disproportionately larger. 

Finally, we provide recommendations to address the growing water demand of U.S. data centers, including reporting peak water use, developing corporate-community partnerships, adopting a Water Capacity Neutral approach (colloquially “Pipe Neutral”) to allow host communities to retain limited water capacity, and implementing coordinated water-power planning to optimally manage the tradeoff. Together, these strategies enhance operational efficiency, protect local water systems, and ensure long-term community resilience while supporting technological advancement and data center growth. 

# **Acknowledgement** 

The authors would like to thank Lisa McFadden (Water Environment Federation) and Gregory Pierce (UCLA) for their feedback and comments on the initial draft of this paper. 

22 



# **References** 

- [1] Mariano-Florentino Cu´ellar, Jeff Dean, Finale Doshi-Velez, John Hennessy, Andy Konwinski, Sanmi Koyejo, Pelonomi Moiloa, Emma Pierson, and David Patterson. Shaping ai’s impact on billions of lives. _Commun. ACM_ , 69(1):54–65, December 2025. 

- [2] Arman Shehabi, Sarah J. Smith, Alex Hubbard, Alex Newkirk, Nuoa Lei, Md Abu Bakar Siddik, Billie Holecek, Jonathan Koomey, Eric Masanet, and Dale Sartor. 2024 United States data center energy usage report. _Lawrence Berkeley National Laboratory LBNL-2001637_ , December 2024. 

- [3] Brad Smith. Building community-first AI infrastructure. `https://blogs.microsoft.com/ on-the-issues/2026/01/13/community-first-ai-infrastructure/` , 2026. 

- [4] Faith McLellan. AI data centres raise public health concerns. _The Lancet_ , 407(10524):119–120, 2026. 

- [5] Tianqi Xiao, Francesco Fuso Nerini, H Damon Matthews, Massimo Tavoni, and Fengqi You. Environmental impact and net-zero pathways for sustainable artificial intelligence servers in the USA. _Nature Sustainability_ , pages 1–13, 2025. 

- [6] Mistral AI. Our contribution to a global environmental standard for AI. `https://mistral.ai/news/ our-contribution-to-a-global-environmental-standard-for-ai` , July 2025. 

- [7] Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of BLOOM, a 176B parameter language model. _Journal of Machine Learning Research_ , 24(253):1–15, 2023. 

- [8] David Patterson, Joseph Gonzalez, Urs H¨olzle, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. The carbon footprint of machine learning training will plateau, then shrink. _Computer_ , 55(7):18–28, 2022. 

- [9] Alex de Vries-Gao. The carbon and water footprints of data centers and what this could mean for artificial intelligence. _Patterns_ , 7(1):101430, 2026. 

- [10] Yuelin Han, Zhifeng Wu, Pengfei Li, Adam Wierman, and Shaolei Ren. The unpaid toll: Quantifying and address the public health impact of data centers. _arXiv 2412.06288_ , 2024. 

- [11] Yu Tao and Peng Gao. Global data center expansion and human health: A call for empirical research. _Eco-Environment and Health_ , 4(3):100157, 2025. 

- [12] U.S. Department of Energy. Recommendations on powering artificial intelligence and data center infrastructure, Jul. 2024. 

- [13] Tyler Norris, Timothy Profeta, Dalia Patino-Echeverri, and Adam Cowie-Haskell. Rethinking load growth: Assessing the potential for integration of large flexible loads in U.S. power systems. 2025. 

- [14] U.S. EPA. EPA hosts roundtable discussion with data center coalition on clean air resources and energy reliability. `https://www.epa.gov/newsreleases/ epa-hosts-roundtable-discussion-data-center-coalition-clean-air-resources-and-energy` , January 2026. 

- [15] Equinix. AI’s engine room: Inside the high-performance data centers powering the future. 2025. 

- [16] Amazon. Amazon to invest $12 billion in first data center campuses in Louisiana. `https: //www.aboutamazon.com/news/company-news/amazon-data-center-louisiana-new-jobs` , February 2026. 

- [17] Google. Environmental report. `https://www.gstatic.com/gumdrop/sustainability/ google-2025-environmental-report.pdf` , 2025. 

23 



- [18] City of Lebanon, Indiana. Water and wastewater agreement: Orla, LLC. `https://lebanon.in.gov/ wp-content/uploads/2025/01/Project-Domino-Water-and-Wastewater-Agreement-Final-11. 03.25.pdf` , November 2025. 

- [19] Meta. Meta’s new data center in Lebanon, Indiana marks a milestone AI investment. `https://about.fb.com/news/2026/02/ metas-new-data-center-lebanon-indiana-marks-milestone-ai-investment/` , February 2026. 

- [20] 10/12 Industry Report. Meta leader addresses Louisiana data center water usage concerns. `https://www.1012industryreport.com/technology/ai/ meta-leader-addresses-louisiana-data-center-water-usage-concerns/` , February 2026. 

- [21] The Western Virginia Water Authority. Utility services funding agreement between Western Virginia Water Authority and Helio Capital LLC. `https://www.westernvawater.org/home/ showpublisheddocument/14540/639017477207000000` , October 2025. 

- [22] U.S. EPA. Information about public water systems. `https://www.epa.gov/dwreginfo/ information-about-public-water-systems` . 

- [23] U.S. EPA. Drinking water infrastructure resilience and sustainability research. `https://www.epa. gov/water-research/drinking-water-infrastructure-resilience-and-sustainability-research` . 

- [24] Susquehanna River Basin Commission. Water use by data centers in the Susquehanna River Basin: FAQ. `https://www.srbc.gov/about/news/docs/srbc-data-centers-faq.pdf` , 2025. 

- [25] The New York Times. Their water taps ran dry when Meta built next door. `https://www.nytimes. com/2025/07/14/technology/meta-data-center-water.html` , July 2025. 

- [26] Hanna Pampaloni. Potomac water supply program. `https://www.loudounwater.org/ potomac-water-supply-program` , 2025. 

- [27] Roanoke Rambler. Water Authority prepares key vote on potential data center, after Botetourt spent months wooing ’Project Raspberry’, September 2025. 

- [28] U.S. EPA. Drinking water infrastructure needs survey and assessment: The 7th report to Congress. _EPA 810R23001_ , September 2023. 

- [29] U.S. EPA. 2022 clean watersheds needs survey report to Congress. _EPA 832-R-24-002_ , April 2024. 

- [30] The Value of Water Campaign. Tapping potential: The economic benefits of investing in water infrastructure. `https://thevalueofwater.org/econimpact` , 2025. 

- [31] U.S. EPA. Water affordability needs assessment: Report to congress. _EPA 830-R-24-015_ , December 2024. 

- [32] Botetcourt County, Virginia. Google data center water. `https://www.botetourtva.gov/1023/ Google-Data-Center-Water` , February 2026. 

- [33] The Western Virginia Water Authority. Water supply and infrastructure planning and development services agreement between Western Virginia Water Authority and Botetcourt County. `https://www. westernvawater.org/home/showpublisheddocument/14538/639017476820900000` , October 2025. 

- [34] Spectrum News 1. Port Washington data center moving ahead as new report urges cities to plan for water supply. `https://spectrumnews1.com/wi/milwaukee/news/2025/09/23/ data-center-port-washington` , October 2025. 

- [35] Wisconsin Department of Natural Resources. Information request for dnr environmental analysis (regarding the data center project in Port Washington). `https://midwestadvocates.org/wp-content/ uploads/2026-01-02-PRR-Response-Vantage-EAS-Documents.pdf` , January 2026. 

24 



- [36] Vantage. OpenAI, Oracle and Vantage data centers announce Stargate data center site in Wisconsin. `https://vantage-dc.com/news/ openai-oracle-and-vantage-data-centers-announce-stargate-data-center-site-in-wisconsin/` , 2025. 

- [37] City of Port Washington, WI. Water utility. `https://www.portwashingtonwi.gov/departments/ public-works/water-utility` . 

- [38] Susquehanna River Basin Commission. Understanding water rights. `https://www.srbc.gov/ our-work/pamphlets/understanding-water-rights.html` . 

- [39] Oregon Public Broadcasting. As Google’s water demands grow, The Dalles aims to pull more from Mount Hood forest. `https://www.opb.org/article/2026/01/15/ as-googles-water-demands-grow-the-dalles-aims-to-pull-more-from-mount-hood-forest/` , January 2026. 

- [40] Citizens Energy Group. Citizens – Lebanon water supply program. `https://info. citizensenergygroup.com/clws` , 2025. 

- [41] WSLS. Google data center water estimates go public, residents in Roanoke and Botetourt react, February 2026. 

- [42] U.S. EPA. Water infrastructure and resiliency finance center. `https://www.epa.gov/ waterfinancecenter` . 

- [43] California State Water Resources Control Board. California drinking water-related statutes and regulations. `https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/Lawbook. html` , 2025. 

- [44] CNBC. To land Meta’s massive $10 billion data center, Louisiana pulled out all the stops. will it be worth it? `https://www.cnbc.com/2025/06/25/ meta-massive-data-center-louisiana-cost-jobs-energy-use.html` , June 2025. 

- [45] The Associated Press. OpenAI shows off Stargate AI data center in Texas and plans 5 more elsewhere with Oracle, Softbank. `https://apnews.com/article/ openai-stargate-oracle-data-center-0b3f4fa6e8d8141b4c143e3e7f41aba1` , September 2025. 

- [46] Steve Solomon. Sustainable by design: Next-generation datacenters consume zero water for cooling. `https://www.microsoft.com/en-us/microsoft-cloud/blog/2024/12/09/ sustainable-by-design-next-generation-datacenters-consume-zero-water-for-cooling/` , 2024. 

- [47] Amazon. FAQ: Why does AI use water? `https://sustainability.aboutamazon.com/stories/ just-back-from-weftec-why-does-ai-use-water` , October 2025. 

- [48] E3: Energy and Environmental Economics. Forecasting largeloads in the age of AI and data centers. _White Paper_ , December 2025. 

- [49] IEA. Energy and AI. `https://www.iea.org/reports/energy-and-ai` , 2025. 

- [50] S. N. Ahmed, K. Bencala, S. Nummer, C. L. Schultz, and A. Seck. 2025 Washington Metropolitan Area water supply study: Demand and resource availability forecast for the year 2050. _Interstate Commission on the Potomac River Basin (ICPRB REPORT NO. ICP-596)_ , December 2025. 

- [51] West Des Moines Water Works. About us. `https://www.wdmww.com/about-us.aspx` . 

- [52] Laura Medalie, Amy E. Galanter, Anthony J. Martinez, Althea A. Archer, Carol L. Luukkonen, Melissa A. Harris, and Jonathan V. Haynes. Water use across the conterminous United States, water years 2010–20. Professional Paper 1894-D, U.S. Geological Survey, Reston, VA, 2025. Chapter D of U.S. Geological Survey Integrated Water Availability Assessment—2010–20. 

25 



- [53] NYC Department of Environmental Protection. One water NYC: 2025 water demand management. `https://www.nyc.gov/assets/dep/downloads/pdf/water/drinking-water/ water-conservation-report2025.pdf` , June 2025. 

- [54] OECD Data. Water withdrawals. `https://data.oecd.org/water/water-withdrawals.htm` . 

- [55] Paul Reig. What’s the difference between water use and water consumption? _World Resources Institute Commentary_ , 2013. `https://www.wri.org/insights/ whats-difference-between-water-use-and-water-consumption` . 

- [56] Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. Making AI less ’thirsty’. _Commun. ACM_ , 68(7):54–61, June 2025. 

- [57] Delaware Department of Natural Resources and Environmental Control. Application for a costal zone act status decision: Project Washington. `https://documents.dnrec.delaware.gov/Admin/ Public-Notices/CCE20250425/status-decision-application.pdf` , October 2025. 

- [58] Equinix. How data centers optimize energy and water for cooling solutions. 2025. 

- [59] Marcus Hopwood and Lindsay Schulz. Top 3 liquid cooling myths debunked. _Equinix Blog_ , June 2024. 

- [60] David E. Sickinger, Otto D. Van Geet, Suzanne A. Belmont, Thomas Carter, and David Martinez. Thermosyphon cooler hybrid system for water savings in an energy-efficient HPC data center: Results from 24 months and the impact on water usage effectiveness. _The U.S. National Renewable Energy Lab. (NREL) Technical Report NREL/TP-2C00-72196_ , September 2018. 

- [61] Ali Heydari, Bahareh Eslami, Vahideh Radmard, Fred Rebarber, Tyler Buell, Kevin Gray, Sam Sather, and Jeremy Rodriguez. Power usage effectiveness analysis of a high-density air-liquid hybrid cooled data center. In _International Electronic Packaging Technical Conference and Exhibition_ , volume 86557, page V001T01A014. American Society of Mechanical Engineers, 2022. 

- [62] Water-AI Nexus Center of Excellence. Principles for sustainable water use by data centers: Building more effective public–private collaboration. _Insight Report_ , 2025. `https://water-ai-nexus.org/ insight-report` . 

- [63] Urs H¨olzle. Our commitment to climate-conscious data center cooling. _Google: The Keyword_ , 2022. 

- [64] Leila Karimi, Leeann Yacuel, Joseph Degraft-Johnson, Jamie Ashby, Michael Green, Matt Renner, Aryn Bergman, Robert Norwood, and Kerri L. Hickenbottom. Water-energy tradeoffs in data centers: A case study in hot-arid climates. _Resources, Conservation and Recycling_ , 181:106194, 2022. 

- [65] Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga, Jinshi Huang, Charles Bai, et al. Sustainable AI: Environmental implications, challenges and opportunities. In _Proceedings of Machine Learning and Systems_ , volume 4, pages 795–813, 2022. 

- [66] Yanran Wu, Inez Hua, and Yi Ding. Not all water consumption is equal: A water stress weighted metric for sustainable computing. _SIGENERGY Energy Inform. Rev._ , 5(2):84–90, August 2025. 

- [67] U.S. Energy Information Administration. U.S. electric power sector continues water efficiency gains. `https://www.eia.gov/todayinenergy/detail.php?id=56820` , June 2023. 

- [68] U.S. Energy Information Administration. Electricity data browser. `https://www.eia.gov/ electricity/data/browser/` . 

- [69] California State Water Resources Control Board. Statutory water rights law, January 2026. 

- [70] Apple. Apple’s water strategy. `https://www.apple.com/environment/pdf/Apples_Water_ Strategy.pdf` , March 2025. 

26 



- [71] Ceres. 2025 valuing water finance initiative benchmark. `https://www.ceres.org/water/ valuing-water-finance-initiative/benchmark` , 2025. 

- [72] U.S. EPA. Primer for municipal wastewater treatment systems. `https://www.epa.gov/sites/ default/files/2015-09/documents/primer.pdf` , September 2004. 

- [73] U.S. EPA. Safe Drinking Water Act (SDWA). `https://www.epa.gov/sdwa` . 

- [74] U.S. EPA. Summary of the Clean Water Act. `https://www.epa.gov/laws-regulations/ summary-clean-water-act` . 

- [75] U.S. Energy Information Administration. Investor-owned utilities served 72% of U.S. electricity customers in 2017. `https://www.eia.gov/todayinenergy/detail.php?id=40913` , 2019. 

- [76] U.S. Bureau of Labor Statistics. Cpi inflation calculator. `https://www.bls.gov/data/inflation_ calculator.htm` . 

- [77] Zia J. Lyle, Jeanne M. VanBriesen, and Constantine Samaras. Climate change risk index and municipal bond disclosures of United States drinking water utilities. _Communications Earth and Environment_ , 7(1):68, 2026. 

- [78] U.S. EPA. Community water system survey, 2025. Accessed March 15, 2026. 

- [79] U.S. Environmental Protection Agency. 2006 community water system survey volume II: Detailed tables and methodology. Report EPA-815-R-09-002, Office of Water (4606M), Washington, D.C., May 2009. 

- [80] U.S. EPA. National characteristics of drinking water systems serving 10,000 or fewer people. Report EPA 816-R-10-022, Office of Water (4606M), Washington, D.C., July 2011. 

- [81] California State Water Resources Control Board. California drinking water state revolving fund policy: Capacity limitations. `https://www.waterboards.ca.gov/drinking_water/services/funding/ documents/srf/dwsrf_policy/appendix_i.pdf` , 2014. 

- [82] California Department of Water Resources. 2025 annual water supply and demand assessment summary report: Bulletin 161-2025. Report Bulletin 161-2025, Water Use Efficiency Branch, Sacramento, CA, September 2025. 

- [83] Norwalk, Iowa. Public statement regarding this week’s council agenda and a proposed future data center project. `https://www.norwalk.iowa.gov/news_detail_T5_R273.php` , March 2026. 

- [84] Virginia Department of Environmental Quality. Report on the Town of Leesburg, Loudoun County, voluntary settlement agreement. `https://www.dhcd.virginia.gov/sites/default/files/DocX/ clg/town-of-leesburg/final-clg-leesburg-loudoun-vsa-report-withappendix-4.30.24. pdf` , April 2024. 

- [85] Loudoun Now. Leesburg council approves data center water agreement. `https://www. loudounnow.com/news/leesburg-council-approves-data-center-water-agreement/article_ b92096b0-a8f9-11ef-a427-a36c0466d0ef.html` , November 2024. 

- [86] The Dalles, Oregon. Generational investments: Drinking water system. `https://www.thedalles. org/department/public_works/master_plans/water_master_plan.php` , 2025. 

- [87] Wisconsin Pubic Radio. Microsoft data centers will use up to 8.4M gallons of water each year, records show. `https://www.wpr.org/news/ microsoft-data-centers-8-million-gallons-water-each-year` , September 2025. 

- [88] Hannah Ritchie and Max Roser. Water use and stress. _Our World in Data_ , 2017. 

27 



- [89] Meta. 2025 sustainability report. `https://sustainability.atmeta.com/wp-content/uploads/ 2025/08/Meta_2025-Sustainability-Report_.pdf` , 2025. 

- [90] Microsoft. Environmental sustainability report. `https://www.microsoft.com/en-us/ corporate-responsibility/sustainability/report` , 2024. 

- [91] Consor. Water system master plan update for The Dalles, Oregon, November 2024. 

- [92] California State Water Resources Control Board. California regulations related to drinking water. `https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/documents/ lawbook/drinking-water-regulations-2025.pdf` , August 2025. 

- [93] SPX Cooling. Water usage calculator. `http://spxcooling.com/green/leed/ water-usage-calculator/` . 

- [94] Microsoft. Microsoft datacenters in Arizona. `https://local.microsoft.com/wp-content/ uploads/2025/10/Microsoft-datacenters-in-Arizona.pdf` , October 2025. 

- [95] West Des Moines, Iowa. Building permit reports. `https://www.wdm.iowa.gov/government/ development-services/building-inspection/building-permit-reports/` . 

- [96] California Legislative Analyst’s Office. Residential water use trends and implications for conservation policy. `https://lao.ca.gov/Publications/Report/3611` , March 2017. 

- [97] Uptime Institute. Explaining the Uptime Institute’s tier classification system (April 2021 update). `https://journal.uptimeinstitute.com/ explaining-uptime-institutes-tier-classification-system/` . 

- [98] Associated Press. Artificial intelligence technology behind ChatGPT was built in Iowa – with a lot of water. `https://apnews.com/article/ chatgpt-gpt4-iowa-ai-water-consumption-microsoft-f551fde98083d17a7e8d904f8be822c4` , September 2023. 

- [99] Jon Gorey. Data drain: The land and water impacts of the AI boom. _Lincoln Institute of Land Policy_ , October 2025. 

- [100] Ben Townsend. A new water infrastructure project is increasing water security in Oregon. `https://blog.google/company-news/outreach-and-initiatives/sustainability/ the-dalles-oregon-water/` , 2025. 

- [101] Katherine Bourzac. Fixing AI’s energy crisis. _Nature_ , 634(8035):766–768, Oct 2024. [102] One Newton Water Resources Analysis. Water resources analysis: A collaborative effort between Newton County, the Newton County Water and Sewerage Authority, and the City of Covington. `https://cityofcovington.org/ckeditorfiles/files/2025_Water_ OneWaterResourcesAnalysis2024.pdf` , December 2024. 

- [103] Md Abu Bakar Siddik, Arman Shehabi, and Landon Marston. The environmental footprint of data centers in the United States. _Environmental Research Letters_ , 16(6):064017, 2021. 

- [104] Mohammad A. Islam, Kishwar Ahmed, Hong Xu, Nguyen H. Tran, Gang Quan, and Shaolei Ren. Exploiting spatio-temporal diversity for water saving in geo-distributed data centers. _IEEE Transactions on Cloud Computing_ , 6(3):734–746, 2018. 

- [105] Yankai Jiang, Rohan Basu Roy, Raghavendra Kanakagiri, and Devesh Tiwari. Waterwise: Cooptimizing carbon- and water-footprint toward environmentally sustainable cloud computing. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ , PPoPP ’25, page 297–311, New York, NY, USA, 2025. Association for Computing Machinery. 

28 



- [106] Kaveh Madani. Global water bankruptcy: Living beyond our hydrological means in the post-crisis era. _United Nations University Institute for Water, Environmentand Health (UNU-INWEH)_ , 2026. 

- [107] Google. 2025 water stewardship project portfolio. `https://sustainability.google/reports/ 2025-google-water-stewardship-project-portfolio/` , 2025. 

- [108] LimnoTech. Meta volumetric water benefits: 2024 report. `https://sustainability.atmeta.com/ water/` , August 2025. 

- [109] Melanie Nakagawa. The journey to water positive. `https://blogs.microsoft.com/on-the-issues/ 2023/03/22/water-positive-climate-resilience-open-call/` , 2023. 

- [110] Amazon. Sustainability — water stewardship. `https://sustainability.aboutamazon.com/ environment/the-cloud/water-stewardship` , 2023. 

- [111] Water Footprint Network. Glossary page. `https://www.waterfootprint.org/water-footprint-2/ glossary/` , 2024. 

- [112] Mesfin M. Mekonnen and Arjen Y. Hoekstra. The green, blue and grey water footprint of farm animals and animal products. Technical Report 48, UNESCO-IHE Institute for Water Education, Delft, Netherlands, 2010. 

- [113] Andrew Batson. North America data center report, midyear 2025. _JLL Research Report_ , 2025. 

- [114] McKinsey. The data center balance: How U.S. states can navigate the opportunities and challenges. _White Paper on Public Sector Practice_ , August 2025. 

- [115] QTS. 2024 sustainability report. `https://qtsdatacenters.com/wp-content/uploads/2025/10/ 2024-QTS-Sustainability-Report.pdf` , 2025. 

- [116] Microsoft. Environmental sustainability report. `https://www.microsoft.com/en-us/ corporate-responsibility/sustainability/report` , 2025. 

- [117] The New York Times. Microsoft pledged to save water. In the A.I. era, it expects water use to soar. `https://www.nytimes.com/2026/01/27/technology/microsoft-water-ai-data-centers. html` , January 2026. 

- [118] Los Angeles Department of Water and Power. Los angeles water systems. `https://www.ladwp.com/ who-we-are/water-system` , 2025. 

- [119] The Greater Memphis Chamber. xAI expands Memphis footprint with major property acquisition in Southwest. _Press Release_ , 2025. 

- [120] EPRI. Powering intelligence: Analyzing artificial intelligence and data center energy consumption. _White Paper on Technology Innovation Report_ , 2024. `https://www.epri.com/research/products/ 3002028905` . 

- [121] Microsoft. Datacenter sustainability efficiency metrics. `https://datacenters.microsoft.com/ sustainability/efficiency/` , 2025. 

- [122] Microsoft. 2025 Environmental Data Fact Sheet. `https://cdn-dynmedia-1.microsoft. com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/ 2025-Microsoft-Environmental-Data-Fact-Sheet-PDF.pdf` , 2025. 

- [123] Meta. 2025 Environmental Data Index. `https://sustainability.atmeta.com/wp-content/ uploads/2025/10/Meta_2025-Environmental-Data-Index.pdf` , 2025. 

- [124] Apple. 2025 Environmental Progress Report. `https://www.apple.com/environment/pdf/Apple_ Environmental_Progress_Report_2025.pdf` , 2025. 

29 



- [125] Amazon. AWS Cloud Sustainability. `https://sustainability.aboutamazon.com/ products-services/aws-cloud` , 2025. 

- [126] Digital Realty. 2024 Impact Report. `https://go2.digitalrealty.com/rs/087-YZJ-646/images/ Report_Digital_Realty_2024_Impact_Report.pdf` , 2025. 

- [127] Equinix. 2024 sustainability report. `https://sustainability.equinix.com/` , 2024. 

- [128] CyrusOne. 2025 Sustainability Report. `https://www.cyrusone.com/hubfs/Website%20Documents% 202025/2025%20Sustainability%20Report.pdf?hsLang=en` , 2025. 

- [129] Switch. Switch Sustainability. `https://www.switch.com/sustainability/` , 2025. 

- [130] Switch. 2024 ESG Report. `https://switchdotcom.s3-sites.data.switch.com/media/reports/ esg/Switch-ESG-Report-2024.pdf` , 2024. 

- [131] NTT. 2025 Global Sustainability Report. `https://services.global.ntt/en-US/pdf-viewer/ 5BC7B7C8-27BB-4636-ACC7-7266C110559E` , 2025. 

- [132] EdgeConneX. 2024 Sustainability Report. `https://www.edgeconnex.com/wp-content/uploads/ 2025/09/ECX-Sustainability-Report-2024_FINAL.pdf` , 2025. 

- [133] City of Phoenix, AZ. Water and wastewater unit cost study (final), August 2024. 

- [134] U.S. EPA. The water infrastructure finance and innovation act (WIFIA) program: Pure water san diego project. 2020. 

- [135] Leesburg, Florida. Ribbon cutting scheduled for Leesburg’s wastewater treatment facility project. `https://www.leesburgflorida.gov/news_detail_T6_R461.php` , April 2022. 

- [136] BAE Urban Economics. Data center market study for Leesburg, Virginia. `https: //www.dhcd.virginia.gov/sites/default/files/Docx/clg/town-of-leesburg/ consultant-reports-supporting-town-reply.pdf` , October 2023. 

- [137] Virginia Department of Environmental Quality. Issued air permits for data centers (registration number: 74220), March 2025. 

- [138] The County of Loudoun, Virginia. Responsive pleading of the County of Loudoun and supporting narrative, information, citations, and materials. `https://www.dhcd.virginia.gov/sites/default/ files/Docx/clg/pending-cases/responsive-pleading-loudoun-7.7.2023.pdf` , July 2023. 

- [139] Microsoft. Microsoft datacenters in Iowa. `https://datacenters.microsoft.com/wp-content/ uploads/2024/08/Iowa%20%28Central%20US%29.pdf` , August 2024. 

- [140] City of Lebanon, Indiana. Memorandum of Understanding: Parkview Hospital water supply and wastewater capacity pre-allocation agreement. `https://lebanon.in.gov/wp-content/uploads/2026/01/ Project-Parkview-Water-Supply-and-Wastewater-Capacity-Pre-Allocation-Agreement.pdf` , January 2026. 

- [141] City of Lebanon, Indiana. Memorandum of Understanding: Project Spring Creek water supply and wastewater capacity pre-allocation agreement, November 2025. 

- [142] City of Lebanon, Indiana. Memorandum of Understanding: Project NewCold, Phase 2 water supply and wastewater capacity pre-allocation agreement, January 2025. 

- [143] City of Lebanon, Indiana. Memorandum of Understanding: Eli Lilly LP1X water supply pre-allocation agreement. `https://lebanon.in.gov/wp-content/uploads/2025/01/ Lilly-LP1X-Pre-Allocation-MOU-Water-2.05.25-final-w-Exhibits.pdf` , February 2025. 

30 



- [144] City of Lebanon, Indiana. Memorandum of Understanding: Eli Lilly Medicine Foundry water supply and wastewater capacity pre-allocation agreement. `https://lebanon.in.gov/wp-content/ uploads/2025/01/Lilly-Medicine-Foundry-Pre-Allocation-MOU-Water-Wastewater-2.05. 25-final-w-Exhibits.pdf` , February 2025. 

- [145] DayZenLLC. Microsoft fourth revised project description – San Jose data center 04 (SJ04, TN: 264764). _California Energy Commission Docket Log_ , July 2025. 

- [146] DayZenLLC. GSOBGF revised project description — Great Oaks south backup generating facility small power plant exemption (TN: 237149). _California Energy Commission Docket Log_ , March 2021. 

- [147] Steve Solomon. Sustainable by design: Next-generation datacenters consume zero water for cooling. _Microsoft AI_ , 2024. 

- [148] Scott Guthrie. Inside the world’s most powerful AI datacenter. _Official Microsoft Blog_ , 2025. 

- [149] Shutong Chen, Zhi Zhou, Fangming Liu, Zongpeng Li, and Shaolei Ren. Cloudheat: An efficient online market mechanism for datacenter heat harvesting. _ACM Trans. Model. Perform. Eval. Comput. Syst._ , 3(3), June 2018. 

- [150] EPRI. DCFlex initiative. `https://msites.epri.com/dcflex` , 2024. 

- [151] Ana Radovanovi´c, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Duarte, Binz Roy, Diyue Xiao, Maya Haridasan, Patrick Hung, Nick Care, Saurav Talukdar, Eric Mullen, Kendal Smith, MariEllen Cottman, and Walfredo Cirne. Carbon-aware computing for datacenters. _IEEE Transactions on Power Systems_ , 38(2):1270–1280, 2023. 

- [152] Jianyi Yang, Pengfei Li, Tongxin Li, Adam Wierman, and Shaolei Ren. Learning-augmented online control for decarbonizing water infrastructures. In _e-Energy_ , 2024. 

- [153] Matthew Hayslep, Edward Keedwell, and Raziyeh Farmani. Leakage prediction in real-world water distribution networks using multi-objective multi-gene genetic programming. _ACM Trans. Evol. Learn. Optim._ , 5(4), December 2025. 

- [154] U.S. Energy Information Administration. How much electricity is lost in electricity transmission and distribution in the United States? `https://www.eia.gov/tools/faqs/faq.php?id=105&t=3` , November 2023. 

- [155] U.S. Energy Information Administration. Construction cost data for electric generators installed in 2023. `https://www.eia.gov/electricity/generatorcosts/` , 2025. 

- [156] Travis W. Shaddox, J. Bryan Unruh, Josh Tapp, Clark D. Brown, Greg Stacey, and Emily Fuger. Survey of water use and management practices on US golf courses from 2005 to 2024. _HortTechnology_ , 35(5):848–857, October 2025. 

- [157] U.S. EPA. Summary of California’s water reuse guideline or regulation for landscaping. `https://www.epa.gov/waterreuse/ summary-californias-water-reuse-guideline-or-regulation-landscaping` . 

- [158] American Water Works Association. _Planning for the Distribution of Reclaimed Water: Manual of Water Supply Practices (M24, Third Edition)_ . American Water Works Association, Denver, CO, 2009. Accessed: 2026-02-25. 

- [159] California Public Utilities Commission. California public utilities code section 2708. `https://docs. cpuc.ca.gov/published/REPORT/99158.htm` , 2009. 

- [160] California Public Utilities Commission. California public utilities commission decisions on water utility service connection moratoriums. `https://docs.cpuc.ca.gov/published/Final_decision/ 77146-01.htm` , 2025. 

31 



# **Appendix** 

# **A Detailed Methodology for Quantifying and Forecasting Data Centers’ Water Demand** 

This section presents the detailed methodology of quantifying and forecasting data centers’ water demand It describes the data sources and calculation procedures used to estimate IT equipment energy consumption (hereafter referred to as IT energy) and water use effectiveness (WUE) under different scenarios from 2024 to 2030, which are subsequently used to quantify the average daily demand (ADD) and water capacity demand measured in terms of the maximum daily demand (MDD). 

## **A.1 Key Metrics and Definitions** 

We first define the key metrics used in this study. Unless otherwise specified, energy, water consumption, and water withdrawal refer to annual values and include only data center operations, excluding office and other non–data-center activities. When converting annual quantities to daily averages, we apply 365 days per year, without distinguishing leap and non-leap years. All energy- and water-related quantities presented in the subsequent analysis are computed based on these definitions. 



$$
WUE = On-site Water Consumption IT Energy (4) PUE = Total Energy IT Energy (5)
$$



Total Energy
IT Energy
(5)



$$
Water Consumptive Ratio = Water Consumption Water Withdrawal (6)
$$



$$
ADD = Water Withdrawal 365 (7)
$$



$$
MDD = ADD \times{}{} Peaking Factor (8)
$$

## **A.2 IT Energy** 

The Lawrence Berkeley National Laboratory (LBNL) report [2] provides a detailed classification of data center types, which are further aggregated into three groups in our study: hyperscale, colocation, and others. As WUEs can differ across data center types, we estimate IT energy at the data center type level to ensure consistent WUE calculation and aggregation. 

LBNL reports total the U.S. data center electricity consumption, PUE, and WUE for 2024–2028 under low and high growth rates (Table 4). In principle, aggregate IT energy can be derived from total energy and PUE using Eq. (5). Over 2024–2028, the reported total data center energy exhibits compound annual growth rates (CAGRs) of approximately 13% and 27% under the low and high rates, respectively. These growth rates are broadly consistent with other projections, which report CAGRs of approximately 20% through 2030 [113] and approximately 22.4% over 2023–2030 under a medium-growth scenario [114]. Given this consistency, the total data center energy for 2029 and 2030 is extrapolated by applying the corresponding CAGRs derived from the 2024–2028 period. 

For PUE, LBNL reports values through 2028. PUE values for 2029 and 2030 are estimated by extending the observed trend, assuming an annual reduction of 0.02. Details regarding WUE are discussed in Section A.3. Using the estimated total energy and PUE values, aggregate IT energy for 2024–2030 is computed accordingly. Details are shown in Table 4. 

However, LBNL does not report the annual distribution of total energy across data center types for 2025 to 2027. As a result, IT energy derived directly from total energy and PUE would only yield aggregate values and would not support type-specific estimation. To address this limitation, we construct type-specific IT energy estimates using a bottom-up approach based on the disaggregated IT energy in 2024 and 2028 reported by LBNL. Therefore, the aggregate IT energy derived from total energy and PUE is not adopted in our type-specific analysis. Instead, it only serves as a reference to evaluate the reasonableness of our resulting bottom-up IT energy estimates. 

32 



**_Table 4:_** _LBNL-reported U.S. data center total energy, PUE, WUE, and aggregate IT energy under low and high assumptions (2024–2030). The values for 2029 and 2030 are extended based on the LBNL values for 2024 to 2028._ 

|**Year**|**WUE(**|**L/kWh)**|**P**|**UE**|**Total Ene**|**rgy (TWh)**|**IT Energ**|**y**|**(TWh)**|
|---|---|---|---|---|---|---|---|---|---|
||Low|High|Low|High|Low|High|Low||High|
|2024|0.39|0.40|1.33|1.45|185.00|232.00|139.10||160.00|
|2025|0.40|0.42|1.23|1.45|203.00|303.00|165.04||208.97|
|2026|0.42|0.44|1.19|1.40|238.00|388.00|200.00||277.14|
|2027|0.44|0.46|1.17|1.37|279.00|481.00|238.46||351.09|
|2028|0.45|0.48|1.15|1.35|325.00|578.00|282.61||428.15|
|2029|0.46|0.49|1.13|1.33|367.25|734.06|325.00||551.92|
|2030|0.47|0.50|1.11|1.31|414.99|932.26|373.87||711.65|



Specifically, in the LBNL report, IT energy is decomposed into server, network, and storage components. Server energy is reported for 2024 and 2028 under low and high rate assumptions for each data center type. In contrast, network and storage energy are reported for 2024–2028 only at the national level, without lowhigh distinctions and without differentiation by data center type. 

Based on the reported server energy values for 2024 and 2028, the CAGRs of server energy over 2024– 2028 are approximately 22% and 31% for hyperscale data centers under the low and high rate assumptions, respectively, and approximately 24% and 34% for colocation data centers. In contrast, server energy in the “Others” category decreases over this period, yielding negative CAGRs of approximately _−_ 7% and _−_ 8% under the low and high assumptions, respectively. These type-specific growth rates are then applied to construct annual server energy trajectories for 2024–2030 by interpolating intermediate years (2025–2027) and extrapolating later years (2029–2030), in line with the extension applied to the aggregate LBNL total energy projections. Although the “Others” category shows higher values under the low assumption than under the high assumption, the terminology of “low” and “high” is consistently defined according to the total server energy assumptions reported by LBNL to maintain consistency across data center types. A mid estimate is further constructed as the arithmetic average of the corresponding low and high estimates. 

The server energy estimates by data center type are summarized in Table 5. 

**_Table 5:_** _Type-specific U.S. data center server energy used in our IT energy estimation (2024–2030). Mid values are defined as the arithmetic average of the corresponding low and high estimates._ 

|**Y**|**Hyp **|**erscale(T**|**Wh)**|**Colo**|**cation(T**|**Wh)**|**Oth**|**ers(T**|**Wh)**|**To**|**tal(TW**|**h)**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**ear**|Low|**Mid**|High|Low|**Mid**|High|Low|**Mid**|High|Low|**Mid**|High|
|2024|48.66|**55.00**|61.33|50.67|**54.67**|58.67|14.00|**14.00**|14.00|113.33|**123.67**|134.00|
|2025|59.30|**69.94**|80.57|62.59|**70.72**|78.84|13.08|**12.98**|12.87|134.98|**153.63**|172.28|
|2026|72.27|**89.06**|105.84|77.32|**91.64**|105.95|12.22|**12.03**|11.83|161.81|**192.72**|223.63|
|2027|88.07|**113.56**|139.05|95.52|**118.95**|142.38|11.42|**11.15**|10.88|195.01|**243.66**|292.30|
|2028|107.33|**145.00**|182.67|118.00|**154.67**|191.33|10.67|**10.34**|10.00|236.00|**310.00**|384.00|
|2029|130.80|**185.39**|239.97|145.77|**201.44**|257.11|9.97|**9.58**|9.19|286.54|**396.41**|506.28|
|2030|159.40|**237.33**|315.26|180.07|**262.79**|345.51|9.31|**8.88**|8.45|348.79|**509.01**|669.22|



For network and storage energy, the LBNL report provides aggregate U.S. totals for 2024–2028. Using the reported values in 2024 and 2028, we compute separate CAGRs for network energy (approximately 26%) and storage energy (approximately 7%). These growth rates are applied to extrapolate total network and storage energy for 2029 and 2030. The resulting annual network and storage energy is then allocated across hyperscale, colocation, and others according to the year-specific server energy shares of the three data center types. This allocation is performed separately under the low and high assumptions. Consistent with the definition adopted above, the terms “low” and “high” refer to the corresponding LBNL total server energy assumptions rather than to alternative allocation rules or proportion levels. The corresponding estimates are summarized in Table 6. 

By summing server, network, and storage energy within each data center type, we obtain IT energy for three data center types in our analysis: Hyperscale, Colocation, and Others. Aggregating across the three 

33 



**_Table 6:_** _Estimated U.S. data center network and storage energy used in our IT energy estimation, 2024–2030. Shares represent each component’s proportion relative to total server energy under the low and high assumptions._ 

|**Y**|**Energy **|**(TWh)**|**Storag**|**e Ratio**|**Netwo**|**rk Ratio**|
|---|---|---|---|---|---|---|
|**ear**|Storage|Network|Low|High|Low|High|
|2024|17.10|9.08|0.15|0.13|0.08|0.07|
|2025|17.75|12.35|0.13|0.10|0.09|0.07|
|2026|18.44|16.18|0.11|0.08|0.10|0.07|
|2027|19.70|19.92|0.10|0.07|0.10|0.07|
|2028|22.02|23.19|0.09|0.06|0.10|0.06|
|2029|23.46|29.32|0.08|0.05|0.10|0.06|
|2030|24.99|37.06|0.07|0.04|0.11|0.06|



types yields total U.S. IT energy. The resulting estimates are summarized in Table 7. 

**_Table 7:_** _Estimated U.S. data center IT energy by data center type in our analysis (2024–2030). IT energy equals the sum of server, network, and storage energy, with network and storage energy allocated across types according to annual server energy shares. Mid values are defined as the arithmetic average of the corresponding low and high estimates._ 

|**Y**|**Hyp **|**erscale(T**|**Wh)**|**Colo**|**cation(T**|**Wh)**|**Oth**|**ers(TW**|**h)**|**Tot**|**al IT(T**|**Wh)**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**ear**|Low|**Mid**|High|Low|**Mid**|High|Low|**Mid**|High|Low|**Mid**|High|
|2024|59.90|**66.81**|73.31|62.38|**66.25**|70.13|17.23|**16.98**|16.74|139.51|**149.85**|160.18|
|2025|72.52|**83.59**|94.65|76.55|**84.58**|92.62|16.00|**15.56**|15.12|165.08|**183.73**|202.38|
|2026|87.73|**104.98**|122.23|93.87|**108.11**|122.35|14.84|**14.25**|13.66|196.43|**227.34**|258.25|
|2027|105.96|**131.93**|157.90|114.93|**138.30**|161.68|13.74|**13.05**|12.35|234.63|**283.28**|331.92|
|2028|127.89|**166.03**|204.18|140.61|**177.23**|213.86|12.71|**11.95**|11.18|281.21|**355.21**|429.21|
|2029|154.89|**209.94**|264.99|172.62|**228.26**|283.91|11.81|**10.98**|10.15|339.31|**449.18**|559.05|
|2030|187.76|**266.12**|344.49|212.11|**294.83**|377.55|10.97|**10.10**|9.24|410.84|**571.05**|731.27|



Comparing the IT energy values in Table 7 (our analysis) and Table 4 (LBNL), we find that the differences are reasonably small. Specifically, from 2024 to 2028, the percentage deviation remains constantly small, ranging from -1.8% to +0.3% under the low assumption and from -6.8% to +0.2% under the high assumption, indicating that our estimates are generally lower than or close to the LBNL values. In 2029–2030, the deviation becomes positive, reaching +4.4% and +9.9% for the low case and +1.3% and +2.8% for the high case, respectively. Although our estimates exceed the LBNL values in these later years, the differences remain moderate and within an acceptable range and align well with other projections through 2030 (e.g., [5,113,114]). 

Overall, the results suggest that our estimation remains close to the LBNL values. Unless otherwise specified, the IT energy values used in this study refer to those reported in Table 7. 

## **A.3 Reference Scenarios** 

This subsection describes the calculation procedures for the Reference (LBNL) and Reference (NS) scenarios. For both scenarios, the respective U.S.-wide average WUE is applied and multiplied by the total IT energy (for all the data center types) to obtain the U.S. data centers’ total water. 

**Reference (LBNL) [2].** The LBNL report provides national-level WUE estimates for 2024–2028 under low and high assumptions only. For 2029–2030, WUE values are estimated by extending the observed trend, assuming an annual increase of 0.01. Water consumption under the LBNL reference is calculated by combining low IT energy with low WUE and high IT energy with high WUE. 

**Reference (NS) [5].** The recent study classifies data centers based on the adoption of advanced liquid cooling (ALC) and reports state-level PUE and WUE values with and without ALC for 2024. The Reference (NS) scenario in our study adopts the baseline case defined in [5], which assumes an initial ALC adoption rate of 5% in 2024, increasing at a CAGR of 20%, together with the corresponding PUE and WUE values reported for that baseline case. Nationwide weighted averages are computed using each state’s total data 

34 



center energy consumption from EPRI [120]. Following the assumed ALC adoption trajectory, the weightedaverage WUE in each year is calculated as: 



$$
WUEyear = (1 -αyear) WUEbase + αyear WUE ALCbase, (9)
$$

where _α_ year denotes the ALC adoption rate in a given year. While the study [5] considers various WUE scenarios and focuses on AI data centers which generally have low WUEs, we apply its (baseline) WUE reference to all data center types solely for orders-of-magnitude cross-checking, helping to prevent overestimation; instead, we rely primarily on the LBNL results as our reference. 

The resulting WUE values for the Reference (LBNL) and Reference (NS) scenarios are summarized in Table 8. Unless otherwise specified, the mid values of water consumption, water withdrawal, ADD and MDD are defined as the arithmetic average of the corresponding low and high quantities. 

**_Table 8:_** _WUE (L/kWh) under Reference, Baseline, Moderate, Optimistic scenarios, 2024–2030._ 

|||**Reference**|||**Bas**|**eline**|||**Mod**|**erate**|||**Opti**|**mistic**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Year**|LB|NL|NS|**U.S.-**|**wide**|Her|Colo|**U.S.-**|**wide**|Her|Colo|**U.S.-**|**wide**|Her|Colo|
||Low|High||Low|High|yp||Low|High|yp||Low|High|yp||
|2024|0.385|0.395|1.582|0.525|0.534|0.546|0.650|0.525|0.534|0.546|0.650|0.525|0.534|0.546|0.650|
|2025|0.401|0.419|1.581|0.541|0.553|0.546|0.650|0.514|0.525|0.519|0.617|0.487|0.497|0.492|0.585|
|2026|0.420|0.444|1.579|0.554|0.566|0.546|0.650|0.500|0.511|0.493|0.586|0.449|0.459|0.443|0.526|
|2027|0.437|0.464|1.577|0.565|0.576|0.546|0.650|0.484|0.494|0.468|0.557|0.412|0.420|0.398|0.473|
|2028|0.450|0.480|1.575|0.573|0.584|0.546|0.650|0.467|0.475|0.445|0.529|0.376|0.383|0.359|0.426|
|2029|0.460|0.490|1.572|0.580|0.589|0.546|0.650|0.449|0.456|0.423|0.503|0.342|0.348|0.323|0.384|
|2030|0.470|0.500|1.569|0.585|0.593|0.546|0.650|0.430|0.436|0.402|0.477|0.311|0.315|0.290|0.345|



## **A.4 Baseline, Moderate, and Optimistic Scenarios** 

This subsection describes the details for the Baseline, Moderate, and Optimistic WUE scenarios. All the three scenarios share the same 2024 WUE values and differ only in their assumed compound annual changes: 0% per year for the Baseline scenario, _−_ 5% per year for the Moderate scenario, and _−_ 10% per year for the Optimistic scenario. Accordingly, the key task in scenario construction is to estimate type-specific WUE values for the base year 2024, from which WUE trajectories for 2025–2030 are derived. 

### **A.4.1 The Year of 2024** 

To estimate the 2024 U.S.-level WUE for different data center types, we select a set of large data center operators to estimate WUE separately for hyperscale and colocation data centers, while WUE for the “Others” data center type is assumed to be zero for a conservative estimate. The estimation relies primarily on publicly available company-wide sustainability reports released for the calendar (or fiscal) year of 2024 released by 12 large operators, including five hyperscale operators and seven colocation operators [17,115,121–132]. We do not link these specific references to individual data center companies in order to avoid direct comparisons and maintain their anonymity while preserving our transparency to the extent possible. 

The selected hyperscale and colocation data center operators collectively represent approximately 86% and 22% of U.S. hyperscale and colocation IT loads in 2024, respectively, based on the mid IT estimates reported in Table 7. Therefore, the WUE values aggregated across these companies provide a reasonable estimate of industry practices. 

Although reporting formats and data granularity vary across companies, the overall data quality is generally sufficient for our analysis. Reported values are followed as closely as possible, and explicit assumptions are introduced only when necessary to ensure internal consistency. Based on these company-level data, U.S.-level type-specific WUE values for 2024 are derived and subsequently used to construct scenariospecific trajectories. 

For each selected operator, we collect (and estimate if needed) its U.S. data center–level IT energy (MWh), total energy (MWh), PUE, water consumption (ML), water withdrawal (ML), WUE (L/kWh), the water consumptive ratio, municipal water ratio, and potable water ratio. The resulting values are summarized in Table 2. Note that, based on publicly disclosed data, we can reliably calculate the U.S. potable water ratio for only one data center operator (94.40% for Hyperscale-1) and the global potable water ratios for 

35 



two other operators. Therefore, we exclude potable ratios in Table 2 and instead present them only in the operator-specific results below. 

The reported values in Table 2 are adopted directly whenever explicitly disclosed. When specific metrics are unavailable, values estimated or assigned based on clearly stated assumptions or publicly available thirdparty information are introduced and identified with an asterisk “<sup>_∗_</sup> ” in Table 2. 

Unless otherwise specified, the municipal and potable water ratios refer to shares of water directly supplied by the municipal/public water systems and potable water relative to the total water withdrawal. If withdrawal-based ratios are explicitly reported, they are used directly; if only consumption-based ratios are disclosed, the corresponding ratios are derived from reported water consumption values and marked with a dagger symbol ( _†_ ). If neither withdrawal- nor consumption-based ratios are available, the corresponding entry is denoted by “–”, indicating that the value is unreported and cannot be reliably estimated, although public potable water is likely used as a general industry practice. Energy and water quantities are rounded to the nearest integer, while all other numerical values are rounded to two decimal places. 

For hyperscale operators, total water consumption generally includes both data center water use and office-related water use. When separate reporting is unavailable, office water _consumption_ is estimated using an industry-standard consumptive ratio of 10% [17], while data center water consumption is estimated using a default consumptive ratio of 75% [50]. For colocation operators, office water use is assumed to be negligible, as they typically employ significantly fewer staff than hyperscalers, which maintain large teams of software developers and corporate employees; accordingly, reported water consumption is attributed primarily to data center operations. Additional default settings and estimation details are provided in the operator-specific calculation procedures presented below. 

**Hyperscale-1.** This operator reports data center water withdrawal and water consumption at the location/site level. We aggregate all U.S. data center sites to obtain total U.S. data center water withdrawal (approximately 29,510 ML) and water consumption (approximately 23,120 ML), yielding a water consumptive ratio of approximately 0.78. The operator reports North America–level electricity consumption (treated as U.S.) but does not separately disclose electricity use attributable to data center operations. To estimate U.S. data center electricity consumption, we apply the ratio of global data center electricity consumption to the combined global electricity consumption for data center and office operations (approximately 96%). This yields an estimated U.S. data center electricity consumption of approximately 22 TWh. A global PUE value of 1.09 is then applied to derive U.S. data center IT energy, resulting in approximately 20 TWh. Finally, WUE is calculated as the ratio of U.S. data center water consumption to U.S. data center IT energy, yielding approximately 1.13 L/kWh. In addition, the operator reports location-level potable water use. Based on the aggregated U.S. data center locations, the U.S.-level potable water ratio is approximately 94.40%. 

**Hyperscale-2.** This operator reports Americas-level (North and South America combined) PUE and WUE values of 1.16 and 0.38 L/kWh for 2024, respectively, which we adopt as proxies for U.S. data center operations. Total water withdrawal and water consumption are reported at the North America level and are treated as U.S.-level totals in this study (5,789 ML and 3,136 ML, respectively). U.S. data center water withdrawal and water consumption are estimated by applying a data center water consumptive ratio of 0.75 and an office water consumptive ratio of 0.10. Combined with the water withdrawal and consumption numbers, this yields approximately 3,934 ML of U.S. data center water withdrawal and 2,951 ML of U.S. data center water consumption. Using the estimated U.S. data center water consumption and the reported WUE, U.S. data center IT energy is calculated as approximately 7.8 TWh. Total electricity consumption is then derived by applying the reported PUE, yielding approximately 9.0 TWh. Although a North America–level electricity consumption of approximately 16.6 TWh is reported, it is not disaggregated for data center operations. We therefore base our estimation on the disclosed water quantities and estimate electricity use accordingly. Note that by applying 3,136 ML (including office use but assumed here to be entirely for data centers) with a WUE of 0.38 L/kWh and a PUE of 1.16, we obtain an alternative estimate of 9.6 TWh for the North America–level total data center electricity consumption, which still differs from the reported total electricity consumption of 16.6 TWh in North America (including non-data center electricity consumption). This discrepancy is likely due to reporting or accounting inconsistencies by the operator. The operator reports global water consumption by source. Third-party water accounts for 5,775 ML out of a total water consumption of 5,807 ML worldwide. We use this proportion as a proxy for the U.S. data center municipal water ratio, yielding approximately 99.45%. 

36 



**Hyperscale-3.** This operator reports data center water withdrawal at the location/site level, which allows us to aggregate all facilities explicitly identified as located in the United States and obtain total U.S. data center water withdrawal. Based on the disclosed values, U.S. water withdrawal is approximately 2,365 ML out of a global data center water withdrawal of approximately 4,143 ML, corresponding to a share of about 0.57. We directly sum the water withdrawals of all disclosed data center locations, and the resulting total is slightly different from the reported total water withdrawal 4,145 ML due to rounding errors. The U.S. data center water consumption is then estimated by scaling the reported global data center water consumption (2974 ML) according to the observed U.S. withdrawal share, yielding approximately 1,698 ML. This procedure implicitly assumes a uniform data center water consumptive ratio of approximately 0.72 across all facilities. The operator also reports global third-party water withdrawal of 5,625 ML out of a total withdrawal of 5,637 ML. We treat this proportion as a proxy for the U.S. municipal water ratio, yielding approximately 99.79%. 

The operator reports a global WUE of 0.19 L/kWh and a global PUE of 1.08. Because approximately 43% of total data center water withdrawal/consumption occurs outside the United States, the reported global WUE may not accurately represent U.S.-level performance. We therefore do not directly adopt the global WUE for estimating U.S. operations. However, in the absence of U.S.-level PUE disclosure, we adopt the reported global PUE of 1.08 for U.S.-level estimation, since average PUE values are typically more consistent across U.S. and non-U.S. locations. 

The operator also discloses location/site-based data center electricity consumption. By aggregating all facilities explicitly identified as located in the United States, we obtain total U.S. data center electricity consumption of approximately 12.8 TWh. Dividing this value by the global PUE yields U.S. IT energy of approximately 11.8 TWh. Using the estimated U.S. water consumption, we derive a U.S.-level WUE of approximately 0.14 L/kWh, which is lower than the reported global WUE. 

**Hyperscale-4.** The operator reports location-based data center electricity consumption by region, from which we aggregate approximately 1.7 TWh for U.S. hyperscale data centers (excluding colocation data centers). The reported global data center electricity consumption is approximately 2.5 TWh (including colocation data centers), implying that self-managed U.S. operations account for about 68% of total data center electricity use. This share is subsequently used to scale global water-related quantities to the U.S. level. 

Specifically, global data center water withdrawal and discharge are reported as approximately 1,800 million gallons (MG) and 900 MG, respectively. We therefore compute water consumption as the difference between withdrawal and discharge, yielding approximately 900 MG. After separating data center and office water use using consumptive ratios of 0.75 for data centers and 0.10 for office operations, the resulting global data center water withdrawal is scaled using the 68% U.S. energy share, yielding approximately 2,842 ML for U.S. data centers. Applying the default data center consumptive ratio of 0.75 results in an estimated U.S. data center water consumption of approximately 2,132 ML. To derive IT energy, we assume a PUE of 1.09, consistent with the value adopted for Hyperscale-1 due to the lack of disclosure by this operator, yielding an estimated U.S. data center IT energy of approximately 1.6 TWh. Dividing the estimated water consumption by IT energy results in a WUE of approximately 1.36 L/kWh. 

The operator further reports total global water use of 1,756 MG (slightly different from 1,800 MG), consisting of 1,532 MG of freshwater, 197 MG of recycled water, and 27 MG of other alternative water sources. The report indicates that freshwater and recycled water are primarily supplied through municipal systems, with less than 5% obtained from other providers. Accordingly, we attribute 95% of the combined freshwater and recycled water volumes to municipal supply, yielding an estimated U.S. municipal water ratio of approximately 93.54%. 

**Hyperscale-5.** For this operator, U.S. data center total electricity consumption and IT energy are not directly reported. The operator reports a North America–level WUE of 0.13 L/kWh and a PUE of 1.14, which are treated as values of its U.S. operations in this study. 

Despite the lack of detailed electricity consumption data, Hyperscale-5 reportedly maintains one of the largest data center footprints in the United States and ranks among the top purchasers of renewable energy, surpassing many of its peers. Thus, for the purpose of setting its contribution to average WUE within the hyperscale data center category, we assume its U.S. IT energy to be 120% of the average combined IT energy of the three largest hyperscale operators (Hyperscale-1, Hyperscale-3, and Hyperscale-2, in descending order), yielding approximately 16.0 TWh. Based on the reported PUE of 1.14, the corresponding total elec- 

37 



tricity consumption is estimated to be approximately 18.2 TWh. A different electricity consumption value would only slightly affect the average WUE for the hyperscale data center type (e.g., by less than 5% if the total energy consumption of Hyperscale-5 were adjusted to match that of the current largest Hyperscale-1). 

Notably, the reported WUE for Hyperscale-5 is explicitly defined as water withdrawal per unit of IT energy. To maintain consistency, we convert the withdrawal-based WUE to a consumption-based WUE using a data center water consumptive ratio of 0.75, yielding an effective WUE of approximately 0.10 L/kWh. Applying this value to the estimated IT energy results in total water consumption of approximately 1,560 ML and total water withdrawal of approximately 2,081 ML. 

**Colocation-1.** The operator has significant non-U.S. presence and reports global-level water consumption and total electricity consumption. We assume a U.S. share of 50% and apply this allocation factor to estimate U.S. data center water consumption of approximately 2,793 ML and U.S. data center electricity consumption of approximately 5.7 TWh. As colocation data centers generally have higher PUE values than hyperscalers, a default PUE of 1.25 (assumed in this study) is then applied to derive U.S. data center IT energy, yielding approximately 4.5 TWh. WUE is computed as water consumption divided by IT energy, resulting in approximately 0.62 L/kWh. Finally, assuming a data center water consumptive ratio of 0.75, U.S. data center water withdrawal is estimated based on the calculated water consumption. 

In addition, the operator reports global municipal water consumption of 1,475,293 kGal out of a total global water consumption of 1,475,762 kGal. We use this proportion as a proxy for the U.S. municipal water ratio, yielding approximately 99.97%. Among the reported municipal water consumption, 849,319 kGal is identified as potable water, resulting in approximately 57.55%. However, the low potable water ratio is possibly due to the operator’s substantial non-U.S. presence, where stricter regulations on potable water use may apply; the U.S. potable water ratio is likely higher. 

**Colocation-2.** This operator reports global-level water withdrawal and water consumption, as well as energy consumption by region. We compute the ratio of reported energy consumption in the Americas to total global energy consumption and treat this ratio as the U.S. allocation factor in this study (approximately 40%). Using this allocation factor, U.S. data center water withdrawal and water consumption are estimated by scaling down the corresponding global totals, yielding approximately 2,153 ML and 1,654 ML, respectively. The resulting water consumptive ratio is approximately 0.77. The operator reports a WUE of 0.95 L/kWh and a PUE of 1.39. Based on these values, IT energy is computed as water consumption divided by WUE, yielding approximately 1.7 TWh. The total electricity consumption is then derived by applying the reported PUE to the estimated IT energy. 

In addition, the operator reports global municipal water withdrawal of 5,140 ML out of a total global withdrawal of 5,440 ML. We use this proportion as a proxy for the U.S. municipal water ratio, resulting in approximately 94.49%. The operator further reports that non-potable sources account for 37% of total water use at the global level. Accordingly, the complementary share (63%) is the potable water ratio. However, similar to Colocation-1, the low potable water ratio is likely due to the operator’s substantial non-U.S. presence, where stricter regulations on potable water use may apply; the U.S. potable water ratio is likely higher. 

**Colocation-3.** For this operator, the vast majority of data center operations are located in the United States. The operator reports water consumption of 2,119 ML and water withdrawal of 2,323 ML, from which a water consumptive ratio of approximately 0.91 is computed. The operator also reports a WUE of 0.82 L/kWh and a PUE of 1.40. Based on these values, IT energy is calculated as water consumption divided by WUE, yielding approximately 2.6 TWh. Total data center electricity consumption (approximately 3.6 TWh) is then derived by applying the reported PUE to the estimated IT energy. In addition, the operator reports that all water withdrawal is sourced from third-party suppliers. Accordingly, we treat the municipal water ratio as 100%. 

**Colocation-4.** For this operator, nearly all data center operations are located in the United States. The operator reports data center water consumption of approximately 1,021 ML and water withdrawal of approximately 1,312 ML, from which a water consumptive ratio of approximately 0.78 is calculated. The operator also reports a WUE of 0.29 L/kWh and a PUE of 1.46. Based on the reported WUE, U.S. data center IT energy is computed as water consumption divided by WUE, yielding approximately 3.5 TWh. Total data center energy is then derived by applying the reported PUE to the estimated IT energy. 

**Colocation-5.** For this operator, nearly all data center operations are located in the United States. The 

38 



operator reports total electricity consumption and specifies a PUE of 1.23, calculated as a 12-month trailing average across seasoned sectors, which we adopt in this study. A WUE value of 1.28 L/kWh is also disclosed. Using the reported electricity consumption (approximately 1.5 TWh) and the disclosed PUE, we derive the corresponding U.S. data center IT energy (approximately 1.2 TWh). Based on the reported WUE, total water consumption is then estimated at approximately 1,584 ML. Finally, applying the default consumptive ratio of 0.75, we estimate total U.S. data center water withdrawal at approximately 2,112 ML. 

**Colocation-6.** For this non-U.S. operator, we assume that 20% of its data center capacity is located in the United States. The operator reports global total energy consumption, which is treated as total global data center electricity consumption. Applying the 20% allocation factor yields U.S. data center electricity consumption of approximately 0.8 TWh. Using the reported PUE of 1.38, IT energy is derived as total electricity divided by PUE, resulting in approximately 0.6 TWh. The reported WUE of 0.72 L/kWh is then applied to estimate U.S. data center water consumption based on the derived IT energy, yielding approximately 431 ML. Finally, by applying the water consumptive ratio of 0.75, U.S. data center water withdrawal is estimated from the calculated water consumption. 

**Colocation-7.** For this operator with significant non-U.S. presence, we assume that 50% of its data center capacity is located in the United States. The operator reports global water withdrawal for evaporative cooling, which is treated as total data center water withdrawal in this study. Applying the 50% allocation factor yields U.S. data center water withdrawal of approximately 48 ML. U.S. data center water consumption is then estimated by applying the default water consumptive ratio of 0.75, resulting in approximately 36 ML. The operator also reports total purchased electricity at the global level, which is treated as total global data center electricity consumption, together with a reported PUE of 1.33. Applying the same 50% allocation factor yields U.S. data center electricity consumption of approximately 0.8 TWh. IT energy is then derived as total electricity divided by PUE, resulting in approximately 0.6 TWh. Based on the estimated U.S. data center water consumption and IT energy, WUE is computed as water consumption divided by IT energy, yielding approximately 0.06 L/kWh. 

**Data center type-specific WUE.** We assign the 2024 U.S.-level hyperscale WUE as the IT energy–weighted average across the five hyperscale operators (approximately 0.55), and the 2024 U.S.-level colocation WUE as the IT energy–weighted average across the seven colocation operators (approximately 0.65); the WUE for others is set to 0. Note that while IT energy is estimated for some companies and may differ from the actual, undisclosed values, such differences affect only the weights in our aggregation, and our methodology of aggregating multiple companies provides a reasonably robust reflection of the industry’s prevailing WUE. 

### **A.4.2 The Years of 2025 to 2030** 

We obtain WUE trajectories under three different scenarios. For our Baseline, Moderate, and Optimistic scenarios, WUE is differentiated by data center type. Starting from the 2024 values, WUE varies with compound annual rates of 0%, -5%, and -10% under the Baseline, Moderate, and Optimistic scenarios, respectively. The data center type-specific IT energy values in our calculations are reported in Table 7. Accordingly, water consumption is first computed separately for each data center type and then aggregated to obtain national totals. 

The Reference (LBNL and NS) scenarios focus on WUE at the aggregate national level. To enable a consistent and direct comparison, we therefore also compute a total U.S.-average WUE for our three scenarios by aggregating across the data center types. For each year, the total water consumption is calculated as the sum of type-specific IT energy multiplied by the corresponding WUE values, and the U.S.-average WUE is then obtained by dividing this aggregate water consumption by total IT energy. By construction, this total WUE is an IT energy–weighted average across data center types. Because IT energy differs under the low and high assumptions, total WUE is calculated separately for the two cases, yielding corresponding low and high total WUE values. 

The resulting WUE trajectories from 2024 to 2030 for all the scenarios are summarized in Table 8. The corresponding annual water consumption values under each scenario are reported in Table 9. For our Baseline, Moderate, and Optimistic scenarios, both total and type-specific (hyperscale and colocation) values are provided. Annual water consumption values are reported without decimal places. As the differences between the low and high total WUE values under the Baseline, Moderate, and Optimistic scenarios are small, all WUE values in Table 8 are reported to three decimal places to ensure clear numerical distinction. 

39 



**_Table 9:_** _Annual water consumption (million gallons) under Reference, Baseline, Moderate, and Optimistic scenarios, 2024–2030. Mid values are defined as the arithmetic average of the corresponding low and high estimates._ 

|**Y**|**Gth**|**Refe**|**rence**||**Baseline**|||**Moderate**|||**Optimisti**|**c**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**ear**|**row**|**LBNL**|**NS**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|
||Low|14,191|58,323|19,351|8,647|10,704|19,351|8,647|10,704|19,351|8,647|10,704|
|2024|**Mid**|**15,453**|**62,643**|**20,984**|**9,615**|**11,369**|**20,984**|**9,615**|**11,369**|**20,984**|**9,615**|**11,369**|
||High|16,716|66,964|22,618|10,583|12,035|22,618|10,583|12,035|22,618|10,583|12,035|
||Low|17,489|68,951|23,606|10,469|13,137|22,426|9,946|12,480|21,245|9,422|11,823|
|2025|**Mid**|**19,946**|**76,742**|**26,581**|**12,066**|**14,515**|**25,252**|**11,463**|**13,789**|**23,923**|**10,859**|**13,063**|
||High|22,404|84,534|29,556|13,663|15,893|28,078|12,979|15,099|26,600|12,296|14,304|
||Low|21,797|81,964|28,772|12,664|16,108|25,967|11,429|14,537|23,305|10,258|13,047|
|2026|**Mid**|**26,045**|**94,860**|**33,706**|**15,154**|**18,552**|**30,420**|**13,677**|**16,743**|**27,302**|**12,275**|**15,027**|
||High|30,294|107,756|38,640|17,644|20,996|34,873|15,924|18,949|31,299|14,292|17,007|
||Low|27,090|97,781|35,018|15,296|19,722|30,024|13,115|16,909|25,528|11,151|14,377|
|2027|**Mid**|**33,890**|**118,053**|**42,777**|**19,045**|**23,733**|**36,676**|**16,328**|**20,348**|**31,185**|**13,884**|**17,301**|
||High|40,690|138,326|50,537|22,793|27,744|43,329|19,542|23,787|36,841|16,616|20,225|
||Low|33,433|117,016|42,590|18,462|24,128|34,689|15,037|19,652|27,943|12,113|15,830|
|2028|**Mid**|**43,932**|**147,809**|**54,381**|**23,968**|**30,413**|**44,293**|**19,522**|**24,772**|**35,679**|**15,725**|**19,954**|
||High|54,431|178,601|66,172|29,474|36,698|53,897|24,006|29,891|43,415|19,338|24,078|
||Low|41,237|140,939|51,980|22,359|29,621|40,221|17,301|22,920|30,694|13,203|17,491|
|2029|**Mid**|**56,806**|**186,576**|**69,476**|**30,306**|**39,171**|**53,759**|**23,450**|**30,309**|**41,025**|**17,895**|**23,130**|
||High|72,374|232,213|86,972|38,252|48,720|67,297|29,599|37,699|51,356|22,588|28,769|
||Low|51,016|170,280|63,502|27,104|36,398|46,679|19,924|26,756|33,747|14,404|19,343|
|2030|**Mid**|**73,808**|**236,685**|**89,009**|**38,416**|**50,593**|**65,430**|**28,239**|**37,190**|**47,303**|**20,416**|**26,887**|
||High|96,601|303,090|114,516|49,728|64,788|84,180|36,555|47,625|60,858|26,427|34,431|



**_Table 10:_** _Annual water withdrawal (million gallons) under Reference, Baseline, Moderate, and Optimistic scenarios, 2024–2030. Mid values are defined as the arithmetic average of the corresponding low and high estimates._ 

|**Year**|**Growth**|**Refer**<br>|**ence**<br>||**Baseline**<br>|||**Moderate**<br>|||**Optimisti**<br>|**c**<br>|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||**LBNL**|**NS**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|**Total**|**Hyper**|**Colo**|
||Low|18,287|75,160|24,795|11,195|13,600|24,795|11,195|13,600|24,795|11,195|13,600|
|2024|**Mid**|**19,910**|**80,711**|**26,894**|**12,448**|**14,446**|**26,894**|**12,448**|**14,446**|**26,894**|**12,448**|**14,446**|
||High|21,534|86,262|28,993|13,702|15,292|28,993|13,702|15,292|28,993|13,702|15,292|
||Low|22,513|88,758|30,246|13,554|16,691|28,734|12,877|15,857|27,221|12,199|15,022|
|2025|**Mid**|**25,668**|**98,758**|**34,064**|**15,622**|**18,443**|**32,361**|**14,841**|**17,521**|**30,658**|**14,059**|**16,598**|
||High|28,824|108,757|37,883|17,689|20,194|35,989|16,804|19,184|34,094|15,920|18,175|
||Low|28,033|105,414|36,863|16,396|20,467|33,269|14,798|18,471|29,859|13,281|16,578|
|2026|**Mid**|**33,485**|**121,958**|**43,192**|**19,620**|**23,572**|**38,981**|**17,707**|**21,274**|**34,986**|**15,892**|**19,093**|
||High|38,937|138,502|49,522|22,844|26,677|44,693|20,617|24,076|40,112|18,504|21,609|
||Low|34,814|125,664|44,863|19,804|25,059|38,464|16,980|21,485|32,705|14,437|18,268|
|2027|**Mid**|**43,539**|**151,665**|**54,812**|**24,657**|**30,155**|**46,994**|**21,140**|**25,854**|**39,958**|**17,975**|**21,983**|
||High|52,263|177,666|64,761|29,510|35,251|55,525|25,301|30,224|47,211|21,513|25,698|
||Low|42,941|150,295|54,559|23,902|30,657|44,439|19,468|24,970|35,796|15,682|20,114|
|2028|**Mid**|**56,407**|**189,783**|**69,674**|**31,031**|**38,643**|**56,750**|**25,275**|**31,475**|**45,713**|**20,359**|**25,354**|
||High|69,873|229,270|84,788|38,159|46,629|69,060|31,081|37,979|55,629|25,036|30,593|
||Low|52,940|180,936|66,585|28,948|37,637|51,522|22,399|29,123|39,318|17,094|22,224|
|2029|**Mid**|**72,904**|**239,452**|**89,007**|**39,236**|**49,770**|**68,872**|**30,360**|**38,511**|**52,558**|**23,169**|**29,389**|
||High|92,868|297,968|111,429|49,525|61,904|86,221|38,321|47,900|65,798|29,244|36,554|
||Low|65,468|218,519|81,338|35,091|46,247|59,791|25,795|33,996|43,227|18,649|24,578|
|2030|**Mid**|**94,691**|**303,653**|**114,020**|**49,737**|**64,284**|**83,815**|**36,561**|**47,254**|**60,595**|**26,432**|**34,163**|
||High|123,915|388,787|146,702|64,382|82,320|107,840|47,327|60,513|77,964|34,215|43,748|



## **A.5 Consumptive Ratio** 

Type-specific consumptive ratios are applied for the three data center types in the calculation of ADD and MDD. Specifically, the U.S.-level consumptive ratio for hyperscale data centers is approximately 0.77, computed as the ratio of aggregated water consumption to aggregated water withdrawal across five hyperscale operators. The corresponding ratio for colocation data centers is approximately 0.79, calculated analogously across seven colocation operators. 

The consumptive ratio for “Others” is set to 0.75 by default [50]. This assumption does not affect the Baseline, Moderate, or Optimistic scenarios, because the WUE for “Others” is set to zero in these scenarios and therefore does not contribute to either water consumption or water withdrawal. In contrast, under the 

40 



Reference (LBNL and NS) scenarios, all three data center types contribute to water consumption because a common U.S.-wide average WUE is applied to each type. 

After obtaining U.S.-level IT energy and WUE for each data center type, the type-specific annual water consumption is calculated as the product of IT energy and WUE. Water withdrawal is then derived by dividing water consumption by the corresponding type-specific consumptive ratio. The resulting annual withdrawal values are reported in Table 10. Since ADD is defined based on water withdrawal, it is computed as follows (without distinguishing leap and non-leap years): 



$$
ADDt = 1 365 � i\in{}{}D EIT i,t \cdot{}{} WUEi,t γi , (10)
$$

where _t_ denotes the year, _i_ denotes the data center type ( _D_ = _{_ hyperscale _,_ colocation _,_ others _}_ ), _Ei,t_<sup>ITis the an-</sup> nual IT energy consumption of type _i_ in year _t_ , WUE _i,t_ is the corresponding WUE, and _γi_ is the consumptive ratio associated with data center type _i_ . Under the Reference (LBNL and NS) scenarios, a U.S.-wide average WUE _i,t_ is applied across hyperscale, colocation, and others. In contrast, under the Baseline, Moderate, and Optimistic scenarios, WUE _i,t_ differs by data center type and is set to zero for “others”. 

## **A.6 Peaking Factor** 

To estimate MDD, a daily peaking factor is required to convert ADD into the water capacity demand measured in terms of the maximum daily demand. In Section 3.1.2, we observe that cooling towers exhibit a relatively lower peaking factor (measured or estimated at approximately 2.0–2.5, which will be higher for planning purposes to account for the worst case [32]), whereas dry cooling with evaporative assistance shows a substantially higher peaking factor due to the concentration of water use on a limited number of days each year. For one leading technology company, the daily peaking factor is estimated at approximately 6.5 (based on the measured monthly peaking factor of 4.3) in Iowa, planned at 8.0+ in Leesburg, Virginia, and planned at 30+ in Wisconsin. Likewise, a leading technology company’s state-of-the-art data center under construction to host AI and core products has secured an allocation of up to 8 MGD of water capacity, resulting in a peaking factor exceeding 6.3 based on the company’s existing U.S.-wide average water withdrawal intensity (see Appendix D for details). 

A recent report analyzing water utility data indicates that the measured daily peaking factor of multiple data centers within the Prince William Water service area reached 10 in 2024 [50]. When hundreds of data centers with diverse cooling system designs and operational configurations are aggregated, the overall peaking factor is moderated by multiplexing effects. Nevertheless, the weighted estimate for the actual peaking factor (not for infrastructure planning purposes) in Northern Virginia remains in the range of 3.5 to 3.7 under different scenarios (Table 6-5 of [50]). 

For infrastructure planning purposes, the peaking factor will be even higher than the measured value, as an additional safety margin is applied to account for worst-case scenarios and sharp rises in IT loads and cooling demand [32]. This is analogous to additional backup generator capacity (commonly “2N,” or doubling the capacity relative to actual power needs [50]) and power capacity reservation (e.g., a commonly assumed average power utilization factor of 50%, i.e., reserving twice the actual power need [2]). Thus, even assuming a 25% additional reservation, the effective peaking factor for hundreds of data centers in Northern Virginia [50] would be approximately 4.5. 

Therefore, to provide a conservative and reasonable estimate grounded in empirical values observed from available sources, we adopt an average industry-wide peaking factor of 4.5, without further increase through 2030. The water capacity demand measured in terms of MDD for planning purposes is subsequently calculated under the peaking factor of 4.5 using Eq. (8). 

In the future, as evaporative-assisted cooling becomes more common, it can improve the annual water efficiency while simultaneously increasing the peaking factor, since such systems “use zero water for a majority of the year” [19] and typically operate only 5 to 15% of the year [16]. Thus, the actual peaking factor is likely higher than 4.5, and this variation can be partially accommodated by the different growth rates used in our analysis. As peak water use or the peaking factor is rarely reported publicly, we recommend that data centers include site-level peaking factors in their annual reports to support improved water capacity management. 

41 



**_Table 11:_** _Estimated cost for water projects. Projects 12–17 are specifically or partly associated with data centers._ 

|**Project**|**Description**|**Unit Cost ($M/MGD)**|
|---|---|---|
|1|Water treatment expansion only [133]|7.6|
|2|Water treatment expansion only [133]|6.6|
|3|Advanced water treatment, wastewater treatment, concentrate management, and solids handling [133]|86|
|4|Advanced water treatment, wastewater treatment, concentrate management, and solids handling [133]|41|
|5|Advanced water treatment, wastewater treatment, concentrate management, and solids handling [133]|23–31|
|6|Advanced water treatment, wastewater treatment, concentrate management, and solids handling [133]|23–32|
|7|New water reuse/recycling facility [134]|46|
|8|Water treatment expansion only [102]|9|
|9|Reservoir and new water treatment plant [102]|12|
|10|Wastewater treatment construction and expansion only [102]|20|
|11|Wastewater treatment facility improvement and expansion [135]|13|
|12|Wastewater recycling facility only [119]|6|
|13|Water (0.64 MGD) and sewer service upgrades [3,84]|40|
|14|New water source and other necessary facilities (8 MGD) [27]|37|
|15|Water and wastewater services (0.11 MGD water and 0.08 MGD wastewater) [85]|50|
|16|Supplying 25 MGD water to an innovation district [40]|22|
|17|Wastewater system improvements and expansion [18]|23|



## **A.7 Estimated Costs for Water Infrastructure Projects** 

Water infrastructure project costs vary widely depending on factors such as project complexity, geographic location, labor costs, and inflation. For example, a recent public water infrastructure upgrade to support a leading technology company’s data centers cost up to $400 million for an undisclosed water capacity [16], whereas a smaller upgrade providing approximately 0.11 MGD of drinking water capacity and 0.08 MGD of wastewater capacity to a data center reportedly cost $5.4 million [85]. 

To derive a reasonable cost range, we review multiple recent or planned water infrastructure projects documented in the public domain, including official cost estimates from local governments and water utilities as well as publicly reported project costs. Among these, Projects 12-17 are specifically or partly associated with data center water needs. The results are shown in Table 11. All reported costs are in 2025 dollars or earlier without inflation adjustment; consequently, actual project costs may increase in future years due to inflation. 

Notably, some reported cost estimates exclude transmission and distribution components, such as pipelines, which can represent a substantial share of total costs and often exceed the cost of water or wastewater treatment facilities themselves [28]. Based on this review and to account for additional applicable infrastructure components including pipes, pumps, and storage, we adopt a conservative unit cost range of $10–40 million per MGD of capacity for valuing data center water infrastructure requirements in our study. 

Importantly, our estimate of cost ranges is intended to represent the scale of infrastructure investment for each MGD capacity and should _not_ be interpreted as the actual financial obligation of a specific data center operator, which may vary depending on contractual arrangements, accounting rules, local rate structures, among others. 

# **B Data Center Water Use in Loudoun County, Virginia** 

We use Loudoun County, Virginia, as a case study, given its high concentration of data centers. As a large public/community water system according to the EPA’s classification [28], Loudoun Water provides service to over 81,000 households as well as many data centers [26]. In 2024, the measured peak daily water _withdrawal_ by data centers in the Loudoun Water service area reached 10.9 MGD, resulting in a peak of 2,716 gallons per megawatt of IT load per day (or 0.428 L/kWh) consolidated over all the served data centers [50]. The 10.9 MGD of water withdrawal includes both potable water and non-potable water, corresponding to a total IT load of approximately 4,013 MW across the Loudoun Water service area. The relatively low peak water withdrawal reflects the current practice that many Northern Virginia data centers predominantly rely on dry-cooling systems, despite the fact that dry coolers are more power-intensive than evaporative cooling, particularly during the summer months. Note that the same report [50] also notes a lower peak of 2,435 gallons per megawatt on Page 6-14, which would imply a higher corresponding IT load and, consequently, a higher peak water use in the following estimate. To remain conservative, we therefore adopt the larger value of 2,716 gallons per megawatt per day. 

42 



We consider a hypothetical scenario in which all IT loads are cooled using the more power-efficient evaporative cooling method. To improve the result robustness, we consider the following methods to estimate the corresponding total peak daily water demand. 

**The estimate in [50].** The report [50] assumes in Table 6-5 that, under a scenario in which evaporative systems account for 90% of the share, peak water _withdrawal_ reaches 5,200 gallons per megawatt of IT load per day. This corresponds to a pWUE of approximately 0.615 L/kWh, which is compared to (or even substantially lower) than many of the average company-wide WUE values reported by leading colocation providers and technology companies (see Table 2). This discrepancy may reflect the particular operational configuration and local climatic conditions, as well as the possibility that some facilities classified as using evaporative cooling in fact serve only a limited portion of their IT load with evaporative systems [50]. Consequently, the estimate of 5,200 gallons per megawatt of IT load per day (under the assumption of 90% evaporative cooling) is specific to the selected set of data centers in the study [50] and may not be representative of the current average WUE or pWUE of U.S. data centers. Nonetheless, under the assumed setting in [50], if all data centers were to adopt evaporative cooling, peak water withdrawal would easily exceed 5,500 gallons per megawatt of IT load per day, corresponding to a total peak withdrawal of approximately 22.1 MGD. In practice, public water utilities also adopt more conservative planning assumptions by incorporating higher peaking factors than actually metered peaks and reserving additional capacity to manage worst-case scenarios, such as extreme heatwaves. This can effectively drive up the total water capacity need to 30 MGD or more. 

**Industry disclosures.** To complement the estimates derived based on [50], we consider industry disclosures. A large colocation provider reports an annual average WUE of 1.55 L/kWh for facilities using evaporative cooling in 2024 [127]. In comparison, a large technology company operating multiple data centers in the United States with a mix of water-intensive evaporative cooling and non-evaporative systems reports an estimated annual WUE of 1.13 L/kWh [17]. Note that excluding the four air-cooled U.S. data center sites (whose energy consumption is not disclosed) from the 23 sites in total [17] would increase the estimated annual WUE for evaporatively cooled sites to above 1.13 L/kWh, bringing it closer to the 1.55 L/kWh reported for evaporative cooling by [127]. These company-wide WUE values provide empirical U.S./global benchmarks for water efficiency in large-scale data centers using evaporative cooling. To account for the relatively cooler climate of Northern Virginia, we apply the reported company-wide average WUE values directly and do not incorporate additional peaking factors or redundancy-related capacity allocations. This approach yields estimates that are plausibly conservative. Specifically, the implied peak daily water consumption in 2024 would range from 28.8 to 39.4 million gallons. Using a consumptive use ratio of 0.75 as assumed in [50], the corresponding peak daily water withdrawal would range from 38.3 to 52.6 million gallons. 

**Water allocation example.** The total water capacity allocated to a data center is often not publicly disclosed. Nonetheless, public county records [84, 136] indicate that, as of April 2024, the Town of Leesburg, Virginia, allocates 1.23 MGD of water capacity to a large data center campus operated by a leading technology company. This campus is also approved for approximately 314 MW of total diesel generator capacity by the Virginia Department of Air Quality as of March 2025 [137]. 

Assuming the same “2N” redundancy configuration and an 80% IT load utilization factor as in [50], a 314 MW diesel generator capacity corresponds to an effective IT load of approximately 126 MW. When compared against the 1.23 MGD water capacity allocation, this implies roughly 9,762 gallons of water withdrawal per MW of IT load per day. While future load growth without additional water capacity allocation may be feasible and could reduce the effective pWUE, the current pWUE estimate more closely reflects the planned design value, which accounts for worst case operating conditions as well as anticipated future expansion. Given that tens of large data centers with an estimated aggregate IT load of 4,013 MW are served by Loudoun Water, the total required water capacity could approach 40 MGD if all facilities were to rely on evaporative cooling. 

In summary, if all data centers within the Loudoun Water service area were to rely on evaporative cooling during the summer months, the aggregate peak water capacity requirement would likely fall in the range of 20–50 MGD, even under conservative assumptions. In contrast, Loudoun Water reports a total drinking water design capacity of 70 MGD, with a measured peak demand of 41 MGD and an available capacity of less than 30 MGD as of 2022 [138]. The capacity available for new allocation is likely even smaller after accounting for operational reserves and committed but not yet fully utilized allocations. These figures sug- 

43 



gest that the remaining infrastructure capacity, after meeting the needs of residential, commercial, and other existing users, would likely be insufficient to accommodate the aggregate peak water demand or allocation requests of data center customers as of the end of 2024, should all data centers rely on evaporative cooling. In addition, even if the total water treatment plant capacity may be sufficient, other components of the system, including pump stations and distribution pipelines, may constitute binding constraints in meeting the peak water demand associated with evaporative cooling across all the data centers. 

Accordingly, in the absence of further expansion of water treatment and distribution infrastructure, accommodating the high peak demand associated with a full transition to evaporative cooling for all the data centers would likely be difficult and necessitate alternative strategies. These may include large-scale adoption of reclaimed water (which does not reduce the underlying consumptive water use of evaporative cooling) as well as deployment of dry cooling systems (which shift peak resource burdens from regional water systems to the electricity grid, particularly during summer periods). 

# **C Water Capacity Need for Cooling 100 MW IT Loads** 

Assuming sufficient water capacity is available, we now estimate the water capacity need for cooling a 100 MW IT load using evaporative methods. We rely on industrial disclosures and public data sources to develop plausible estimates. 

A large colocation provider reports an annual average WUE of 1.55 L/kWh for facilities using evaporative cooling in 2024 [127]. In comparison, a large technology company operating multiple data centers in the United States with a mix of water-intensive evaporative cooling and non-evaporative systems reports an estimated annual WUE of 1.13 L/kWh [17]. Note that excluding the four air-cooled U.S. data center sites (whose energy consumption is not disclosed) from the 23 sites in total [17] would increase the estimated annual WUE for evaporatively cooled sites to above 1.13 L/kWh, bringing it closer to the 1.55 L/kWh reported for evaporative cooling by [127]. These company-wide WUE values provide empirical U.S./global benchmarks for water efficiency in large-scale data centers using evaporative cooling (mostly cooling towers). 

The measured daily peaking factor (or estimated based on monthly values) for data center cooling towers is approximately 2.2 (Section 3.1.2). For planning purposes, we adopt a conservative average WUE of 1.2 L/kWh and assume a peaking factor of 2.5 to reflect operational safety margins. This implies a peak daily pWUE of 3.0 L/kWh. Assuming a consumptive ratio of 0.75, consistent with [50], an evaporative cooling system supporting a 100 MW IT load would require a peak water withdrawal capacity of approximately 2.5 MGD. In hotter climates, this requirement could be substantially higher, even exceeding 5.0 MGD. For example, measured monthly peak WUE values exceeding 9 L/kWh have been reported for data centers in Arizona [64]. 

Next, we examine dry cooling with evaporative assistance. While this approach significantly reduces annual WUE compared to conventional cooling towers, peak water demand can still be substantial. As a real example, consider a leading technology company’s data center campus in Iowa. In 2022, the reported annual WUE was 0.19 L/kWh, [139]. The peak daily WUE (pWUE) can be estimated by multiplying the monthly peaking factor of approximately 3 in 2022 (as shown in Figure 2a) by a minimum adjustment factor of 1.5 based on regulatory guidance [92], yielding a pWUE of 0.855 L/kWh. This is substantially higher than the annual average. Applying the default consumptive ratio of 0.75 [50] results in a water withdrawal intensity of 1.14 L/kWh. Thus, for this Iowa data center campus, cooling a 100 MW IT load requires approximately 0.72 MGD, which also needs to be further adjusted upward to approximately 1.0 MGD for planning to maintain operational safety margins in practice. 

The same technology company also operates a large data center campus in Leesburg, Virginia, where it invested $25 million to upgrade local water treatment and sewage infrastructure, ensuring that the associated costs are not passed on to local ratepayers [3]. Public county records [84, 136] indicate that, as of April 2024, the Town of Leesburg, Virginia, allocates 1.23 MGD of water capacity to the data center campus. This campus is also approved for approximately 314 MW of total diesel generator capacity by the Virginia Department of Air Quality as of March 2025 [137]. Following the study [50] to apply the same “2N” redundancy configuration and an 80% IT load utilization factor, a 314 MW diesel generator capacity corresponds to an effective IT load of approximately 126 MW. When compared against the 1.23 MGD water capacity allocation, this implies roughly 9,762 gallons of water withdrawal per MW of IT load per day, corresponding 

44 



to a peak water withdrawal intensity of 1.540 L/kWh. Thus, cooling a 100 MW IT load can need a water capacity of approximately 0.98 MGD. With a consumptive ratio of 0.75 as considered in [50], the peak water withdrawal intensity corresponds to a pWUE of 1.155 L/kWh. By comparison, the reported annual WUE for the company’s data center fleet across Northern Virginia is 0.14 L/kWh as of 2023 (which may be lower due to efficiency improvements in 2025) [56]. This comparison suggests that the peaking factor for the Leesburg data center campus is approximately 8 or higher. While future load growth without additional water capacity allocation may be feasible and could reduce the effective pWUE, the current pWUE estimate more closely reflects the planned design value, which accounts for worst case operating conditions as well as anticipated future expansion. In colder climates, the peaking factor can even be substantially higher (e.g., reaching 30 [87]), because evaporative cooling is utilized only on a limited number of days each year. 

We now consider company-wide average WUE values measured and reported by leading data centers that primarily employ evaporative assistance. These values typically range from 0.1 to 0.2 L/kWh (Table 2). Assuming 0.15 L/kWh and accounting for a peaking factor of 6.5 (Figure 2c), the peak daily WUE (pWUE) can be roughly 1.0 L/kWh, which corresponds to a peak water withdrawal intensity of approximately 1.3 L/kWh under the default consumptive ratio of 0.75 [50]. For planning purposes, a higher value is appropriate, suggesting that a 1 MGD water capacity allocation is a reasonable estimate for cooling a 100 MW IT load. 

While the benefits of water evaporative cooling remain (e.g., a recent $400 million investment to upgrade public water infrastructure to support data centers hosting AI and cloud services [16]), the adoption of liquid-cooled servers can tolerate higher server-level temperature setpoints, allowing air-cooled heat rejection to be used for a greater portion of the year. This can further reduce total, and potentially peak, water usage. To account for this effect, as well as regional climate variations, we apply a 50% reduction to the water capacity need of 1.0 MGD for evaporatively cooling a 100 MW IT load, based on the current planning data from the leading technology company. This yields an adjusted water capacity need of roughly 0.5 MGD. 

In summary, cooling a 100 MW IT load with evaporative cooling typically requires approximately 0.5– 2.5 MGD of water capacity, depending on the cooling system design and operational configuration, while excluding extreme climate conditions. 

# **D Water Allocation in an Innovation District in Boone County, Indiana** 

The Limitless Exploration/Advanced Pace (LEAP) Lebanon Innovation District in Boone County, Indiana, is a state-led economic development initiative designed to attract high-tech, advanced manufacturing, and research businesses. To support anticipated water demand, the local water utility is upgrading infrastructure to supply up to an additional 25 MGD of potable water capacity and expand wastewater treatment capacity by an additional 15 MGD. These upgrades are estimated to cost approximately $1 billion, including $560 million (financed through Indiana’s Drinking Water State Revolving Fund) for new water supply, [40], $225 million for upgrades to existing infrastructure of the host town (Lebanon, Indiana), and $350 million for wastewater facility expansion [18]. The full 25 MGD is expected to be delivered by 2031. 

We review disclosed water allocation agreements and pre-agreements across several projects spanning different categories to further examine how data centers use water differently. Unless otherwise specified, all water allocation values correspond to the projected full buildout capacity. 

- Project 1 (Hospital) [140]: Peak water supply capacity of 34,200 gallons per day (GPD) and average demand of 11,400 GPD; peak wastewater capacity of 27,000 GPD and average flow of 7,000 GPD. The peaking factors for water and wastewater are 3.00 and 3.86, respectively. 

- Project 2 (Housing) [141]: Water and wastewater allocations are not directly disclosed, except for 387 equivalent dwelling units (EDUs) and a peaking factor of 1.49. In the innovation district, each EDU corresponds to an average water demand of 500 GPD [142], resulting in an estimated average demand of 193,500 GPD and peak capacity of approximately 288,315 GGD for water supply. 

- Project 3 (Industry) [142]: Peak water supply capacity of 135,000 GPD and average demand of 100,000 GPD; peak wastewater capacity of 40,000 GPD and average flow of 33,500 GPD. The peaking factors for water and wastewater are 1.35 and 1.19, respectively. 

45 



- Project 4 (Pharmaceutical) [143]: Peak water supply capacity of 1,675,529 GPD and average demand of 1,581,933 GPD; peak wastewater capacity of 1,350,000 GPD and average flow of 864,000 GPD. The peaking factors for water and wastewater are 1.06 and 1.56, respectively. 

- Project 5 (Pharmaceutical) [144]: Peak water supply capacity of 749,970 GPD and average demand of 513,482 GPD; peak wastewater capacity of 290,970 GPD and average flow of 205,232 GPD. The peaking factors for water and wastewater are 1.46 and 1.42, respectively. 

- Project 6 (Data Center) [18]: Peak water supply capacity of 8,000,000 GPD and average demand of 4,000,000 GPD; peak wastewater capacity of 4,000,000 GPD and average flow of 2,000,000 GPD. This is a final executed agreement. The peaking factors for both water and wastewater are 2.00 based on the disclosed average and peak water use. However, the operator’s existing U.S. data centers (15 different locations) report an average water withdrawal intensity of 0.20 L/kWh as of 2024 [123]. Considering that the planned “state-of-the-art data center” located in the Innovation District has a capacity of 1 GW and will use a closed-loop liquid cooling system that recirculates water and “will use zero water for a majority of the year” [19], it is highly likely that the new facility will achieve an average water withdrawal intensity below 0.20 L/kWh. This would correspond to an average water demand of no more than 1.27 MGD, even under the assumption that all the full 1 GW capacity corresponds to IT load. Given the peak water capacity of 8 MGD, this implies a planned daily peaking factor of at least 6.31, consistent with other large technology operators using similar cooling systems (Appendix A.6). Therefore, the specified average water demand in the agreement likely serves primarily as an administrative placeholder with limited relevance to the data center’s actual operational needs, ensuring that sufficient withdrawal volumes remain authorized if the facility’s design or cooling configuration changes in the future. Nonetheless, although peak water capacity allocations are also often requested above expected operating levels to hedge against worst-case scenarios such as extreme heatwaves or unanticipated IT load growth, securing such peak capacity is costly and requires substantial financial commitment, and thus more closely reflects both the anticipated operational demand and necessary reliability margins. 

Importantly, based on the operator’s existing U.S. data center water efficiency, it is highly likely that the planned peaking factor is substantially higher than those of other typical water users, indicating more variability between average and peak demand. 

Moreover, the data center’s peak water allocation alone accounts for 32% of the new 25 MGD water capacity and 27% of the expanded 15 MGD wastewater capacity supplied to the innovation district, making it likely the single largest user of the added capacity. Considering $560 million for the new water capacity of 25 MGD [40] and $350 million for wastewater capacity expansion of 15 MGD [18], the total estimated investment for the data center’s water infrastructure is approximately $270 million. We exclude the anticipated $225 million funding required to upgrade existing water infrastructure [18]. 

Our cost estimate is intended to represent the scale of infrastructure investment to serve the data center’s water capacity need and should _not_ be interpreted as the actual financial obligation of the data center operator, which may vary depending on contractual arrangements, accounting rules, local rate structures, among others. 

# **E Water-Power Tradeoff for Data Center Cooling** 

Data center cooling and heat rejection strategies present a fundamental tradeoff between water use and (peak) electricity consumption, particularly during summer ambient conditions. At the facility level, dry aircooled heat rejection systems (a.k.a., dry coolers) can avoid water use entirely but generally use more energy compared to evaporative cooling methods, which use water to reduce the temperature of the air entering the condenser or facility loop. According to industry disclosures, during peak summer conditions, evaporativebased/-assisted coolers can use 10–35% less electricity than equivalent air-cooled systems [47,63]. Notably, multiple leading technology companies have recently signed agreements with local water utilities to secure substantial water capacity (e.g., up to 8 MGD at full buildout) for cooling their newest AI and cloud data center campuses across various states [16,18–21], with some water capacity expected to be delivered in 2031 due to required infrastructure upgrades [18]. 

46 



The basic thermodynamic distinction that underlies facility-level cooling choices is that evaporative cooling leverages the ambient wet-bulb temperature, whereas dry cooling leverages the ambient dry-bulb temperature. The dry-bulb temperature is the routine measure of air temperature and does not depend on humidity. By contrast, the wet-bulb temperature reflects the lowest temperature that air can reach by evaporative cooling, and is always less than or equal to the dry-bulb temperature except at saturation (100% relative humidity) where they coincide. This is because evaporation removes sensible heat from the air until equilibrium is reached. 

For a given target temperature setpoint required by server-level cooling, the cooling energy scales with the difference between that setpoint and the ambient reference temperature. When employing purely dry cooling, the system can reject heat only down toward the dry-bulb temperature; therefore, dry cooling is sufficient to maintain a facility loop (e.g., 29<sup>_◦_</sup> C for air-cooled servers or around 40<sup>_◦_</sup> C for liquid-cooled servers) without mechanical or evaporative assistance only when the dry-bulb temperature is a few degrees lower. In contrast, evaporative cooling can achieve effective heat rejection even when the dry-bulb temperature exceeds the loop setpoint, provided the ambient wet-bulb temperature is a few degrees lower than the setpoint. This evaporative supplement enables cooling at conditions where the dry-bulb temperature alone would be too high without other mechanically-/energy-intensive cooling methods. 

Consequently, adjusting the facility-loop temperature setpoint can impact the water-power tradeoff: higher setpoints can decrease the energy savings achievable through evaporation, while lower setpoints increase power reduction benefits of evaporative cooling but require more water. Effectively, the optimal balance between water and electricity depends on site-specific constraints, including local water infrastructure capacity, electricity cost, and peak power capacity. 

The introduction of liquid cooling for high-density AI servers at the server level further complicates the design. Liquid-cooled servers allow the facility loop to operate at a higher supply temperature setpoint while still maintaining safe component temperatures. This higher loop setpoint enables a greater fraction of the year to be handled by dry coolers without additional water usage. However, even in these scenarios, evaporative cooling continues to provide energy benefits: by lowering the effective temperature seen by the dry cooler, evaporative assistance reduces pump and fan power (which typically changes in a cubic manner with the fan speed), particularly during peak summer conditions in which the PUE for data centers with dry coolers can be significantly higher than the annualized average PUE [145, 146]. Consequently, evaporative cooling remains a valuable tool for mitigating peak electricity demand, even when higher loop temperatures are permissible with liquid cooling [15,147]. In other words, dry coolers supplemented with evaporative cooling effectively operate as if the ambient air were cooler than the actual summer conditions. This benefit is especially significant in hotter climates. For example, a leading technology company’ state-ofthe-art data center under construction to host AI and core products employs a “closed-loop, liquid-cooled system that recirculates the same water” [19], but still needs up to 8 MGD for facility-level (evaporative) cooling assistance during the hottest days of the year [18]. 

While the high power density of AI server racks often necessitates liquid cooling (e.g., direct-to-chip and immersion cooling) because of its superior heat removal capability [58, 60], they still have approximately 20% or more loads cooled by air that have lower temperature setpoint than liquid-cooled IT loads [15,148]. This means that evaporative assistance can have significant benefits of peak power reduction for the aircooled portion of IT loads. For example, to balance the peak power demand and the host community’s limited available water capacity, a large data center uses dry coolers for its liquid-cooled AI servers while evaporative assitance for other servers [147]. Indeed, a leading technology company acknowledges that “water is the most efficient means of cooling in many places” [17]. 

It is important to note that batteries in data centers are typically deployed to support IT loads and provide uninterruptible power during outages before backup generators are activated, but using large-scale energy storage to directly reduce facility-level cooling power remains both technically challenging and economically costly relative to the achievable peak power reduction. As a result, data centers generally do not rely on battery-based solutions for cooling power reduction. Instead, many facilities employ evaporative cooling assistance, particularly during hot periods, to reduce peak energy consumption for heat rejection. Nonetheless, the limited availability of public water capacity is increasingly emerging as a binding—and often under-recognized—constraint [24], which can play a more decisive role in shaping the future water–power tradeoff for data center cooling. 

47 



# **F Additional Recommendations** 

We provide additional recommendations to complement those in Section 5 for comprehensively addressing the growing water demand of data centers. 

**Additional Recommendation 1: Waste Heat Recovery.** Waste heat from data centers represents a largely untapped resource that can be leveraged to improve overall energy and water efficiency, particularly as liquid-cooled AI servers increasingly operate at elevated temperatures exceeding 40<sup>_◦_</sup> C. By capturing and repurposing this excess thermal energy, data centers can supply heating to buildings, district heating networks, or industrial processes, thereby reducing dependence on additional energy- and water-intensive cooling systems [149]. Although significant engineering challenges exist—such as the relatively low “quality” of the heat and the spatial separation from heat demand centers—waste heat recovery remains a promising strategy. It can not only deliver tangible benefits to surrounding communities but also help lower the water demand of data center operations. 

**Additional Recommendation 2: Water-Aware Computing.** Water storage tanks can serve as a buffer to smooth hourly water demand peaks. However, it is challenging to use storage to fully offset multi-day or even month-long peak water demand during prolonged heatwaves, as this requires costly large-scale tanks, sanitation, maintenance, and other operational considerations. Data center workloads often exhibit substantial scheduling flexibility [150, 151], which, when combined with multi-region site deployment, may effectively reduce peak water demand. Therefore, “water-aware” computing in combination with water tanks presents a promising approach to lowering peak water demand and strengthening public water system resilience. 

**Additional Recommendation 3: Water-AI Nexus.** Technologies such as advanced AI have the potential to substantially enhance the operation, efficiency, and resilience of public water systems [152]. For instance, AI-driven leak detection and predictive maintenance can identify losses in real time, reducing water waste and lowering operational costs [153]. These interventions are especially important given that water losses in distribution networks constitute a significant inefficiency, analogous to transmission and distribution losses in U.S. power grids, which exceeded the total energy consumption of all U.S. data centers in 2023 [2, 154]. By mitigating such losses, AI can effectively increase available water supply without new infrastructure investment. We therefore recommend development and deployment of AI and related technologies to assist public water systems in managing increasingly complex infrastructure networks, improving resilience, and simultaneously accommodating the growing water demands of data centers. 

# **G Economic Potential of Evaporative Cooling for Peak Power Reduction** 

We now perform an order-of-magnitude comparison to highlight the potential benefits of evaporative cooling as an economically competitive approach for shaving peak power demand. 

Assuming an average power capacity utilization of 50% [2] and a peak PUE reduction of 0.15 as a _reference_ case, using evaporative cooling for a 100 MW IT load can effectively reduce peak power demand by 30 MW, while requiring an estimated water capacity of 0.5–2.5 MGD (Appendix C). Based on the valuation of $10–40 million per MGD capacity (Appendix A.7), we assume the mid valuation at $25 million per MGD, yielding a total cost range of $12.5–62.5 million of water capacity to cool a 100 MW IT load. By comparison, installing power generators to supply 30 MW, excluding long-distance transmission costs, would cost approximately $37.2 million in the U.S. South and $70.9 million in the U.S. Northeast, according to the most recent 2023 capacity-weighted construction cost estimates from the U.S. Energy Information Administration [155]. 

Although it does not account for other factors such as environmental reviews and construction lead times, this simplified orders-of-magnitude comparison highlights the potential of evaporative cooling as an economically competitive approach to shaving peak power demand, emphasizing the need for coordinated water-power planning to support growing data center loads. It also aligns with a leading technology company’s statement that “water is the most efficient means of cooling in many places” [17], and that some data centers need up to 8 MGD [18, 21] of water capacity (for peak power shaving) with capital costs for the necessary water infrastructure upgrades reaching hundreds of millions of dollars [16,18,27]. Nonetheless, 

48 



![Figure](assets/figure_0006_page_0049.svg)**_Figure 6:_** _Statistics of the total water use of U.S. golf courses [156]_ 

overall water availability at regional watersheds remains important, and actual peak power reductions can vary depending on local climate conditions and cooling system design. 

# **H Water Footprint of Animal Products** 

The study [112] offers a comprehensive assessment of the water footprint associated with farm animals and their products. To make our study self-contained and for clarity, we summarize the key findings in [112] related to animal products. 

In hydrological accounting, the total water footprint is divided into three distinct categories: green water, which is the volume of rainwater consumed by crops that is stored in the soil; blue water, which refers to surface and groundwater sourced from aquifers and rivers for agricultural irrigation, livestock consumption, and industrial processes, including but not limited to municipal public water supply; and grey water, which represents the volume of freshwater required to dilute pollutants, such as fertilizer runoff, to meet water quality standards. 

The total water footprint of global animal production was roughly 2,400 Gm<sup>3</sup> of water annually during the period 1996-2005. The vast majority of the water footprint is green water (rainfall stored in soil), with smaller portions coming from blue (surface and groundwater) and grey (polluted) water sources. Beef production alone accounts for about one-third of the total water footprint, while dairy production contributes 19%. The overwhelming portion of the water, i.e., approximately 98%, is the indirect water used to grow feed for the animals, whereas direct water use make up only a small fraction of the total. More specifically, drinking water for the animals, service water and feed mixing water account only for 1.1%, 0.8% and 0.03%, respectively. 

Even assuming that all the direct water use originates from public or municipal water systems, it accounts for less than 2% of the total water footprint. Therefore, comparing the total water footprint of animal products with data centers’ Scope 1 municipal water use is uninformative and obscures the actual challenges faced by public water systems. 

# **I Water Use of U.S. Golf Courses** 

We summarize the results of a national water use survey of U.S. golf courses reported in [156]. Although golf courses represent a substantial water user in aggregate when measured by total volumetric withdrawals, the reliance on municipal drinking water is relatively limited, representing approximately 7-9% of their total applied water nationwide between 2005 and 2024. The limited use of municipal water is partly attributable to state and local regulations that restrict or discourage the use of potable water for landscape irrigation, particularly in water-stressed regions [157]. In addition, total projected irrigation demand for U.S. golf courses declined by approximately 31% nationally between 2005 and 2024, reflecting changes in course management practices and efficiency improvement. Region-wise results and further breakdowns are provided in 

49 



the original paper [156]. The summary of nation-level water use statistics is presented in Fig. 6. The numbers refer to “applied water,” representing all the water sprayed, dripped, or otherwise applied to turf and landscaped areas. 

Additionally, in the context of golf course irrigation, the daily peaking factor is typically less than 3.0 [158]. Golf course irrigation is generally concentrated during nighttime or the early morning, taking place before golfers begin their rounds, despite fluctuations in seasonal water requirements. Even when supplied by public water systems, a typical fully-irrigated golf course uses up to 0.5 MGD [24]. 

# **J Capacity Planning and Management in Public Water Systems** 

In the United States, public water system capacity planning is governed by a combination of federal capacity development requirements under the Safe Drinking Water Act (SDWA) [73] and state-level implementation through permitting, financing, and utility regulation. While federal law does not prescribe explicit numerical utilization thresholds at which capacity expansion must occur, states typically operationalize capacity planning through a combination of engineering design criteria, financing rules, and legally enforceable limits on service commitments. Here, we use California as a representative and well-documented example [43], which separates planning triggers based on actual utilization from legal limits based on allocated capacity. 

**Planning triggers based on utilization.** Capacity expansion planning is typically initiated well before a system reaches its physical design limits. This expectation is reflected in Drinking Water State Revolving Fund (DWSRF) policies administered by the California State Water Resources Control Board [81]. Specifically, under California’s DWSRF rules, capacity expansions are generally limited to approximately 10% above existing MDD. This is intended to ensure engineering reliability, maintain service to existing customers, and reduce financial and operational risk. 

As a result, when observed or projected demand approaches roughly 80–90% of existing design capacity, utilities are expected to begin formal planning activities, including demand forecasting, engineering evaluation, environmental review, and financing arrangements. The utilization range of 80–90% serve as planning references rather than enforceable legal thresholds, and are intended to ensure that infrastructure expansion occurs proactively rather than in response to emergency conditions. **Allocated capacity and legal limits on new connections.** Distinct from the actual operational utilization is the concept of _allocated capacity_ , which accounts for existing demand together with approved but possibly not completed service commitments. In California, once allocated capacity equals a system’s demonstrated or permitted supply capacity, additional service connections may be legally restricted even if actual water usage still remains below physical limits. 

Specifically, for regulated water utilities, Section 2708 of the California Public Utilities Code empowers regulators to prohibit new or additional service connections when further commitments would impair service reliability or violate capacity constraints [159]. Importantly, this determination is based on the system’s ability to serve its committed load, not necessarily on its actual average usage. Thus, regulatory decisions require utilities that cannot demonstrate adequate source, treatment, storage, or distribution capacity to comply with or request a service connection moratorium until additional capacity is constructed or otherwise secured [160]: “ _A system’s facilities shall have the capacity to meet the system’s MDD, PHD plus any required fire flow in the system as a whole and in each individual pressure zone. If, at any time, the system does not have this capacity, the system shall request a service connection moratorium until such time as it can demonstrate the source capacity has been increased to meet system requirements_ .” As a result, a public water system may be considered legally “at capacity” once its allocated commitments fully exhaust approved capacity, even if actual demand remains, for example, near or lower than 70% of the physical design capacity. 

**Implications for new service connections.** Combined together, these mechanisms illustrate a general framework for public water system infrastructure management. First, infrastructure planning and investment are triggered as systems approach high utilization levels but below full design capacity. Second, legally binding restrictions on new service connections may apply once allocated capacity is fully committed, regardless of the actual water use. This structure aligns long-term infrastructure planning with public health protection and service reliability while preventing over-extension of public water system resources. 

Consequently, a public water system’s ability to accommodate new service connection requests is governed by its available capacity, decided by (the more restrictive of) the remaining physical capacity and the 

50 



uncommitted capacity (i.e., total physical capacity minus already allocated capacity). The system’s full design capacity is therefore not the primary determinant in decisions regarding new connections. Even a large public water system serving multiple existing large users may deny a new service connection request if its available capacity is limited, even when the requested capacity is substantially smaller than that of some existing users. 

This practice is also reflected in recent data center projects. For example, a major technology company stated that its new data centers in Louisiana will draw water exclusively from the host community’s “verified surplus water” capacity, aiming to ensure “no strain on local water supplies” [16]. 

51 

