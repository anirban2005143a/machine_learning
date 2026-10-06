1 

# Estimating the Increase in Emissions caused by AI-augmented Search 

Wim Vanderbauwhede School of Computing Science 

University of Glasgow 

Glasgow, UK 

Email: wim.vanderbauwhede@glasgow.ac.uk 

Abstract—AI-generated answers to conventional search queries dramatically increase the energy consumption. By our estimates, energy demand increase by 60-70×. This is a based on an updated estimate of energy consumption for conventional search and recent work on the energy demand of queries to the BLOOM model, a 176B parameter model, and OpenAI’s GPT-3, which is of similar complexity. 

## I. INTRODUCTION 

The new trend in search engines, to provide an AI-generated answer to the search query, has a considerable impact on the energy consumption and therefore CO2 emissions per query. To illustrate the impact of AI augmented search queries more clearly, I compare the energy consumption and emission of a query to Google’s BLOOM model with that of a conventional Google search-style query. If all search queries are replaced by AI-augmented queries, what does that mean for energy consumption and emissions? 

## II. GOOGLE SEARCH ENERGY AND EMISSIONS 

In 2009, The Guardian published an article about the carbon cost of Google search [1]. Google had posted a rebuttal [2] to the claim that every search emits 7 g of CO2 on their blog. What they claimed was that, in 2009, the energy cost was 0.0003 kWh per search, or 1 kJ. That corresponded to 0.2 g CO2, and I think that was indeed a closer estimate. 

This number is still often cited but it is entirely outdated. In the meanwhile, computing efficiency has rapidly increased [3]: Power Usage Effectiveness (PUE, metric for overhead of the data centre infrastructure) dropped by 25% from 2010 to 2018; server energy intensity dropped by a factor of four; the average number of servers per workload dropped by a factor of five, and average storage drive energy use per TB dropped by almost a factor of ten. Google has released some figures about their data centre efficiency REF that are in line with these broad trends. It is interesting to see that PUE has not improved much in the last decade. 

Therefore, with the current AI hype, I wanted to revise that figure from 2009. Three things have changed: the carbon intensity of electricity generation has dropped [4], server energy efficiency has increased a lot, and PUE of data centres has improved [5]. Combining all that, my new estimate for energy consumption and the carbon footprint of a Google search is 0.00004 kWh and 0.02 g CO2 (using carbon intensity for the US). According to Masanet[3], hardware efficiency 

increases with 4.17× from 2010 to 2018. This is a power law, so extrapolating this to 12 years gives 6.70×<sup>1</sup> . The calculation of the updated emissions per search query is shown in Alg. 1. 

So the energy consumption per conventional search query has dropped by 7× in 14 years. There is quite some uncertainty on this estimate, but it is conservative, so it will not be less than that, but could be up to 10×. Microsoft has not published similar figures but there is no reason to assume that their trend would be different; in fact, their use of FPGAs should in principle lead to a lower energy consumption per query. In that same period, carbon emissions per search have dropped about 10× because of the decrease in carbon intensity of electricity. 

## III. BLOOM ENERGY CONSUMPTION PER QUERY 

In a recent paper [6], Luccioni et al. analysed the energy consumption per query for the 176B-parameter BLOOM model, which is of the same complexity as GPT-3. The model was used to provide AI-generated summaries of search queries. They measured power consumption over a period of 18 days, in which the model received an average of 558 requests per hour, for 230,768 requests in total. This resulted 914 kWh of electricity. With the above figure for electricity carbon intensity, the emissions per query to the summarising model are shown in Alg. 2. 

In other words, the query to the BLOOM model to summarise the search result costs 75× more energy than the conventional search query itself. 

## IV. CHATGPT ENERGY CONSUMPTION PER QUERY 

There are several estimates of the energy consumption per query for ChatGPT. I have summarised the ones that I used in the following table. There are many more, these are the top ranked ones in a conventional search. 

Reference [11] by de Vries uses the estimates from [12] for energy consumption but does not present a per-query value so I used the query estimate from [12]. Overall, the estimates lie between 24× and 236× (from [7], which is a collation of estimates from Reddit and therefore very broad) or 28× to 160× (all other sources). 

> 1I use 12 years instead of 14 from 2009 as typically servers have a life of 4 years. Therefore the most likely estimate is that the current servers are two years old, i.e. they have the efficiency from 2021. 



2 

Algorithm 1 Calculation of emissions per search query 

PUE: 1.16 in 2010; 1.1 in 2023; efficiency increase of hardware in 12 years: 6.70x US overall carbon intensity: 367 gCO2/kWh 0.0003*(1.1/1.16)*(1/6.70) = 0.0000424 kWh per search 

0.0000424*367 = 0.02 g CO2 per search 

Algorithm 2 Calculation of emissions per query to the LLM 914/230768 = 0.004 kWh/request 

US overall carbon intensity: 367 gCO2/kWh 367*914/230768=1.5 g CO2 per request 

|Ref|Estimate (kWh/query)|Increase vs conventional search|
|---|---|---|
|[7]|0.001-0.01|42--236|
|[8]|0.0017 - 0.0026|40–61|
|[9]|0.0068|160|
|[10]|0.0012|28|
|[11]|0.0029<br>Ta|68<br>ble I|



ESTIMATES OF ENERGY CONSUMPTION PER QUERY FOR CHATGPT 

![Figure](assets/figure_0001_page_0002.svg)MEAN INCREASE ESTIMATE FOR CHATGPT QUERIES COMPARED TO CONVENTIAL SEARCH QUERIES 

original training; they are almost certainly much lower as the changes in the corpus are small, so it is tuning. 

## B. Data centre efficiency 

As far as I can tell, PUE is not taken into account in the above estimates. For a typical hyperscale data centre, it is around 1.1, so it will not changes the estimate appreciably. 

## C. Embodied carbon 

Neither the Google search estimate nor the ChatGPT query estimates include embodied carbon. The embodied carbon can be anywhere between 20% and 50% of the emissions from use, depending on many factors. My best guess is that the embodied emission are proportionate to the energy consumption, so this would not affect the factor much. 

I consider any estimate lower than 0.002 kWh/query overly optimistic and any estimate higher than 0.005 kWh/query overly pessimistic. However, rather than judging, I calculated the mean over all these estimates. I used four types of means. Typically, an ordinary average gives more weight to large numbers; a harmonic mean gives more weight to small numbers. Given the nature of the data, I think the geometric mean is the best estimate: 

As you can see, there is not that much difference between the geometric mean and the median. So we can conclude that ChatGPT consumes between fifty and ninety times more energy per query than a conventional ("Google") search, with sixty times being the most likely estimate. These estimates agree remarkably well with those of the BLOOM model (seventy-five times). 

## V. OTHER FACTORS CONTRIBUTING TO EMISSIONS 

## A. Training 

Contrary to popular belief, it is the use of ChatGPT, not its training, that dominates emissions. I wrote about this in [13]. In the initial phase of adoption, with low numbers of users, emissions from training are not negligible, but in the scenario where conventional search is replaced by ChatGPTstyle queries, which is now the case for Bing, Google, Yandex, Baidu and many of the less popular search engines, emissions from training are only a small fraction. How much is hard to say as we don’t know how frequently the model gets retrained and what the emissions are from retraining as opposed to the 

## VI. CONCLUSION 

Taken all this into account, it is possible that the emissions from generating AI summaries for search are more than a hundred times that of the conventional search query. As I don’t have enough data to back this up, I will keep the conservative estimates from above (50× – 90×; 60× most likely for ChatGPT; 75× for BLOOM). 

Now, if we want sustainable ICT, then the sector as a whole needs to reduce its emissions to a quarter from the current ones by 2040. The combined increase in energy use and growth in adoption of AI-augmented search and other generative AI applications is therefore deeply problematic. 

## REFERENCES 

- [1] L. Hickman, “The carbon cost of Googling,” 2009. [Online]. Available: https://www.theguardian.com/environment/ethicallivingblog/2009/jan/12/carbon-emissions- 

- [2] Google, “Powering a Google search ,” 2009. [Online]. Available: https://googleblog.blogspot.com/2009/01/powering-google-search.html 

- [3] E. Masanet, A. Shehabi, N. Lei, S. Smith, and J. Koomey, “Recalibrating global data center energy-use estimates,” Science, vol. 367, no. 6481, pp. 984–986, 2020. [Online]. Available: https://www.science.org/doi/abs/10.1126/science.aba3758 

- [4] Ember, “Data Page: Carbon intensity of electricity generation,” 2024. [Online]. Available: https://ourworldindata.org/grapher/carbon-intensity-electricity 

- [5] Google, “Efficiency,” 2023. [Online]. Available: https://www.google.co.uk/about/datacenters/efficiency/ 

- [6] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, “Estimating the carbon footprint of bloom, a 176b parameter language model,” 2022. 

- [7] N. Sreedhar, “AI and its carbon footprint: How much water does ChatGPT consume?” 2023. [Online]. Available: https://lifestyle.livemint.com/news/big-story/ai-carbon-footprint-openai-chatgpt-water-goo 



3 

|[8]|K.<br>Ludvigsen,<br>“ChatGPT’s<br>energy<br>use<br><br><br><br>|
|---|---|
||per<br>query,”<br>2023.<br>[Online].<br>Available:|
||https://towardsdatascience.com/chatgpts-energy-use-per-query-9383b8654487<br><br><br><br><br>|
|[9]|Zodhya,<br>“How<br>much<br>energy<br>does<br>Chat-<br><br>|
||GPT<br>consume?”<br>2023.<br>[Online].<br>Available:<br>|
||https://medium.com/@zodhyatech/how-much-energy-does-chatgpt-consume-4cba1a7aef85|
|[10]|C.<br>Pointon,<br>“The<br>carbon<br>footprint|
||of<br>ChatGPT,”<br>2023.<br>[Online].<br>Available:|
||https://medium.com/@chrispointon/the-carbon-footprint-of-chatgpt-e1bc14e4cc2a<br>i|
|[11]|A. de Vries, “The growing energy footprint of artificial intelligence,”<br>|
||Joule, vol. 7, no. 10, pp. 2191–2194, 2023. [Online]. Available:<br>https://www.cell.com/joule/fulltext/S2542-4351(23)00365-3|
|[12]|D. Patel and A. Ahmad, “The Inference Cost Of Search Disruption<br>– Large Language Model Cost Analysis,” 2023. [Online]. Available:<br>https://www.semianalysis.com/p/the-inference-cost-of-search-disruption|
|[13]|W.<br>Vanderbauwhede,<br>“The<br>climate<br>cost<br>of<br><br><br><br><br><br>|
||the<br>AI<br>revolution,”<br>2023.<br>[Online].<br>Available:|
||https://wimvanderbauwhede.codeberg.page/articles/climate-cost-of-ai-revolution/|



