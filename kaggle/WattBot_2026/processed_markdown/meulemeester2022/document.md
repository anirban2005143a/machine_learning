How sustainable is “common” data science in terms of power consumption? 

Bjorge Meulemeester<sup>a</sup> , prof. D. Martens<sup>a</sup> 

_aPrinsstraat 13, 2000 Antwerpen, BEL_ 

# **Abstract** 

Continuous developments in data science have brought forth an exponential increase in complexity of machine learning models. Additionally, data scientists have become ubiquitous in the private market, academic environments and even as a hobby. All of these trends are on a steady rise, and are associated with an increase in power consumption and associated carbon footprint. The increasing carbon footprint of large-scale advanced data science has already received attention, but the latter trend has not. This work aims to estimate the contribution of the increasingly popular “common” data science to the global carbon footprint. To this end, the power consumption of several typical tasks in the aforementioned common data science tasks will be measured and compared to: large-scale “advanced” data science, common computer-related tasks, and everyday non-computer related tasks. This is done by converting the measurements to the equivalent unit of “km driven by car”. Our main findings are: “common” data science consumes 2 _._ 57 more power than regular computer usage, but less than some common everyday power-consuming tasks such as lighting or heating; large-scale data science consumes substantially more power than common data science. 

_Keywords:_ Sustainability, carbon emission, AI, data science, energy consumption, carbon footprint, common data science 

# **1. Introduction** 

Data science and Artificial Intelligence (AI) have become a fundamental aspect of our digital world; be it for marketing strategies, uncovering shopping patterns of clients, computer-mediated language comprehension, detecting cardiac arrhythmias or simply predicting the weather. Deep neural networks have experienced a steep increase in complexity, and correspondingly a steep increase in the hardware required to train these models (Anthony et al., 2020). As the hardware requirements have grown exponentially, so has the power consumption attributed to these hardware requirements. Large models are commonly trained in big datacenters, powered by various types of energy (Cook et al., 2017). It is estimated that these datacenters currently account for about 1% of the worldwide electricity use (Masanet et al., 2020). Their projected energy consumption and carbon emission for the upcoming years is subject of debate, and estimates range between the status quo 

_Email addresses:_ 

bjorge.meulemeester@uantwerpen.be (Bjorge Meulemeester), david.martens@uantwerpen.be (prof. D. Martens) 

(Masanet et al., 2020) up to 8% of global carbon emission by 2030 (Cao et al., 2022). Major takeaways from this discussion is that datacenters are gradually shifting towards green energy (Belkhir & Elmeligi, 2018), are getting increasingly efficient (Masanet et al., 2020), but are simultaneously being used more and more for hyperscale data science (IEA, 2021) and, despite the shift towards green energy in datacenter’s infrastructure being widely adopted (Bashroush & Lawrence, 2020), measures targeting other areas, such as inefficiencies in servers, circular economy, and heat reuse, are still lagging (Rteil et al., 2022). 

These values represents the steep increase in data center demand and their usages, but does not take into account the increase in “common” data science jobs that do not require these large data centers. The term common data science will be used in this work to denote any data science that can be performed by one (or a team of) data scientist(s) on their home computer/laptop. 

Data science entails more than huge complex neural networks and big datacenters; common data scientists are omnipresent in the private market in various fields. Data science related jobs often end up in the top 5 fastest growing jobs in terms of demand: according to LinkedIn, _Machine learning engineer_ is the 4<sup>_th_</sup> 

_Preprint submitted to Elsevier_ 

_July 27, 2022_ 



fastest rising job in the U.S. (LinkedIn, 2022a), and the 2<sup>_nd_</sup> fastest rising job in The Netherlands, Italy and the U.K. (LinkedIn, 2022b). Other data science related jobs such as _Data scientist_ and _Data engineer_ tend to end up high on these lists as well. Does this increase in common “everyday” data science also imply an increase in power consumption and carbon emission? Masanet et al. (2020) estimate that in 2020, powering digital devices (excluding smartphones) accounted for 20% of all ICT-related greenhouse gas emissions (GHGe), and communication networks account for 25%; the sum of which is comparable to the estimate of data center GHGe of 45%. The contribution to the global GHGe of household ICT devices, such as those used by common data scientists, should not be underestimated. 

This work aims to compare the previously mentioned advanced data science, or “big datacenter data science” to the latter common data science. The main question is if the latter common data science has a substantial contribution to the global GHGe. To this end, the carbon emissions of common data science will be estimated and compared to common computer tasks, common non-computer related everyday tasks and the aforementioned “big datacenter” advanced data science. This is done by means of performing various tasks on a laptop while measuring their power consumption, comparing these values to one another, and to values found in literature. 

# **2. Methods** 

In order to measure the carbon emission of different kinds of computer usage, three steps need to be taken: 

1. Measuring the power consumption and/or energy consumption of a single task. 

2. Converting the energy usage to carbon emission. 

3. Converting the carbon emission to an interpretable quantity, e.g. km driven by car. 

The “everyday tasks, common data science tasks and advanced data science tasks that are to be compared will be explained in further detail in Section 2.6. 

# _2.1. Assumptions_ 

Inevitably, assumptions need to be made in order to compare the measurements in a comprehensive manner. The measurements in this work will only be as truthful as the truth value of these assumptions. These will also be revisited in Section 4.5. 

1. It suffices to measure the power consumption of the CPU to gain insight in the power consumption of performing various computer tasks. 

2. The power demand of the hardware used in this work is representative for other hardware in common data science jobs, or can be extrapolated to other hardware types by considering the Thermal Design Power (TDP) of the CPU. 

3. The authors’ work efficiency and methods are representative for the work of other data scientists. 

4. Measuring the power consumption of the hardware every 10 seconds suffices to give an accurate representation of the real-time power consumption of the hardware. 

5. A car emits as much _gCO_<sup>2</sup> _eq_ as the EU average from 2019. 

6. Energy production emits as much _CO_<sup>2</sup> _eq_ as the EU average from 2020. 

# _2.2. Measuring the power consumption_ 

Measuring the power consumption and/or energy consumption of a task performed on the computer will be done by using Ubuntu’s built-in powerstat command to sample the intel RAPL interface. This can be done by running the shell command 



$$
sudo powerstat -DRgf -d=<s> ∆t N > output.txt (1)
$$

where -D enables the -R option, -R denotes sampling should be done on the RAPL interface, -f enables showing the average CPU frequency, -d denotes a delay before starting the measurement (in seconds), ∆ _t_ denotes the time interval between samples (in seconds), N denotes the amount of samples and -g enables measuring the GPU power consumption as well. GPU power consumption and GPU programming will not be explored in this work, but was still measured for completeness. 

Where possible, CarbonTracker (Anthony et al., 2020) will also be used to measure energy consumption. CarbonTracker is designed to track and predict the energy consumption of training AI’s, but can be used for any piece of code. It must be noted that CarbonTracker was designed for measuring the power consumption of training an AI during one or more epochs, and not shorter code snippets prevalent in common data science. Due to this, its accuracy may deviate, especially if the runtime of the code is short. For this reason, interpreting the data will be based on the RAPL measurements, and the CarbonTracker measurements will only be shown for completeness, and as a benchmark to check if it produces similar results. Deviating results will also be discussed in Section 4. 

2 



# _2.3. Converting_ 

In order to link the energy consumption to carbon emission, the EU-28 average greenhouse gas emission associated with electricity generation during 2020 (EEA, 2021b) will be used, yielding an average emission per energy unit of: 



$$
CO2eq/energy = 230.7 g/kWh (2)
$$

where _CO_ 2 _eq_ denotes the amounts of gram _CO_ 2 that would yield a greenhouse effect of equal magnitude as the considered greenhouse gases, i.e. the seven greenhouse gases considered by the Kyoto Protocol (Protocol, 1997): carbon dioxide ( _CO_ 2), methane ( _NH_ 4), nitrous oxide ( _N_ 2 _O_ ), hydrofluorocarbons ( _HFCs_ ), perfluorocarbons ( _PFCs_ ), sulphur hexafluoride ( _SF_ 6) and nitrogen trifluoride ( _NF_ 3). 

In order to increase the interpretability of the _gCO_ 2 _eq_ quantity, they will be converted to the quantity “km driven by car, as suggested by Anthony et al. (2020). To do so, we will use the average GHGe of every registered car in the EU up until 2019 (EEA, 2021a), yielding a distance per equivalent carbon emission of: 



$$
8.17661488144 m/gCO2eq (3)
$$

_The RAPL interface._ measures by directly sampling the power consumption of the CPU every other time interval. This measurement is code-independent. Whether or not your code is running, this method will measure the power consumption. This method gives a good overview of the entire power consumption of some task, including the non-code related parts, e.g. looking up documentation and articles, debugging, running code that finishes with an error, etc. This measurement gives the most accurate measurement of a certain task, but will include variability depending on the efficiency, experience and knowledge of the data scientist performing the task. Note that only the power consumption of the CPU is considered. This may deviate from the true power consumption, especially when running code that relies heavily on RAM. This drawback will be discussed in Section 4.5, when revisiting the assumptions made earlier (see Section 2.1). 

_CarbonTracker._ measures the energy consumption of a single piece of code. This measurement gives a good overview of how much energy was consumed by running the code, and nothing but the code. If a script finishes on an error, no measurement will be written out. This measurement has a higher repeatability, but lacks information on the overhead of the tasks: the non-code related parts and running bugged code that finishes with an error. 

# _2.4. Hardware & software in this work_ 

All measurements were taken on a Dell Inspiron 15 7570 laptop with: 

- 7.5GB of RAM 

- 64-bit Intel® Core™i7-8550U CPU @ 1.80GHz × 8 

- NVIDIA Corporation GM108M [GeForce 940MX] / NVIDIA GeForce 940MX/PCIe/SSE2 graphical processing unit 

- running the latest stable version of Ubuntu (20.04.3 LTS), with GNOME version 3.36.8 

_RAPL vs CarbonTracker._ Note that the samples for the Carbontracker measurements is not the same kind of samples as those fore the RAPL measurements. Carbontracker reports aggregate values at the end of every successful exit of a piece of code, independent of how long it took, while RAPL measurements produce a real-time measurement every 10 seconds, independent of what the computer is doing. The variance of the Carbontracker measurements is thus a variance between the total power consumptions of different successfully run code snippets. The variance of the RAPL measurements, on the other hand, is the variance on the real-time power consumption. This difference in sample types should be kept in mind when interpreting the results. 

# _2.6. Notes on the measurements_ 

# _2.5. A note on the measurement tools_ 

As mentioned before, both sampling the RAPL interface direcly, as well as CarbonTracker (Anthony et al., 2020) will be used to measure the power/energy consumption. The resulting values should be interpreted differently. 

The considered measurements on common data science and common computer usage inevitably have a low repeatability due to the specific hardware used, the specific data science project and methods considered, as well as the workflow and work efficiency of the authors. For completeness, the considered measurements are described in as much detail as possible below. 

3 



Other reported values for everyday tasks and advanced data science are averages, educated guesses, or can only be reported under certain assumptions. These tasks will also be described in detail below. 

# _2.6.1. Measurements of regular computer usage_ 

_Baseline (idle)._ This “task is done by leaving a computer running with the screen on, and no open programs. The value of this task will vary depending on the efficiency of your hardware, background tasks and screen brightness. 

_Watching a movie._ This task was completed by streaming _Avengers: Age of Ultron_ via Disney+ on Firefox 96.0 for Ubuntu. No other tabs or programs were running. The carbon emission of the datacenter providing the movie is not taken into account due to lack of available data. 

_Normal usage._ Normal usage entails browsing the internet, reading a pdf and opening/closing various programs: Inkscape, Firefox, text editor and file browser. Tasks that involve video rendering (such as gaming, watching Youtube or viewing local video files) were purposely avoided, as these are comparable to the task _Watching a movie_ . 

_Working in Excel._ This task entailed making plots and performing various column transformations on a 4 _,_ 383 _×_ 20 dataset containing only continuous numerical values. These column transformations were aimed to reflect common data science operations such as: deleting columns, scaling columns, and power- and log-transformations. Plots were limited to histograms. These Excel operations were performed in LibreOffice Calc version 1:6.4.7-0ubuntu0.20.04.2. 

_Data science project._ This task entailed doing a common data science project, namely a credit scoring classification problem: identifying defaulters on a loan. This project is split into two main parts: 

WOE-encoding), scaling to normal distributions with QuantileTransformer(), filling missing values with means or modes and over- and undersampling with ADASYN and RandomUnderSampler(). 

Gridsearch & fit entailed performing two hyperparameter gridsearches for the machine learning models Random Forest, KNN and Logistic Regression on the dataset as described in the previous paragraph. Gridsearches were performed by starting with a coarse Gridsearch and refining the grid intervals once with a second Gridsearch. The considered hyperparameter grids are shown in Table 1. A 5-fold cross-validation scheme was used. 

# _2.6.2. Measurements of everyday tasks_ 

_Burning a lightbulb for an hour._ Assuming a 10 _W_ light bulb. 

_Streaming Season 1 of Friends._ The carbon emission of this task was calculated using the same measurement for power consumption as the task _Watching a movie_ and extrapolating this to the duration of the first season of Friends (2 _h_ 28<sup>_′_</sup> ). 

_Leaving the office lights on over the weekend._ This is calculated by assuming the office is lit by 8 10 _W_ TL lights, buring from friday 5pm until monday 9am. 

_Heating an office for a day._ This is calculated by assuming an office with following properties: 

   - 50 _m_<sup>2</sup> area 

   - One outside wall 

   - Solid and insulated cavity walls 

   - A ceiling height of 3 _m_ 

   - A minimum roof insulation of 75 _mm_ 

1. Exploration & preprocessing 

2. Gridsearch & fit 

Exploration & preprocessing entailed constructing plots of features and processing features according to their meaning, type and distribution. This is done by means of column transformations on a 20 _,_ 000 _×_ 35 dataset with mixed continuous and categorical features. All data exploration and processing was done in Python3.8 using matplotlib, pandas and scikit-learn. Column transformations included: encoding (one-hot encoding and 

- Electrically heated at 21<sup>_◦_</sup> _C_ (70<sup>_◦_</sup> _F_ ) 

Data was taken from electricpoint (n.d.). 

_Commercial airflight for 1 passenger (BRU −→ JFK NY)._ This carbon emission was calculated assuming a flight from Brussels (BRU) to New York (JFK NY), where all seats are economy seats and occupied. Data taken from ICAO (2016). 

4 



|Model|Gridsearch #1|Gridsearch #2|
|---|---|---|
||n<br>~~e~~stimators: [100, 500, 1000, 2000],|n<br>~~e~~stimators: [2000, 3000],|
|Random Forest|max<br>depth: [2, 10, 50, 100],<br>min<br>~~s~~amples<br>~~s~~plit: [2, 5, 10, 50]|max<br>~~d~~epth: [30, 50, 70],<br>min<br>~~s~~amples<br>~~s~~plit: [2]|
|KNN|n<br>~~n~~eighbors: [20, 100, 500, 1000],<br>p: [1, 2]|n<br>~~n~~eighbors: [10, 20, 50],<br>p: [1]|
||penalty: [l2, l1, elasticnet],|penalty: [l2],|
|Logistic Regression|C: [0.1, 1, 10, 100, 1000],<br>max<br>iter: [100, 1000, 2000]|C: [80, 100, 200, 300, 400],<br>max<br>~~i~~ter: [1500, 2000, 3000]|



Table 1: Hyperparameter grids for the two gridsearches performed during the task _Gridsearch & fit_ . 

# _2.6.3. Measurements of advanced data science_ 

_Training a CNN on images._ This measurement was taken from Anthony et al. (2020), where more details can be found. This advanced data science task was performed by training the CNN U-NET for 100 epochs on the LIDC medical image dataset. This required 1 _._ 25 _±_ 0 _._ 25 _kWh_ , as is visible in Figure 1 of Anthony et al. (2020). Note that: 

- This value was measured using different hardware 

- The dataset was already processed; processing does not contribute to the reported value 

- Only one training session is reported, which does not reflect a realistic case of trial and error, with multiple training sessions. 

- While computationally challenging, it is feasible to replicate this task on a home computer. This measurements lives in the grey zone between common and advanced data science. 

_Training GPT-3 in the EU._ This measurement is an estimate taken from Anthony et al. (2020), assuming Microsoft’s average datacenter PUE (Power Usage Effectiveness) of 2015, re-calculated with updated values for the GHGe associated with energy production (EU28 average from 2020) and GHGe of cars (EU28 average from 2021). 

# **3. Results** 

![Figure](assets/figure_0001_page_0005.svg)Figure 1: Power consumption distributions of working on common computer tasks and data science, measured by CarbonTracker (dark blue) and sampling the RAPL interface (light teal). RAPL samples are taken every 10 seconds, while CarbonTracker samples are taken every successful run of a piece of code. 

Figure 1 shows the power consumption distribution per task. 

Figure 2 shows the energy consumption of working on a certain task. For tasks with an unspecified end time, a duration of 8 hours is used to reflect a day worth of work. 

5 



![Figure](assets/figure_0002_page_0006.svg)Figure 2: Energy consumption and carbon emission (in km driven by car equivalent) of working on common computer tasks and data science related activities. Only RAPL measurements are shown. 

These are compared to large-scale data science and other everyday tasks in terms of carbon emission in Figure 3. 

The Thermal Design Power (TDP) of the CPU used in this work is compared to other CPUs in Figure 4. 

# **4. Discussion** 

# _4.1. Power consumption_ 

We can see on Figure 1 how CarbonTracker yields similar results on power consumption as directly sampling the RAPL interface for _Gridsearch & fit_ , but there is a notable difference between the two measurements for the task _Exploration & preprocessing_ . This can be explained by the fact that the latter task includes a lot of overhead, such as looking up documentation, scaling columns one by one, programming, debugging and running faulty code. All of the previously mentioned overhead is not captured by CarbonTracker measurements; only code that runs without error is measured by CarbonTracker. CarbonTracker can be interpreted as the hypothetical power consumption of a hyper-efficient data scientist that does not make any mistakes, knows the sklearn library by heart and writes their code instantaneously. _Gridsearch & fit_ , on the other hand, is defined by few to no programming, but simply leaving a script running. This explains the similarity of the mea- 

![Figure](assets/figure_0003_page_0006.svg)Figure 3: Carbon emission of common non-computer related tasks and data science (common and large-scale) on a logarithmic scale, expressed in units of _km driven by car_ . 

# Thermal Design Power of all CPUs recorded by PassMark 

![Figure](assets/figure_0004_page_0006.svg)Figure 4: Thermal Design Power of all CPUs as recorded by Passmark (2022). The TDP of the CPU considered in this work is denoted by a vertical blue line, the median TDP by a black one. 

6 



surements of CarbonTracker and directly sampling the RAPL interface. 

# _4.2. Energy consumption_ 

Figure 2 reports the energy consumption of the same tasks as shown in Figure 1, sorted from low to high. If no duration is reported in the label, a duration of an 8 hour working day is considered. 

It is notable how streaming a 1 _._ 5 hour movie requires less energy than sitting through a 1 hour Teams call, despite both entailing similar tasks: real-time video rendering. This is possibly due to the fact that online meetings cannot make use of the same video compression methods as streaming a movie, since the video data of the former is produced in real-time. Processing less compressed data would then require more computational power and put the CPU under heavier load, requiring more energy. It must be noted that the carbon emission of the datacenter providing the movie is not taken into consideration. This would increase the associated carbon emission. 

It is also notable how the 2 gridsearches make up about 1 _/_ 3 of the total energy consumption of the data science project, despite making up only 14% of the time (1 _h_ 17<sup>_′_</sup> out of 9 _h_ 04<sup>_′_</sup> ). This makes sense, as performing a gridsearch makes a CPU run in parallel under heavy load, yielding a high power throughput. This was already visibli in Figure 1. 

# _4.3. Carbon emission_ 

Figure 3 shows a log-scale of the carbon emission of various everyday tasks, common data science tasks, and advanced data science tasks. The carbon emission is expressed in units _km driven by car_ as described in Section 2.3. It is visible how all considered common data science tasks emit less _CO_<sup>2</sup> _eq_ than e.g. leaving the office lights on over the weekend. Even an advanced data science task such as training a CNN on images emits less _CO_<sup>2</sup> _eq_ , when performed under the same conditions as described by Anthony et al. (2020). Note that these conditions are specific and impose great variance. For example, training your model in Estonia will yield about 64 times more _CO_<sup>2</sup> _eq_ emission than training it in Iceland, and training for twice as many epochs will yield an emission about twice as big. The measurement is supposed to reflect a representative training session. 

# _4.4. Hardware_ 

Figure 4 shows a distribution of the Thermal Design Power (TDP) of all CPUs recorded by Passmark (2022). The distribution does not represent the relative 

frequency of these CPUs; each CPU is a single datapoint, independent of its popularity in modern hardware. It shows how the CPU used in this work is more efficient than most CPUs, yielding a TDP of 15 _._ 0 _W_ compared to the median TDP of 51 _._ 0 _W_ . Measurements of the tasks as described in Section 2.6.1 may yield higher values when performed on other hardware. 

# _4.5. Assumptions revisited_ 

Let us revisit the assumptions made in Section 2.1 and verify if they hold true, or in which way they should be adapted. They are numbered in the same way as in Section 2.1 

1. Anthony et al. (2020) shows that the RAM can easily make up to 50% of the power consumption during the training of a neural network. Measuring only the power consumption of the CPU is not representative, especially during RAM-needy code. It can be estimated that the results associated with the hardware-specific tasks as described in Section 2.6.1 are, at worst, only half of the true power consumption. 

2. The CPU used in this work may not be very representative for other hardware architectures. If we assume the median TDP of all CPUs recorded by Passmark (2022) is more representative for other CPUs, then the results associated with the hardware-specific tasks as described in Section 2.6.1 are 3 _._ 4 _×_ too small. The latter assumption, i.e. assuming the difference between CPU TDP values is representative for the difference in realistic power consumption across CPUs in various hardware structures, is yet another dangerous assumption. TDP values should be interpreted carefully, as they do not reflect realistic power consumption: 

   - The TDP values from the manufacturers are often incorrect (Cutress, 2018). 

   - They only apply when the CPU is under full load on all cores, which is rare. 

   - Actual power usage can be altered when the computer is running via power plans (e.g. switching to battery power). 

   - They don’t take into account the electrical usage of other components, such as the screen, hard drive, RAM, etc. 

In combination with the previous assumption about hardware, it is thus possible that a more realistic power consumption (and associated carbon emission) of the hardware specific tasks in this work 

7 



can be up to 6 _._ 8 _×_ larger when including an estimate for the RAM power consumption. The relative difference between regular computer usage and common data science as shown in Equation 4 is not influenced by this systematic error. 

3. A comparison between the author’s efficiency and methods to other data scientists is next to impossible to measure. This assumption retains its status of hypothesis. 

4. Looking at the RAPL measurements of power consumption shown in Figure 1, it is visible how the lower three measurements have very low variance, and the remaining measurements have a welldefined ‘blob’. The former indicates that the tasks had a low variance in power consumption throughout time and this was successfully captured with the considered sampling rate of 10 _s_ . The latter indicates that the most frequent value for power consumption (the ‘blob’) was successfully captured by the considered sampling rate, but outliers may not have been accurately measured. If the power consumption varied too quickly, a sampling rate of 10 _s_ was too large to properly capture this behaviour. As the blob is well-defined for each task, the general behaviour of the time-dependency of the power consumption is well captured, and this effect is not considered as substantial. 

5. The conversion of _gCO_<sup>2</sup> _eq_ to _km driven by car_ is based on the EU average from 2019. EEA (2021a) shows a clear declining trend in car emission. The values expressed in the latter unit in this paper will become increasingly outdated every year. Of course, due to the variance in carbon emission for each car, the values also vastly differ depending on which car you consider. 

6. As already mentioned at the end of Section 4.3, the conversion from energy to _gCO_<sup>2</sup> _eq_ depends heavily on which country you consider. Outsourcing heavy computational work to countries with a greener energy production can substantially decrease the carbon footprint. 

To estimate the additional contribution of common data science to the global GHGe, let’s use the results and the revisited assumptions to compare common data science to regular computer usage. When using the RAPL measurements, and scaling everything to the same duration, and under the assumptions that: 

1. the hardware-specific power consumption in this work can be extrapolated linearly to other hardware configurations, independent of the power load. 

2. the data science project and normal computer usage in this work are representative for other computer users, data scientists and data science in general. 

The extra load in global GHGe of common data science when compared to normal computer usage for a single device is equal to 



$$
EdatascienceP roject ENormalUsage = 8 hr 9.1 hr 100.9955 Wh 34.72 Wh = 2.57 (4)
$$

In order to estimate the additional load in GHGe due to common data science on a global scale, rather than for a single device, one would have to link this value to the relative global GHGe of household ICT devices of 45% (Belkhir & Elmeligi, 2018). To do so, one would need a value for how many of these devices are used for data science (and how often), which is very hard to obtain. 

# **5. Conclusion** 

The results made clear that common data science (i.e. any form of data science that can be performed on your laptop at home) requires substantially more power than other common computer tasks, yielding an associated carbon footprint that’s about 2 _._ 57 times higher than normal computer usage. Keeping in mind that data science jobs have been on a steady popularity increase over the last few years, this implies a higher global GHGe associated with common data science. 

When comparing common data science to advanced data science, it is clear how the power consumption increases exponentially along with the complexity of the data science. While advanced data science tasks, such as training GPT-3, require a substantial amount of power, this is not the case for common data science. 

The discussion revealed that the measurements for common data science in this work can be up to 6 _._ 8 times larger than reported. Even with this value in mind, common data science does not appear to contribute substantially to the global carbon emissions compared to other everyday tasks. While the uprise in data science jobs can indeed be associated with an increased carbon footprint, it will prove much more substantial to be mindful about the power consumption and associated carbon emission of some everyday tasks. Efficient heaters and lighting, insulation, green energy production and green travel methods will always beat, by a landslide, preferring e.g. a RandomsearchCV() over 

8 



a GridsearchCV() when doing common data science. 

That does not mean, however, that efforts to reduce the carbon footprint of data science are in vain. Common data science may be easier on the energy bill than bad insulation, but the computational needs of data science in general, including advanced data science, scale along with their complexity. If mindfulness about the carbon footprint of data science becomes ubiquitous, independent of the scale of the data science project, the difference in GHGe will scale accordingly. The bigger the model, the bigger the difference. 

# **References** 

- Anthony, L. F. W., Kanding, B., & Selvan, R. (2020). Carbontracker: Tracking and predicting the carbon footprint of training deep learning models. arXiv:2007.03051. 

- Bashroush, R., & Lawrence, A. (2020). Beyond pue: tackling it’s wasted terawatts. _Uptime Institute Available from: https://uptimeinstitute. com/beyond-puetackling-it’s-wastedterawatts [Accessed 11 June 2020]_ , . 

   - LinkedIn, N. (2022a). Linkedin jobs on the rise 2022: The 25 u.s. roles that are growing in demand. https://www. linkedin.com/pulse/linkedin-jobs-rise-202225-us-roles-growing-demand-linkedin-news. Last accessed on 2022-02-28. 

   - LinkedIn, N. (2022b). Linkedin jobs on the rise 2022: The roles that are growing in demand. https://www.linkedin. com/pulse/linkedin-jobs-rise-2022-rolesgrowing-demand-linkedin-news-europe/. Last accessed on 2022-02-28. 

   - Masanet, E., Shehabi, A., Lei, N., Smith, S., & Koomey, J. (2020). Recalibrating global data center energy-use estimates. _Science_ , _367_ , 984–6. doi:10.1126/science.aba3758. 

   - Passmark (2022). Cpu mega list. https://www. cpubenchmark.net/CPU_mega_page.html. Last accessed on 2022-03-01. 

   - Protocol, K. (1997). Kyoto protocol. _UNFCCC Website. Available online: http://unfccc.int/kyoto_protocol/items/ 2830.php (accessed on 1 January 2011)_ , . 

   - Rteil, N., Bashroush, R., Kenny, R., & Wynne, A. (2022). Interact: It infrastructure energy and cost analyzer tool for data centers. _Sustainable Computing: Informatics and Systems_ , _33_ , 100618. URL: https://www.sciencedirect.com/science/ article/pii/S2210537921001062. doi:https: //doi.org/10.1016/j.suscom.2021.100618. 

- Belkhir, L., & Elmeligi, A. (2018). Assessing ict global emissions footprint: Trends to 2040 & recommendations. _Journal of Cleaner Production_ , _177_ , 448–63. URL: https://www.sciencedirect.com/science/ article/pii/S095965261733233X. doi:https: //doi.org/10.1016/j.jclepro.2017.12.239. 

- Cao, Z., Zhou, X., Hu, H., Wang, Z., & Wen, Y. (2022). Towards a systematic survey for carbon neutral data centers. arXiv:2110.09284. 

- Cook, G., Lee, J., Tsai, T., Kongn, A., Deans, J., Johnson, B., Jardim, E., & Johnson, B. (2017). Clicking clean: Who is winning the race to build a green internet? Technical report, Greenpeace. 

- Cutress, I. (2018). Why intel processors draw more power than expected: Tdp and turbo explained. https://www.anandtech. com/show/13544/why-intel-processors-drawmore-power-than-expected-tdp-turbo. Last accessed on 2022-03-08. 

- EEA, E. E. A. (2021a). Co2 performance of new passenger cars in europe. https://www.eea.europa.eu/ims/co2performance-of-new-passenger. Last accessed on 2022-02-04. 

- EEA, E. E. A. (2021b). Greenhouse gas emission intensity of electricity generation by country. https://www.eea.europa.eu/ data-and-maps/daviz/co2-emission-intensity9. Last accessed on 2022-02-04. 

- electricpoint (n.d.). Electric heating room size calculator. https: //www.electricpoint.com/heating/electricheating/how-to-calculate-kw-required-toheat-a-room. Last accessed on 2022-03-01. 

- ICAO (2016). Icao carbon emissions calculator. https: //www.icao.int/environmental-protection/ CarbonOffset/Pages/default.aspx. Last accessed on 2022-03-01. 

- IEA (2021). Global data centre energy demand by data centre type, 2010-2022. https://www.iea.org/dataand-statistics/charts/global-data-centreenergy-demand-by-data-centre-type-2010-2022. Last accessed on 2022-03-08. 

9 

