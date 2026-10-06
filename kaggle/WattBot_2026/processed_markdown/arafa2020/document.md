# Verified Instruction-Level Energy Consumption Measurement for NVIDIA GPUs 

Yehia Arafa<sup>_∗_</sup> , Ammar ElWazir<sup>_∗_</sup> , Abdelrahman Elkanishy<sup>_∗_</sup> , Youssef Aly<sup>§</sup> , Ayatelrahman Elsayed<sup>_∗_</sup> , Abdel-Hameed Badawy<sup>_∗_†</sup> , Gopinath Chennupati<sup>†</sup> , Stephan Eidenbenz<sup>†</sup> , and Nandakishore Santhi<sup>†</sup> 

_∗_ New Mexico State University, Las Cruces, NM, USA 

_{_ yarafa, ammarwa, anasser, aynasser, badawy _}_ @nmsu.edu 

§ Arab Academy for Science, Technology & Maritime Transport, Alexandria, Egypt 

† Los Alamos National Laboratory, Los Alamos, NM, USA 

_{_ gchennupati, eidenben, nsanthi _}_ @lanl.gov 

**_Abstract_ —GPUs are prevalent in modern computing systems at all scales. They consume a significant fraction of the energy in these systems. However, vendors do not publish the actual cost of the power/energy overhead of their internal microarchitecture. In this paper, we accurately measure the energy consumption of various PTX instructions found in modern NVIDIA GPUs. We provide an exhaustive comparison of more than 40 instructions for four high-end NVIDIA GPUs from four different generations (** **_Maxwell, Pascal, Volta, and Turing_ ). Furthermore, we show the effect of the CUDA compiler optimizations on the energy consumption of each instruction. We use three different software techniques to read the GPU on-chip power sensors, which use NVIDIA’s NVML API and provide an in-depth comparison between these techniques. Additionally, we verified the software measurement techniques against a custom-designed hardware power measurement. The results show that** **_Volta_ GPUs have the best energy efficiency of all the other generations for the different categories of the instructions. This work should aid in understanding NVIDIA GPUs’ microarchitecture. It should also make energy measurements of any GPU kernel both efficient and accurate.** 

### **_Index Terms_ —GPU Power Usage, PTX, NVML, PAPI, Internal Power Sensors, External Power Meters** 

## I. INTRODUCTION 

Applications that rely on graphics processor units (GPUs) have increased exponentially over the last decade. GPUs are now used in various fields, from accelerating scientific computing applications to performing fast searches in dataoriented applications. A typical GPU has multiple streaming multiprocessors (SMs). Each can be seen as standalone processors operating concurrently. These SMs are capable of running thousands of threads in parallel. Over the last decade, GPUs’ microarchitecture has evolved to be very complicated. However, the increase in complexity means more processing power. Hence, the recent development of embedded/integrated GPUs and their application in edge/mobile computation have made power and energy consumption a primary metric for evaluating GPUs performance. Especially that researchers have shown that large power consumption has a significant effect on the reliability of the GPUs [1]. Hence, analyzing and predicting the power usage of the GPUs’ hardware components remains an active area of research for many years. 

Several monitoring systems (hardware & software) have been proposed in the literature [2]–[5] to measure the total power usage of GPUs. However, measuring the energy consumption of the GPUs’ internal hardware components is particularly challenging as the percentage of updates in the microarchitecture can be significant from one GPU generation/architecture to another. Moreover, GPU vendors never publish the data on the actual energy cost of their GPUs’ microarchitecture. 

In this paper, we provide an accurate measurement of the energy consumption of almost all the instructions that can execute in modern NVIDIA GPUs. Since the optimizations provided by the CUDA ( _NVCC_ ) compiler [6] can affect the latency of each instruction [7]. We show the effect of the CUDA compiler’s high-level optimizations on the energy consumption of each instruction. We compute the instructions energy at the PTX [8] granularity, which is independent of the underlying hardware. Thus, the measurement methodology introduced has minimum overhead and is portable across different architectures/generations. 

To compute the energy consumption, we use three different software techniques based on the NVIDIA Management Library (NVML) [2], which query the onboard sensors and read the power usage of the device. We implement two methods using the native NVML API, which we call _Sampling Monitoring Approach (SMA)_ , and _Multi-Threaded Synchronized Monitoring (MTSM)_ . The third technique uses the newly released CUDA component in the PAPI v.5.7.1 [9] API. Furthermore, we designed a hardware system to measure the power usage of the GPUs in real-time. The hardware measurement is considered as the ground truth to verify the different software measurement techniques. 

To the best of our knowledge, we are the first to provide a comprehensive comparison of the energy consumption of each PTX instruction in modern high-end NVIDIA GPGPUs. Furthermore, the compiler optimizations effect on the energy consumption of each instruction has not been explored before in the literature. Also, we are the first to provide an indepth comparison between different NVML power monitoring software techniques. 



In summary, the followings are this paper **contributions** : 

- 1) Accurate measurement of the energy consumption of almost all PTX instructions for four high-end NVIDIA GPUs from four different generations ( _Maxwell, Pascal, Volta, and Turing_ ). 

- 2) Show the effect of CUDA compiler optimizations levels on the energy consumption of each instruction. 

- 3) Utilize and Compare three different software techniques ( _SMA, MTSM, and PAPI_ ) to measure GPU kernels’ energy consumption. 

- 4) Verify the different software techniques against a custom in-house hardware power measurement on the _Volta TITAN V_ GPU. 

The results show that _Volta TITAN V_ GPU has the best energy efficiency among all the other generations for different categories of the instructions. Furthermore, our verification show that _MTSM_ leads to the best results since it integrates the power readings and captures the start and the end of the GPU kernel correctly. 

The rest of this paper is organized as follows: Section II provide a brief background on NVIDIA GPUs’ internal architecture. Section III describes the methodology of calculating the instructions energy consumption. Section IV depicts the differences between the software techniques. Section V shows the in-house direct hardware design. In Section VI we present the results. Section VII shows the related work and finally, Section VIII concludes the paper. 

## II. GPGPUS ARCHITECTURE 

GPUs consist of a large number of processors called _Streaming Multiprocessor_ (SMX) in CUDA terminology. These processors are mainly responsible for the computation part. They have several _scalar cores_ , which has some computational resources, including fully pipelined integer Arithmetic Units (ALUs) for performing 32-bit integer instruction, FloatingPoint units (FPU32) for performing floating-point operations, and Double-Precision Units (DPU) for 64-bit computations. Furthermore, it includes Special Function Units (SFU) that executes intrinsic instructions, and Load and Store units (LD/ST) for calculations of source and destination memory addresses. In addition to the computational resources, each SMX is coupled with a certain number of warp schedulers, instruction dispatch units, instruction buffer(s), and texture and shared memory units. Each SMX has a private L1 memory, and they all share access to L2 cache memory. The exact number of SMXs on each GPU varies with the GPU’s generation and the computational capabilities. 

GPU applications typically consist of one or more kernels that can run on the device. All _threads_ from the same kernel are grouped into a _grid_ . The grid is made up of many _blocks_ ; each is composed of groups of 32 threads called _warps_ . Grids and blocks represent a logical view of the thread hierarchy of a CUDA kernel. Warps execute instructions in a SIMD manner, meaning that all threads from the same warp execute the same instruction at any given time. 

![Figure](assets/figure_0001_page_0002.svg)Fig. 1. An Overview of the Compilation Procedure 

## III. INSTRUCTIONS ENERGY CONSUMPTION 

We designed special micro-benchmarks to stress the GPU to be able to capture the power usage of each instruction. 

We used Parallel-Thread Execution (PTX) [8] to write the micro-benchmarks. PTX is a virtual-assembly language used in NVIDIA’s CUDA programming environment. PTX provides an open-source machine-independent ISA. The PTX ISA itself does not run on the device but rather gets translated to another machine-dependent ISA named Source And Assembly (SASS). SASS is not open. NVIDIA does not allow writing native SASS instructions, unlike PTX, which provides a stable programming model for developers. There have been some research efforts [10], [11] to produce assembly tool-chains by reverse engineering and disassembling the SASS format to achieve better performance. Reading the SASS instructions can be done using CUDA binary utilities ( _cuobjdump_ ) [12]. The use of PTX helps control the exact sequence of instructions executing without any overhead. Since PTX is a machine-independent, the code is portable across different CUDA runtimes and GPUs. 

Figure 1 shows the compilation workflow, which leverages the compilation trajectory of the NVCC compiler. Since the PTX can only contain the code which gets executed on the device ( _GPU_ ), we pass the instrumented PTX device code to the NVCC compiler for linking at runtime with the host ( _CPU_ ) CUDA C/C++ code. PTX optimizing assembler ( _ptxas_ ) is first used to transform the instrumented machine-independent PTX code to a machine-dependent ( _SASS)_ instructions then to a CUDA binary file ( _.cubin_ ). The binary is used to produce a 



![Figure](assets/figure_0002_page_0003.svg)Fig. 2. Unsigned Div Instruction Microbenchmark written in PTX 

_fatbinary_ file, which gets embedded in the host C/C++ code. An empty kernel gets initialized in the host code, which is then gets replaced by the instrumented PTX kernel, which has the same header and the same name inside the ( _.fatbin.c_ ). The kernel is executed with one block one thread. 

Figure 2 shows an example of the instrumented PTX kernel for the unsigned _Div_ instruction. In our previous work [7], we presented a similar technique to find the instruction latency. We executed the instruction only once, and red the _clk_ register before and after its execution. The design here is different since we need to capture the change in power usage, which would be unnoticeable if we execute the instruction only once. The key idea here is unrolling a loop and execute the same instruction millions of times and record the power then divide by the number of instructions to get the power consumption of the single instruction. The kernel in Figure 2 shows an example of the micro-benchmark of the unsigned _div_ instruction. We begin by initializing the used registers, lines [3–5]. Since PTX is a virtual-assembly and gets translated to the SASS, there is no limit on the number of registers to use. Still, in the real SASS assembly, the number of registers is limited and will vary from one generation/architecture to another. When the limit exceeds, register variables will be spilled to memory, causing changes in performance. Line [10] sets the loop count to 1M iterations. The loop body, lines [13–27], is composed of 5 back-to-back unsigned _div_ instructions with dependencies, to make sure that the compiler does not optimize any of them. We do a load-add-store operation on the output of the 5<sup>th</sup> _div_ operation and begin the loop with new values each time to force the compiler to execute the instructions. Otherwise, the compiler would run the loop only the first time and squeeze 

![Figure](assets/figure_0003_page_0003.svg)Fig. 3. Add & Div kernels Power Consumption vs time on TITAN V GPU 

the remaining iterations. We follow the same approach for all the instructions, and the kernel is the same, the only difference is the instruction itself. 

GPUs drain power as static power and dynamic power. The static power is a constant power that the GPU consumes to maintain its operation. However, dynamic power is affected by the kernel’s instructions and operations. To eliminate the static power and any overhead dynamic power other than the instruction power consumption, we measure the power and compute the kernel’s energy consumption twice. First, we run the kernel as shown in figure 2, we call that the total energy. Second, while commenting out the back-to-back instructions (lines [16- 20]), we call that the overhead energy. We then use Eq. 1 to calculate the energy of the instruction. This way, only the real energy of the instruction is calculated. 



$$
Einstruction = Etotal -Eoverhead \# of instructions (1)
$$

## _A. NVCC Compiler Optimization_ 

The kernel is compiled with ( _–O3_ ) and ( _–O0_ ) optimization flags. This way, we capture the effect of the CUDA compiler’s higher levels of optimizations on the energy consumption of each PTX instruction. To make sure that in case of ( _–O3_ ), the compiler does not optimize the instructions and squeeze them, we made sure that the output of the kernel is correct. 



![Figure](assets/figure_0004_page_0004.svg)Line 28 of Figure 2, stores the output of the loop. We read it and validate its correctness. Furthermore, we validate the _clk_ register for each instruction against our previous work [7]. 

## IV. SOFTWARE MEASUREMENT 

NVIDIA provides an API named NVIDIA Management Library (NVML) [2], which offers direct access to the queries exposed via the command line utility, NVIDIA System Management Interface ( _nvidia-smi_ ). NVML allows developers to query GPU device states such as GPU utilization, clock rates, GPU temperature _etc._ Additionally, it provides access to the board power draw by querying its instantaneous onboard sensors. The community has widely used NVML since its first release with CUDA v4.1 in 2011. NVIDIA display driver is equipped with NVML, and the SDK offers the API for its use. We use NVML to read the device power usage while running the PTX micro-benchmarks and compute the energy of each instruction. There are several techniques for collecting power usage using NVML. We found that the methods do vary. Therefore, we provide an in-depth comparison of the quality of these techniques on the energy of the individual instructions. 

## _A. Sampling Monitoring Approach (SMA)_ 

The C-based API provided by NVML can query the power usage of the device and provide an instantaneous power measurement. Therefore, it can be programmed to keep reading the hardware sensor with a certain frequency. This basic approach is popular and was used in other related works [13], [14]. The _nvmlDeviceGetPowerUsage()_ function is used to retrieve the power usage reading for the device, in _milliwatts_ . This function is called and executed by the CPU. We configured the sampling frequency of reading the hardware sensors to its maximum, 66.7 Hz [13] (15 ms window between each call to the function). 

We read the power sensor according to the sample interval in the background while the micro-benchmarks are running. Example of the output using this approach are shown in Figures 3(a) and 3(b). The two figures show the power 

consumption over time for integer _Add_ and unsigned integer _Div_ kernels for the _TITAN V (Volta)_ GPU. The power usage jumps shortly after the launch of the kernel and decreases in steps after the kernel finishes execution until it reaches the steady-state. This is done in 22 sec and 33 sec windows interval for _Add_ and _Div_ respectively. If we calculate the two kernels actual elapsed time, it takes only 0.28 sec and 13 sec for the _Add_ and the _Div_ kernels, respectively. That is, the GPU does something before and after the actual kernel execution. Hence, identifying the window of the kernel is hard and would affect the output as the power consumption varies through time. One solution is to take the maximum reading between the two steady states, but this would be misleading for some kernels, especially the bigger ones. Therefore, we ignore this approach from reporting owing to these issues. 

## _B. PAPI API_ 

Performance Application Programming Interface (PAPI) [15] provides an API to access the hardware performance counters found on modern processors. We can read different performance metrics through either a simple programming interface from either C or Fortran programming languages. Researchers have used PAPI as a performance and power monitoring library for different hardware and software components [15]–[19]. It is also used as a middleware component in different profiling and tracing tools [20]. 

PAPI can work as a high-level wrapper for different components; for example, it uses the Intel RAPL interface [21] to report the power usage and energy consumption for Intel CPUs. Recently, PAPI version 5.7 added the NVML component, which supports both measuring and capping power usage on modern NVIDIA GPU architectures. 

The advantage of using PAPI is that the measurements are by default synchronized with the kernel execution. The target kernel is invoked between the _papi start_ , and the _papi end_ functions, and a single number, representing the power event we need to measure is returned. The NVML component implemented in PAPI uses the function, _getPowerUsage()_ which 



query _nvmlDeviceGetPowerUsage()_ function. According to the documentation, this function is called only once when the _papi_ _<u>end</u>_ is called. Thus, the power returned using this method is an instantaneous power when the kernel finishes execution. Although synchronizing with the kernel solves the _SMA_ issues, taking the instantaneous measurement when the kernel finishes execution can provide non-accurate results especially, for large and irregular kernels as shown in Section VI. Note that PAPI provides an example that works like the _SMA_ approach, which we refrain from this paper. 

## _C. Multi-Threaded Synchronized Monitoring (MTSM)_ 

In MTSM, we identify the exact window of the kernel execution. We modified _SMA_ to synchronize the kernel execution. This way, only the power readings of the kernel are recorded. Since the host CPU monitors the NVML API, we use Pthreads for synchronization where one thread calls and monitors the kernel while the other thread records the power. 

Algorithm 1 shows the MTSM. We initialize a volatile atomic variable ( _flag_ ) to zero, which we use later to record the power readings according to the start and end of the target kernel. On line 6 we create a new thread ( _th1_ ) which executes a function ( _func1_ ) [line 17] in parallel. This function completes the power monitoring, depending on the atomic _flag_ . This uses the NVML function, _nvmlDeviceGetPowerUsage()_ which returns the device power in milli-watts. The readings of the power during the kernel window are recorded and saved in an array ( _power readings_ ), which is used later in computing the kernel energy. In lines [7–12], flip the _flag_ value and start computing the elapsed time and the launch kernel, which means starting the power monitoring. At the end of the kernel execution, we record the elapsed time and change the flag. We use the CUDA synchronize function to make sure that the power is recorded correctly. We do not specify any reading sampling frequency for the NVML functions. Although this would give us redundant values, it would be more accurate. With this setup, we found that the power reading frequency is nearly 2 _kHz_ . 

Figures 4(a) and 4(b) show the corresponding kernels in Figures 3(a) and 3(b) after identifying the exact kernel execution window. The new graphs are annotated with the start and end of the kernel. We observe that the kernel does not start after the sudden rise in the power from the steady-state, rather after a couple of ms from this sudden increase in power consumption (see _add_ kernel in Figure 4(a) for clarity). After the kernel finishes execution, the power remains high for a small-time, and then it starts descending in steps until it reaches the steadystate again. To compute the kernel’s energy, we calculated the area under the curve for the kernel using Eq. 2. We believe that this approach would provide the most accurate measurement since the power readings of only the kernel are recorded. Computing the energy as the area under the curve is more rigorous than just taking the last power reading multiplied by the time elapsed for the kernel, as is done in PAPI. 

We configured _MTSM_ as a shared library that can be linked with the application binaries at runtime. The code 

**Algorithm 1** MTSM Approach 

- 1: **volatile** _elapsed_ _<u>time</u>_ , _energy_ 2: **volatile atomic** _flag ←_ 0 3: **procedure** _kernel_ _<u>energy</u>_ : 4: **time t** _time ←_ 0 5: _pthread_ _<u>create</u>_ [ _th_ 1 _, func_ 1] 6: _flag ←_ 1 7: _start_ _<u>timing</u>_ (& _time_ ) 8: _Kernel call ≪ Dg, Db ≫_ 9: _end_ _<u>timing</u>_ ( _synchronize,_ & _time_ ) 

- 10: _flag ←_ 0 11: _elapsed_ _<u>time ←</u> time_ 12: _pthread join_ ( _th_ 1) 13: **return** _energy_ 14: **end procedure** 15: **procedure** _func_ 1 : 16: _power readings ←_ [ ] 17: _monitor power_ (& _power readings_ ) 18: _energy ← calculate energy_ (& _power readings_ ) 19: **end procedure** 20: **procedure** _monitor power_ ( _power_ _<u>readings</u>_ ): 21: **while** flag **do** 22: _power readings ← read NVML power usage_ 23: **end while** 24: **end procedure** 

is first compiled and then injected or preloaded at runtime using LD <u>PRELOAD [22] environment variable to any device</u> executable binary file with a kernel that executing on NVIDIA GPUs. The start timing and end timing are automatically triggered by intercepting the CUDA runtime API [23] calls. 



$$
E (mJ) = Elapsed time (sec) \# of power readings (N) \times{}{} N \sum{}{} i=0 Poweri (mW) (2)
$$

## V. HARDWARE MEASUREMENT 

Modern GPUs have two primary sources of power. The first power source is the direct DC power (12 _V_ ) supply, provided through the side of the card. While the second one is the PCI-E (3 _._ 3 _V_ and 12 _V_ ) power source, provided through the motherboard. We have designed a system to measure each power source in real-time. The hardware measurement is considered as the ground truth to verify the different software measurement techniques. 

Figure 5 shows the experimental hardware setup with all the components. To capture the total power, we measure the current and voltage for each power source simultaneously. A clamp meter and a shunt series resistor are used for the current measurement. For voltage measurement, we use a direct probe on the voltage line using an oscilloscope to acquire the signals. Equation 3 is used to calculate the total hardware power drained by the GPU from the two different power sources. 



![Figure](assets/figure_0005_page_0006.svg)Fig. 5. Hardware Measurement Setup 

**Direct DC Power Supply Source:** Power supply provides a 12 _V_ voltage through a direct wired link. We use both a 6- pin and 8-pin PCI-E power connectors to deliver a maximum of 300 _W_ . Thus, the direct DC power supply source is the main contributor to the card’s power. Figure 5 shows a clamp meter measuring the current of the direct power supply connection. The voltage of the power supply is measured using an oscilloscope probe. The current and voltage are acquired using an oscilloscope, as shown in Figure 5. Therefore, the Direct DC power supply source is calculated using simple multiplication. The third addition term in Eq. 3 shows the calculation of the power which is multiplying _Iclamp_ by _V DPS_ . In which, _V DPS_ is the voltage of the direct power supply. 

**PCI-E Power Source:** Graphics cards are connected to the motherboard through the PCI-E x16 slot connection. 3 _._ 3 _V_ and 12 _V_ voltages are provided through this slot. To accurately measure the power that goes through this slot, an intermediate power sensing technique should be installed between the card and the motherboard. We designed a custom made PCI-E riser board that measures the power supplied through the motherboard. Two in-series shunt resistors are used as a power sensing technique. As shown in Figure 6, each shunt resistor ( _RS_ ) is connected in series with 3 _._ 3 _V_ and 12 _V_ separately. Using the series property, the current that flows through the _RS_ is the same current that goes to the graphics card. Therefore, we measure the voltages _V S_ 1 and _V G_ 1 which are across _RS_ using oscilloscope. We then divide it with the _RS_ value. The voltage level is measured using the riser board. We duplicate the same calculation technique for the 3 _._ 3 _V_ voltage level, as shown in Eq. 3. 



$$
PHW = Vs1 -Vg1 Rs \times{}{}Vg1 + Vs2 -Vg2 Rs \times{}{}Vg2 +Iclamp \times{}{}VDPS (3)
$$

![Figure](assets/figure_0006_page_0006.svg)Fig. 6. Circuit Diagram of Hardware Measurement 

## VI. RESULTS 

We show the energy consumption of each instruction found in the latest PTX ISA, v.6.4 [8]. We report the results of using _MTSM_ and _PAPI_ on four different NVIDIA GPUs from four different generations/architectures; **_GTX TITAN X:_** GPU [24] from Maxwell architecture. It has 3584 cores with 151 MHz clock frequency. **_GTX 1080 Ti:_** GPU [25] from Pascal architecture. It has 3584 cores with 1481 MHz clock frequency. **_TITAN V:_** GPU [26] from Volta architecture. It has 5120 cores with 1200 MHz clock frequency. **_TITAN RTX:_** GPU [27] from Turing architecture [28]. It has 4608 cores with 1350 MHz clock frequency. 

We used CUDA NVCC compiler v.10.1 [6] to compile and run the codes. CUDA compiler comes equipped with NVML library [2]. Table I shows an enumeration of the energy consumption of the various ALU instructions for the different GPUs. For simplicity, we used each GPU generation to refer it. We denote the ( _O3_ ) version as _Optimized_ and the ( _O0_ ) version as _Non-Optimized_ . 

The results show that overall _Volta_ GPUs have the lowest energy consumption per instruction among all the tested GPUs. _Pascal_ preceded the _Volta_ while _Maxwell_ and _Turing_ are power hungry devices except for some categories of the instructions. 

For _Half Precision_ (FP16) instructions, _Volta_ and _Turing_ have much better results than _Pascal_ . Hence, this confirms that both architectures are suitable for approximate computing applications ( _e.g._ , deep learning, and energy-saving computing). We did not run FP16 instructions on _Maxwell_ as _Pascal_ architecture was the first GPU that offered FP16 support. The same trend can be found in _Multi Precision_ (MP) instructions where _Volta_ and _Pascal_ have better energy consumption compared to the two other generations. MP [29] instructions are essential in a wide variety of algorithms in computational mathematics ( _i.e._ , number theory, random matrix problems, experimental mathematics). Also, it is used in cryptography algorithms and 



![Figure](assets/figure_0007_page_0007.svg)Fig. 7. Instructions-level verification of **MTSM** & **PAPI** against the **HW** on _Volta TITAN V_ GPU. _<_ Int _>_ , _<_ F _>_ & _<_ D _>_ denote _Integer_ , _Double_ and _Float_ instructions respectively 

## security. 

Overall, the energy of _Non-Optimized_ is always more than the _Optimized_ . One reason is that the number of cycles at the _(O0)_ optimization level are more than the _(O3)_ level [7]. This can be because the translation from PTX instruction to native SASS instruction is not one-to-one conversion. Thus, the instruction can take more time to finish execution if it got translated to more than one instruction. 

**PAPI vs. MTSM:** The dominant tendency of the results is that PAPI readings are always more than the _MTSM_ . Although the difference is not significant for small kernels, it can be up to 1 _µ_ J for bigger kernels like Floating Single and Double Precision _div_ instructions. 

## _A. Verification with the Hardware Measurement_ 

We verified the different software techniques (MTSM & PAPI) against the hardware setup on _Volta TITAN V_ GPU. Compared to the ground truth hardware measurements, for all the instructions, the average Mean Absolute Percentage Error (MAPE) of _MTSM_ Energy is **6.39** and the mean Root Mean Square Error (RMSE) is **3.97** . In contrast, PAPI average MAPE is **10.24** and the average RMSE is **5.04** . Figure 7 shows the error of _MTSM_ and PAPI relative to the hardware measurement for some of the instructions. The results prove that _MTSM_ is more accurate than PAPI as it is closer to what has been measured using the hardware. 

## VII. RELATED WORK 

Several works [31], [32] in the literature tried to directly measure the instantaneous power usage of the GPUs using various profiling techniques. On the other hand, Researchers have proposed different techniques [33]–[35] to indirectly estimate and predict the total GPU’s power/energy consumption. Additional details are discussed by Bridges _et al._ [30]. 

GPU power profiling can be carried out in two different approaches, a software-oriented solution, where the internal power sensors are queried using NVML, and hardwareoriented solutions using external hardware setups. 

**Software-oriented approaches:** Arunkumar _et al._ [14] used a direct NVML sampling motoring approach running in the background while using a special micro-benchmark to 

calculate basic compute/memory instructions energy consumption and feed that to their model. They run their evaluation on ( _Tesla K40_ ) Kepler GPU. They intentionally disabled all compiler optimizations and compiled their micro-benchmarks with ( _–O3_ ) flag. Burtscher _et al._ [13] analyzed the power consumption measured by NVML for ( _Tesla K20_ ) GPU. Kasichayanula _et al._ [31] used NVML to calculate the energy consumption of some GPU units which drive their model and validate it with a Kill-A-Watt power meter. While these types of hardware power meters are cheap and straightforward to use, they do not give an accurate measurement, especially in HPC settings. 

**Hardware-oriented approaches:** Zhao _et al._ [36] used an external power meter on an old GPU ( _GeForce GTX 470_ ) from Fermi [37] architecture, where they designed a microbenchmark to compute the energy of some PTX instructions and feed that into their model. The authors of [32] validate their roofline model by using PowerMon 2 [5] and a custom PCIe inter-poser to calculate the instantaneous power of ( _GTX 580_ ) GPU. 

Recently, Sen _et al._ [38] assessed the quality and performance of the power profiling mechanisms using hardware and software techniques. They compared a hardware approach using PowerInsight [4] (a hardware power instrumentation product) to the software NVML approach on a developed matrix multiplication CUDA benchmark. 

In a similar spirit, we follow the same line of research. Nevertheless, we focus on the energy consumption of individual instructions while having a detailed comparison of the different software/hardware approaches. 

## VIII. CONCLUSION & FUTURE DIRECTIONS 

In this paper, we accurately measure the energy consumption of various PTX instructions that execute on NVIDIA GPUs. We also show the effects of different optimization levels of the CUDA ( _NVCC_ ) compiler on energy consumption of each instruction. We provide an in-depth comparison of various software techniques that query the onboard internal GPU sensors and verify against an in-house custom-designed hardware power measurement. Overall, the paper provides an easy and straightforward way ( _Multi-Threaded Synchronized Monitoring (MTSM)_ ) that can be used to measure the energy consumption of any NVIDIA GPU kernel<sup>1</sup> . Furthermore, the results give GPU architects and developers a concrete understanding of NVIDIA GPUs’ microarchitecture. This work will help GPU modeling frameworks [39] to have a precise prediction of energy/power consumption of GPUs. Along with GPU/CPU memory [40]–[42] and pipeline [43] models, a heterogeneous system can be accurately modeled [44]. 

## APPENDIX: ENERGY CONSUMPTION RESULTS 

Table I has per-instruction energy breakdown for different generations of NVIDIA GPUs. 

> 1The source code is available on our laboratory page on Github at https: //github.com/NMSU-PEARL/GPUs-Energy. 



TABLE I 

Energy Consumption of GPU Instructions. _{_ s _}_ & _{_ u _}_ denote signed and unsigned instructions respectively. The first number in the results is PAPI and the second one is the MTSM. All the numbers are in ( _µ_ J). 

|||**_Optim_**|**_ized_**|||**_Non-Opt_**|**_imized_**||
|---|---|---|---|---|---|---|---|---|
||**Maxwell**|**Pascal**|**Volta**|**Turing**|**Maxwell**|**Pascal**|**Volta**|**Turing**|
|**_Instruction_**||||**_(1) Integer Ari_**|**_thmetic Instructions_**||||
|add / sub /<br>min / max|0.0942 , 0.0461|0.0277 , 0.0200|0.0064 , 0.0012|0.0293 , 0.0281|1.2453 , 1.0264|0.6509 , 0.6203|0.2531 , 0.2384|0.8340 , 0.7905|
|mul / mad|3.0239 , 2.7309|0.2853 , 0.1727|0.0092 , 0.0014|0.0434 , 0.0233|4.5826 , 4.2959|3.6194 , 3.5986|0.5228 , 0.4912|0.9969 , 0.9675|
|_{_s_}_ div|10.5921 , 6.7819|5.0270 , 4.9889|4.2489 , 4.0660|7.2119 , 6.5499|64.7649 , 64.6306|44.5100 , 44.5609|27.4008 , 27.0584|48.7700 , 48.4940|
|_{_s_}_ rem|7.8512 , 6.6833|5.0539 , 4.9687|4.2100 , 4.0138|7.3197 , 6.7982|61.1036 , 61.3000|42.4800 , 42.0521|25.4413 , 25.1175|48.3881 , 47.9075|
|abs|0.800 , 0.747|0.2927 , 0.2349|0.0647 , 0.0621|0.3000 , 0.2710|2.1611 , 1.8841|1.2170 , 1.2448|1.4263 , 1.4013|2.3084 , 2.3880|
|_{_u_}_ div|7.44783 , 6.2899|4.7398 , 4.5889|3.9254 , 3.8706|6.6068 , 6.038|52.5558 , 52.3220|36.2400 , 36.0963|20.4411 , 20.2517|35.8200 , 35.6736|
|_{_u_}_ rem|7.5357 , 6.4006|4.8380 , 4.7603|3.9587 , 3.9471|6.8026 , 6.3093|50.6491 , 50.4818|35.0700 , 34.8370|19.6811 , 19.2906|35.1347 , 35.0062|
|||||**_(2) Logic and_**|**_Shift Instructions_**||||
|and / or / not /<br>xor|0.0942 , 0.0461|0.0277, 0.0200|0.0064 , 0.0012|0.0293 , 0.0281|1.2453 , 1.0264|0.6509 , 0.6203|0.2531 , 0.2384|0.8340 , 0.7905|
|cnot|0.3362 , 0.0343|0.3227 , 0.2423|0.0071 , 0.0077|0.2840 , 0.1011|2.0562 , 1.7680|1.8762 , 1.8498|2.3421 , 2.3174|3.9990 , 3.9181|
|shl / shr|0.0942 , 0.0461|0.0277 , 0.0200|0.0064 , 0.0012|0.0293 , 0.0281|1.2453 , 1.0264|0.6509 , 0.6203|0.2531 , 0.2384|0.8340 , 0.7905|
|||||**_(3) Floating Singl_**|**_e Precision Instructions_**||||
|add / sub /<br>min / max|0.0942 , 0.0461|0.0277 , 0.0200|0.0064 , 0.0012|0.0293 , 0.0281|1.2453 , 1.0264|0.6509 , 0.6203|0.2531 , 0.2384|0.8340 , 0.7905|
|mul / mad /<br>fma|3.0239 , 2.7309|0.2778 , 0.2008|0.0021 , 0.0014|0.2933 , 0.2811|4.5826 , 4.2959|0.6509 , 0.6203|0.4874 , 0.4797|0.8340 , 0.7905|
|div|10.6203 , 9.4351|6.9934 , 6.8707|5.1096 , 5.0355|8.6425 , 7.9232|57.2252 , 56.6529|50.4741 , 49.9350|34.1050 , 33.3816|58.8700 , 58.6767|
|||||**_(4) Double Pr_**|**_ecision Instructions_**||||
|add / sub /<br>min / max|2.3058 , 2.0061|1.8610 , 1.8606|0.3608 , 0.3567|2.6810 , 2.6176|3.3017 , 2.5143|2.0070 , 2.0586|0.5158 , 0.5114|4.6315 , 4.1623|
|div|30.0634 , 28.8160|19.6393 , 19.2843|3.7249 , 3.6828|25.7757 , 23.6016|101.3056 , 100.7807|50.3810 , 50.0800|31.0212 , 30.4121|67.4127 , 67.4025|
|||||**_(5) Half Prec_**|**_ision Instructions_**||||
|add / sub /|NA|2.9601 , 2.8788|0.0924 , 0.0624|0.3740 , 0.1220|NA|3.5727 , 3.4259|0.5027 , 0.4656|0.9972 , 0.9631|
|mul|||||||||
|||||**_(6) Multi Pre_**|**_cision Instructions_**||||
|add.cc / addc /<br>sub.cc|0.3922 , 0.0791|0.3152 , 0.1492|0.0669 , 0.0535|0.1293 , 0.1065|1.2502 , 1.0270|0.6685 , 0.6317|0.5187 , 0.4938|0.9979 , 0.9680|
|subc|0.6934 , 0.3593|0.3655 , 0.3499|0.1006 , 0.089|0.4339 , 0.1677|2.1672 , 1.8927|1.2646 , 0.9002|0.9889, 0.952467|1.9107 , 1.8704|
|mad.cc / madc|1.1575 , 0.7697|0.7981 , 0.6768|0.0730 , 0.0631|0.1621 , 0.1357|4.7049 , 4.4307|3.7018 , 3.6865|0.5165 , 0.5043|0.9979 , 0.9657|
|||||**_(7) Special Math_**|**_ematical Instructions_**||||
|rcp|6.4492 , 5.3416|4.1609 , 4.0320|2.4265 , 2.4270|4.3064 , 3.9514|18.6762 , 18.2830|13.1662 , 13.1460|10.2930 , 10.0538|19.3208 , 19.2601|
|sqrt|6.3630 , 5.3923|4.1114 , 4.0068|2.4349 , 2.4219|4.3816 , 4.0533|19.0402 , 18.6694|13.4900 , 13.4185|10.5023 , 10.2700|19.7800 , 19.6984|
|approx.sqrt|0.8527 , 0.4961|0.3598 , 0.2345|1.2311 , 1.2076|2.1648 , 2.0121|15.9024 , 15.5452|10.7200 , 10.6812|8.3867 , 8.2517|15.4991 , 15.4438|
|rsqrt|0.5174 , 0.303|0.2573 , 0.1163|1.2488 , 1.2432|2.2491 , 2.0898|15.0459 , 14.6802|10.6800 , 10.6920|8.3784 , 8.2320|15.8300 , 15.7700|
|sin / cos|0.3410 , 0.1507|0.1345 , 0.2742|0.5887 , 0.5867|1.0070, 0.9065|1.2927 , 0.8940|1.1390 , 0.8650|1.0046 , 0.9788|1.9340 , 1.8835|
|lg2|0.5075 , 0.3098|0.3618 , 0.2371|1.2357 , 1.2287|2.1451 , 2.1634|14.6789 , 15.0598|10.7646 , 10.6786|8.4058 , 8.2500|15.6505 , 15.6127|
|ex2|0.5147 , 0.3094|0.2383 , 0.3372|0.4798 , 0.4709|1.0188 , 0.6971|14.0001 , 13.6377|9.9840 , 9.9685|7.3144 , 7.2030|13.5252 , 13.4070|
|copysign|0.2099 , 0.1700|0.2989 , 0.2339|0.0910, 0.0880|0.1627 , 0.1379|3.8932 , 3.5953|3.1134 , 3.1020|2.3692 , 2.3490|4.0546 , 3.9487|
|||||**_(8) Integer In_**|**_trinsic Instructions_**||||
|mul24() /<br>mad24()|0.3915 , 0.2939|0.2853 , 0.2727|0.2263 , 0.2119|0.3713 , 0.3415|6.7332 , 6.4263|4.8636 , 4.8636|2.3732 , 2.3249|4.6364 , 4.5942|
|sad()|0.0316 , 0.015|0.2523 , 0.1243|0.0075 , 0.0038|0.2428 , 0.0422|1.2495, 1.0277|0.6371 , 0.6646|0.5158 , 0.5029|1.0100 , 0.9757|
|popc()|0.074 , 0.057|0.1347 , 0.2674|0.3968 , 0.3990|0.0815 , 0.0603|2.0281 , 1.7728|1.89123 , 1.9133|1.4984 , 1.4601|2.8428 , 2.7949|
|clz()|0.0729 , 0.0479|0.2644 , 0.3339|0.5683 , 0.5657|0.3124 , 0.2817|2.0755, 1.7944|1.1670 , 1.2034|0.9145 , 0.8961|1.1554 , 1.4956|
|bfind()|0.0488 , 0.0374|0.2326 , 0.3081|0.2915 , 0.2902|0.0304 , 0.0052|1.1997 , 0.9821|0.5912 , 0.6185|0.4688 , 0.4582|0.8010 , 0.7546|





## REFERENCES 

- [1] L. B. Gomez, F. Cappello, L. Carro, N. DeBardeleben, B. Fang,S. Gurumurthi, K. Pattabiraman, P. Rech, and M. S. Reorda, “Gpgpus: How to combine high computational power with high reliability,” in _Proceedings of the Conference on Design, Automation & Test in Europe,_ DATE 2014. 

- [2] NVIDIA Corporation. (2019) Nvidia management library (nvml). [Online]. Available: https://docs.nvidia.com/deploy/nvml-api 

- [3] J. W. Romein and B. Veenboer, “Powersensor 2: A fast power measurement tool,” in _2018 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS),_ April 2018, pp. 111–113. 

- [4] J. H. Laros, P. Pokorny, and D. DeBonis, “Powerinsight - a commodity power measurement capability,” in _2013 International Green Computing Conference Proceedings,_ ser. IGCC, 2013, pp. 1–6. 

- [5] D. Bedard, M. Y. Lim, R. Fowler, and A. Porterfield, “Powermon: Finegrained and integrated power monitoring for commodity computer systems,” in _Proceedings of the IEEE SoutheastCon 2010 (SoutheastCon),_ 2010, pp. 479–484. 

- [6] NVIDIA Corporation. (2019) Cuda compiler driver nvcc. [Online]. Available: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc 

- [7] Y. Arafa, A. A. Badawy, G. Chennupati, N. Santhi, and S. Eidenbenz, “Low overhead instruction latency characterization for nvidia gpgpus,” in _2019 IEEE High Performance Extreme Computing Conference (HPEC),_ 2019, pp. 1–8. 

- [8] NVIDIA Corporation. (2019) Parallel thread execution ISA. [Online]. Available: https://docs.nvidia.com/cuda/pdf/ptxisa6.4.pdf 

- [9] Performance Application Programming Interface (PAPI). (2019)Version 5.7. [Online]. Available: https://icl.utk.edu/papi 

- [10] X. Zhang, G. Tan, S. Xue, J. Li, K. Zhou, and M. Chen, “Understanding the gpu microarchitecture to achieve bare-metal performance tuning,” in _Proceedings of the 22nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming,_ ser. PPoPP, 2017, pp. 31–43. 

- [11] S. Gray, _MaxAs: Assembler for NVIDIA Maxwell architecture,_ 2011. [Online]. Available: https://github.com/NervanaSystems/maxas 

- [12] NVIDIA Corporation. (2019) CUDA Binary Utilities. [Online]. Available: https://docs.nvidia.com/cuda/pdf/CUDABinaryUtilities.pdf 

- [13] M. Burtscher, I. Zecena, and Z. Zong, “Measuring gpu power with the k20 built-in sensor,” in _Proceedings of Workshop on General Purpose Processing Using GPUs,_ ser. GPGPU, 2014, pp. 28-36. 

- [14] A. Arunkumar, E. Bolotin, D. Nellans, and C.-J. Wu, “Understanding the future of energy efficiency in multi-module gpus,” in _2019 IEEE International Symposium on High Performance Computer Architecture (HPCA),_ 2019, pp. 519–532. 

- [15] D. Terpstra, H. Jagode, H. You, and J. Dongarra, “Collecting performance data with papi-c,” in _Tools for High Performance Computing,_ 2010, pp. 157–173. 

- [16] A. D. Malony, S. Biersdorff, S. Shende, H. Jagode, S. Tomov, G. Juckeland, R. Dietrich, D. Poole, and C. Lamb, “Parallel performance measurement of heterogeneous parallel systems with gpus,” in _International Conference on Parallel Processing,_ ser. ICPP, 2011, pp. 176–185. 

- [17] V. M. Weaver, M. Johnson, K. Kasichayanula, J. Ralph, P. Luszczek, D. Terpstra, and S. Moore, “Measuring energy and power with papi,” in _2012 41st International Conference on Parallel Processing Workshops,_ ser. ICPPW, 2012, pp. 262–268. 

- [18] H. McCraw, D. Terpstra, J. Dongarra, K. Davis, and R. Musselman, “Beyond the cpu: Hardware performance counter monitoring on blue gene/q,” in _International Supercomputing Conference (ISC),_ 2013, pp. 213–225. 

- [19] A. Haidar, H. Jagode, A. YarKhan, P. Vaccaro, S. Tomov, and J. Dongarra, “Power-aware computing: Measurement, control, and performance analysis for intel xeon phi,” in _IEEE High Performance Extreme Computing Conference (HPEC),_ 2017, pp. 1–7. 

- [20] A. Agelastos, et al., “The lightweight distributed metric service: A scalable infrastructure for continuous monitoring of large scale computing systems and applications,” in _SC ’14: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis,_ 2014, pp. 154–165. 

- [21] H. David, E. Gorbatov, U. R. Hanebutte, R. Khanna, and C. Le, “Rapl:Memory power estimation and capping,” in _ACM/IEEE International Symposium on Low-Power Electronics and Design (ISLPED),_ 2010, pp. 189–194. 

- [22] Linux Programmer’s Manual, 2019, [Online]. Available: http://man7.org/ linux/man-pages/man8/ld.so.8.html 

- [23] NVIDIA Corporation. (2019) CUDA Runtime API. [Online]. Available: https://docs.nvidia.com/cuda/cuda-runtime-api 

- [24] TechPowerUp. (Mar. 2015)) NVIDIA GeForce GTX TITAN X Specs. [Online]. Available: https://www.techpowerup.com/gpu-specs/ geforce-gtx-titan-x.c2632 

- [25] TechPowerUp. (Mar. 2017)) NVIDIA GeForce GTX 1080 Ti Specs. [Online]. Available: https://www.techpowerup.com/gpu-specs/ geforce-gtx-1080.c2839 

- [26] NVIDIA Corporation. (Jun. 2017)) Volta Tesla V100 GPU Architecture. [Online]. Available: http://images.nvidia.com/content/volta-architecture/ pdf/volta-architecture-whitepaper.pdf 

- [27] TechPowerUp. (Dec. 2018)) NVIDIA TITAN RTX Specs. [Online]. Available: https://www.techpowerup.com/gpu-specs/titan-rtx.c3311 

- [28] NVIDIA Corporation. (2018)) Turing GPU Architecture Whitepaper. 

- [29] N. Emmart, “A study of high performance multiple precision arithmetic on graphics processing units,” Ph.D. dissertation, UMASS, 2018. [Online]. Available: https://scholarworks.umass.edu/dissertations <u>2/1164</u> 

- [30] R. A. Bridges, N. Imam, and T. M. Mintz, “Understanding gpu power: A survey of profiling, modeling, and simulation methods,” in _ACM Comput. Surv,_ vol. 49, no. 3, pp. 41:1–41:27, Sep. 2016. 

- [31] K. Kasichayanula, D. Terpstra, P. Luszczek, S. Tomov, S. Moore, and G. D. Peterson, “Power aware computing on gpus,” in _Symposium on Application Accelerators in High Performance Computing (SAAHPC),_ 2012, pp. 64–73. 

- [32] J. W. Choi, D. Bedard, R. Fowler, and R. Vuduc, “A roofline model of energy,” in _27th International Symposium on Parallel and Distributed Processing (IPDPS),_ 2013, pp. 661–672. 

- [33] S. Hong and H. Kim, “A roofline model of energy,” in _Proceedings of the 37th Annual International Symposium on Computer Architecture,_ ser. ISCA, 2010, pp. 280–289. 

- [34] J. Leng, T. Hetherington, A. ElTantawy, S. Gilani, N. S. Kim, T. M. Aamodt, and V. J. Reddi, “Gpuwattch: Enabling energy optimizations in gpgpus,” in _Proceedings of the 40th Annual International Symposium on Computer Architecture,_ ser. ISCA, 2013. 

- [35] J. Lucas, S. Lal, M. Andersch, M. Alvarez-Mesa, and B. Juurlink, “How a single chip causes massive power bills gpusimpow: A gpgpu power simulator,” in _IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS),_ 2013, pp. 97–106. 

- [36] Q. Zhao, H. Yang, Z. Luan, and D. Qian, “Poigem: A programmingoriented instruction level gpu energy model for cuda program,” in _Proceedings of the 13th International Conference on Algorithms and Architectures for Parallel Processing,_ Springer, 2013, pp. 129–142. 

- [37] C. M. Wittenbrink, E. Kilgariff, and A. Prabhu, “Fermi gf100 gpu architecture,” _IEEE Micro,_ vol. 31, no. 2, p. 50–59, Mar. 2011. 

- [38] S. Sen, N. Imam, and C. Hsu, “Quality assessment of gpu power profiling mechanisms,” in _IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW),_ 2018, pp. 702–711. 

- [39] Y. Arafa, A. A. Badawy, G. Chennupati, N. Santhi, and S. Eidenbenz, “Ppt-gpu: Scalable gpu performance modeling,” _IEEE Computer Architecture Letter,_ , vol. 18, no. 1, pp. 55–58, Jan 2019 

- [40] Y. Arafa, G. Chennupati, A. Barai, A. A. Badawy, N. Santhi, and S. Eidenbenz, “Gpus cache performance estimation using reuse distance analysis,” in _IEEE 38th International Performance Computing and Communications Conference (IPCCC),_ 2019, pp. 1-8. 

- [41] Y. Arafa, A. A. Badawy, G. Chennupati, A. Barai, N. Santhi, and S. Eidenbenz, “Fast, accurate, and scalable memory modeling of gpgpus using reuse profiles,” in _Proceedings of the ACM International Conference on Supercomputing (ICS),_ 2020. 

- [42] G. Chennupati, N. Santhi, S. Eidenbenz, and S. Thulasidasan, “An analytical memory hierarchy model for performance prediction,” in _2017 Winter Simulation Conference (WSC),_ 2017, pp. 908–919. 

- [43] G. Chennupati, N. Santhi, and S. Eidenbenz, “Scalable performance prediction of codes with memory hierarchy and pipelines,” in _Proceedings of the 2019 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation,_ ser. SIGSIM-PADS, 2019, pp. 13–24. 

- [44] G. Chennupati, N. Santhi, S. Eidenbenz, R. J. Zerr, M. Rosa, R. J. Zamora, E. J. Park, B. T. Nadiga, J. Liu, K. Ahmed, and M. A. Obaida, “Performance prediction toolkit, version 00,” Sept. 2017. 

