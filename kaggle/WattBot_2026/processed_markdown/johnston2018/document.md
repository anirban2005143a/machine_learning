# **Dwarfs on Accelerators: Enhancing OpenCL Benchmarking for Heterogeneous Computing Architectures** 

Beau Johnston 

Research School of Computer Science 

Australian National University 

Canberra, Australia 

beau.johnston@anu.edu.au 

## **ABSTRACT** 

For reasons of both performance and energy efficiency, high performance computing (HPC) hardware is becoming increasingly heterogeneous. Te OpenCL framework supports portable programming across a wide range of computing devices and is gaining influence in programming next-generation accelerators. To characterize the performance of these devices across a range of applications requires a diverse, portable and configurable benchmark suite, and OpenCL is an atractive programming model for this purpose. 

We present an extended and enhanced version of the OpenDwarfs OpenCL benchmark suite, with a strong focus placed on the robustness of applications, curation of additional benchmarks with an increased emphasis on correctness of results and choice of problem size. Preliminary results and analysis are reported for eight benchmark codes on a diverse set of architectures – three Intel CPUs, five Nvidia GPUs, six AMD GPUs and a Xeon Phi. 

## **1 INTRODUCTION** 

High performance computing (HPC) hardware is becoming increasingly heterogeneous. A major motivation for this is to reduce energy use; indeed, without significant improvements in energy efficiency, the cost of exascale computing will be prohibitive. From June 2016 to June 2017, the average energy efficiency of the top 10 of the Green500 supercomputers rose by 2.3x, from 4.8 to 11.1 gigaflops per wat.[10] For many systems, this was made possible by highly energy-efficient Nvidia Tesla P100 GPUs. In addition to GPUs, future HPC architectures are also likely to include nodes with FPGA, DSP, ASIC and MIC components. A single node may be heterogeneous, containing multiple different computing devices; moreover, a HPC system may offer nodes of different types. For example, the Cori system at Lawrence Berkeley National Laboratory comprises 2,388 Cray XC40 nodes with Intel Haswell CPUs, and 9,688 Intel Xeon Phi nodes [7]. Te Summit supercomputer at Oak Ridge National Laboratory is based on the IBM Power9 CPU, which includes both NVLINK [22], a high bandwidth interconnect between Nvidia GPUs; and CAPI, an interconnect to support FPGAs and other accelerators. [23] Promising next generation architectures include Fujitsu’s Post-K [21], and Cray’s CS-400, which forms the platform for the Isambard supercomputer [9]. Both architectures use ARM cores alongside other conventional accelerators, with several Intel Xeon Phi and Nvidia P100 GPUs per node. 

Given this heterogeneity of hardware and the wide diversity of scientific application codes, workload characterization, performance prediction and scheduling are all becoming more challenging. 

Josh Milthorpe 

Research School of Computer Science 

Australian National University 

Canberra, Australia 

josh.milthorpe@anu.edu.au 

To evaluate different approaches requires a representative benchmark suite which is portable to a wide variety of devices. We focus on the OpenCL programming model as it is supported on a wide range of systems including CPU, GPU and FPGA devices. While it is possible to write application code directly in OpenCL, it may also be used as a base to implement higher-level programming models. Tis technique was shown by Mitra et al. [20] where an OpenMP runtime was implemented over an OpenCL framework for Texas Instruments Keystone II DSP architecture. Having a common backend in the form of OpenCL allows a direct comparison of identical code across diverse architectures. 

In this paper, we present an extended version of the OpenDwarfs benchmark suite, a set of OpenCL benchmarks for heterogeneous computing platforms.[15] We added new benchmarks to improve the diversity of the suite, and made a number of modifications aimed at improving the reproducibility and interpretability of results, portability between devices, and flexibility of configuration including problem sizes. We report preliminary results for a subset of the enhanced OpenDwarfs suite on a range of platforms including CPU, GPU and MIC devices. 

## **2 ENHANCING THE OPENDWARFS BENCHMARK SUITE** 

Te OpenDwarfs benchmark suite comprises a variety of OpenCL codes, classified according to paterns of computation and communication known as the 13 Berkeley Dwarfs.[1] Te original suite focused on collecting representative benchmarks for scientific applications, with a thorough diversity analysis to justify the addition of each benchmark to the corresponding suite. We aim to extend these efforts to achieve a full representation of each dwarf, both by integrating other benchmark suites and adding custom kernels. 

Marjanovic et al´ . [18] argue that the selection of problem size for HPC benchmarking critically affects which hardware properties are relevant. We have observed this to be true across a wide range of accelerators, therefore we have enhanced the OpenDwarfs benchmark suite to support running different problem sizes for each benchmark. To improve reproducibility of results, we also modified each benchmark to execute in a loop for a minimum of two seconds, to ensure that sampling of execution time and performance counters was not significantly affected by operating system noise. 

For the Spectral Methods dwarf, the original OpenDwarfs version of the FFT benchmark was complex, with several code paths that were not executed for the default problem size, and returned incorrect results or failures on some combinations of platforms 



and problem sizes we tested. We replaced it with a simpler highperformance FFT benchmark created by Eric Bainville [3], which worked correctly in all our tests. We have also added a 2-D discrete wavelet transform from the Rodinia suite [5] (with modifications to improve portability), and we plan to add a continuous wavelet transform code. 

To understand benchmark performance, it is useful to be able to collect hardware performance counters associated with each timing segment. LibSciBench is a performance measurement tool which allows high precision timing events to be collected for statistical analysis [12]. It offers a high resolution timer in order to measure short running kernel codes, reported with one cycle resolution and roughly 6 ns of overhead. We used LibSciBench to record timings in conjunction with hardware events, which it collects via PAPI [24] counters. We modified the applications in the OpenDwarfs benchmark suite to insert library calls to LibSciBench to record timings and PAPI events for the three main components of application time: kernel execution, host setup and memory transfer operations. Trough PAPI modules such as Intel’s Running Average Power Limit (RAPL) and Nvidia Management Library (NVML), LibSciBench also supports energy measurements, for which we report preliminary results in this paper. 

Additionally, our enhanced OpenDwarfs benchmark suite aims to cover a wider range of application paterns by focusing exclusively on OpenCL using higher-level benchmarks. 

Barnes et al. [4] collected a representative set of applications from the current NERSC workload to guide optimization for Knights Landing in the Cori supercomputer. As it is not always feasible to perform such a detailed performance study of the capabilities of different computational devices for particular applications, the benchmarks described in this paper may give a rough understanding of device performance and limitations. 

## **4 EXPERIMENTAL SETUP** 

## **4.1 Hardware** 

Te experiments were conducted on a varied set of 15 hardware platforms: three Intel CPU architectures, five Nvidia GPUs, six AMD GPUs, and one MIC (Intel Knights Landing Xeon Phi). Key characteristics of the test platforms are presented in Table 1. Te L1 cache size should be read as having both an instruction size cache and a data cache size of equal values as those displayed. For Nvidia GPUs, the L2 cache size reported is the size L2 cache per SM multiplied by the number of SMs. For the Intel CPUs, hyper-threading was enabled and the frequency governor was set to ‘performance’. 

## **4.2 Sofware** 

## **3 RELATED WORK** 

Te NAS parallel benchmarks [2] follow a ‘pencil-and-paper’ approach, specifying the computational problem but leaving implementation choices such as language, data structures and algorithms to the user. Te benchmarks include varied kernels and applications which allow a nuanced evaluation of a complete HPC system, however, the unconstrained approach does not readily support direct performance comparison between different hardware accelerators using a single set of codes. 

Martineau et al. [19] collected a suite of benchmarks and three mini-apps to evaluate Clang OpenMP 4.5 support for Nvidia GPUs. Teir focus was on comparison with CUDA; OpenCL was not considered. 

Te Scalable Heterogeneous Computing benchmark suite (SHOC) [16], unlike OpenDwarfs and Rodinia, supports multiple nodes using MPI for distributed parallelism. SHOC supports multiple programming models including OpenCL, CUDA and OpenACC, with benchmarks ranging from targeted tests of particular low-level hardware features to a handful of application kernels. Sun et al. [28] propose Hetero-Mark, a Benchmark Suite for CPU-GPU Collaborative Computing, which has five benchmark applications each implemented in HCC – which compiles to OpenCL, HIP – for a CUDA and Radeon Open Compute back-end, and a CUDA version. Meanwhile, Chai by Gomez-Luna et al. [´ 11], offers 15 applications in 7 different implementations with the focus on supporting integrated architectures. 

Tese benchmark suites focus on comparison between languages and environments; whereas our work focuses on benchmarking for device specific performance limitations, for example, by examining the problem sizes where these limitations occur – this is largely ignored by benchmarking suites with fixed problem sizes. 

OpenCL version 1.2 was used for all experiments. On the CPUs we used the Intel OpenCL driver version 6.3, provided in 16.1.1 and the 2016-R3 opencl-sdk release. On the Nvidia GPUs we used the Nvidia OpenCL driver version 375.66, provided as part of CUDA 8.0.61, AMD GPUs used the OpenCL driver version provided in the amdappsdk v3.0. 

Te Knights Landing (KNL) architecture used the same OpenCL driver as the Intel CPU platforms, however, the 2018-R1 release of the Intel compiler was required to compile for the architecture natively on the host. Additionally, due to Intel removing support for OpenCL on the KNL architecture, some additional compiler flags were required. Unfortunately, as Intel has removed support for AVX2 vectorization (using the ‘-xMIC-AVX512’ flag), vector instructions use only 256-bit registers instead of the wider 512-bit registers available on KNL. Tis means that floating-point performance on KNL is limited to half the theoretical peak. 

GCC version 5.4.0 with glibc 2.23 was used for the Skylake i7 and GTX 1080, GCC version 4.8.5 with glibc 2.23 was used on the remaining platforms. OS Ubuntu Linux 16.04.4 with kernel version 4.4.0 was used for the Skylake CPU and GTX 1080 GPU, Red Hat 4.8.5-11 with kernel version 3.10.0 was used on the other platforms. 

As OpenDwarfs has no stable release version, it was extended from the last commit by the maintainer on 26 Feb 2016. [25] LibSciBench version 0.2.2 was used for all performance measurements. 

## **4.3 Measurements** 

We measured execution time and energy for individual OpenCL kernels within each benchmark. Each benchmark run executed the application in a loop until at least two seconds had elapsed, and the mean execution time for each kernel was recorded. Each benchmark was run 50 times for each problem size (see §4.4) for 

2 



**Table 1: Hardware** 

|Name|Vendor|Type|Series|Core<br>Count|Clock Frequency<br>(MHz)<br>(min/max/turbo)|Cache (KiB)<br>(L1/L2/L3)|TDP<br>(W)|Launch<br>Date|
|---|---|---|---|---|---|---|---|---|
|Xeon E5-2697 v2|Intel|CPU|Ivy Bridge|24∗|1200/2700/3500|32/256/30720|130|Q3 2013|
|i7-6700K|Intel|CPU|Skylake|8∗|800/4000/4300|32/256/8192|91|Q3 2015|
|i5-3550|Intel|CPU|Ivy Bridge|4∗|1600/3380/3700|32/256/6144|77|Q2 2012|
|Titan X|Nvidia|GPU|Pascal|3584†|1417/1531/–|48/2048/–|250|Q3 2016|
|GTX 1080|Nvidia|GPU|Pascal|2560†|1607/1733/–|48/2048/–|180|Q2 2016|
|GTX 1080 Ti|Nvidia|GPU|Pascal|3584†|1480/1582/–|48/2048/–|250|Q1 2017|
|K20m|Nvidia|GPU|Kepler|2496†|706/–/–|64/1536/–|225|Q4 2012|
|K40m|Nvidia|GPU|Kepler|2880†|745/875/–|64/1536/–|235|Q4 2013|
|FirePro S9150|AMD|GPU|Hawaii|2816∥|900/–/–|16/1024/–|235|Q3 2014|
|HD 7970|AMD|GPU|Tahiti|2048∥|925/1010/–|16/768/–|250|Q4 2011|
|R9 290X|AMD|GPU|Hawaii|2816∥|1000/–/–|16/1024/–|250|Q3 2014|
|R9 295x2|AMD|GPU|Hawaii|5632∥|1018/–/–|16/1024/–|500|Q2 2014|
|R9 Fury X|AMD|GPU|Fuji|4096∥|1050/–/–|16/2048/–|273|Q2 2015|
|RX 480|AMD|GPU|Polaris|4096∥|1120/1266/–|16/2048/–|150|Q2 2016|
|Xeon Phi 7210|Intel|MIC|KNL|256‡|1300/1500/–|32/1024/–|215|Q2 2016|



∗ HyperTreaded cores 

† CUDA cores 

- ∥ Stream processors 

‡ Each physical core has 4 hardware threads per core, thus 64 cores 

both execution time and energy measurements. A sample size of 50 per group – for each combination of benchmark and problem size – was used to ensure that sufficient statistical power _β_ = 0.8 would be available to detect a significant difference in means on the scale of half standard deviation of separation. Tis sample size was computed using the t-test power calculation over a normal distribution. 

To help understand the timings, the following hardware counters were also collected: 

- total instructions and IPC (Instructions Per Cycle); 

- L1 and L2 data cache misses; 

- total L3 cache events in the form of request rate (requests / instructions), miss rate (misses / instructions), and miss ratio (misses/requests); 

- data TLB (Translation Look-aside Buffer) miss rate (misses 

- / instructions); and 

- branch instructions and branch mispredictions. 

For each benchmark we also measured memory transfer times between host and device, however, only the kernel execution times and energies are presented here. 

Energy measurements were taken on Intel platforms using the RAPL PAPI module, and on Nvidia GPUs using the NVML PAPI module. 

## **4.4 Setting Sizes** 

For each benchmark, four different problem sizes were selected, namely **tiny** , **small** , **medium** and **large** . Tese problem sizes are 

based on the memory hierarchy of the Skylake CPU. Specifically, **tiny** should just fit within L1 cache, on the Skylake this corresponds to 32 KiB of data cache, **small** should fit within the 256 KiB L2 data cache, **medium** should fit within 8192 KiB of the L3 cache, and **large** must be much larger than 8192 KiB to avoid caching and operate out of main memory. 

Te memory footprint was verified for each benchmark by printing the sum of the size of all memory allocated on the device. 

For this study, problem sizes were not customized to the memory hierarchy of each platform, since the CPU is the most sensitive to cache performance. Also, note for these CPU systems the L1 and L2 cache sizes are identical, and since we ensure that **large** is at least 4× larger than L3 cache, we are guaranteed to have last-level cache misses for the **large** problem size. 

Caching performance was measured using PAPI counters. On the Skylake L1 and L2 data cache miss rates were counted using the PAPI ~~L~~ 1 ~~D~~ CM and PAPI ~~L~~ 2 ~~D~~ CM counters. For L3 miss events, only the total cache counter event (PAPI ~~L~~ 3 ~~T~~ CM) was available. Te final values presented as miss results are presented as a percentage, and were determined using the number of misses counted divided by the total instructions (PAPI ~~T~~ OT ~~I~~ NS). 

Te methodology to determine the appropriate size parameters is demonstrated on the k-means benchmark. 

_4.4.1 kmeans._ K-means is an iterative algorithm which groups a set of points into clusters, such that each point is closer to the centroid of its assigned cluster than to the centroid of any other cluster. Each step of the algorithm assigns each point to the cluster 

3 



with the closest centroid, then relocates each cluster centroid to the mean of all points within the cluster. Execution terminates when no clusters change size between iterations. Starting positions for the centroids are determined randomly. Te OpenDwarfs benchmark previously required the object features to be read from a previously generated file. We extended the benchmark to support generation of a random distribution of points. Tis was done to more fairly evaluate cache performance, since repeated runs of clustering on the same feature space (loaded from file) would deterministically generate similar caching behavior. For all problem sizes, the number of clusters is fixed at 5. 

Given a fixed number of clusters, the parameters that may be used to select a problem size are the number of points _Pn_ , and the dimensionality or number of features per point _Fn_ . In the kernel for k-means there are three large one-dimensional arrays passed to the device, namely **feature** , **cluster** and **membership** . In the **feature** array which stores the unclustered feature space, each feature is represented by a 32-bit floating-point number, so the entire array is of size _Pn_ × _Fn_ × sizeof (float). **cluster** is the working and output array to store the intermediately clustered points, it is of size _Cn_ × _Fn_ × sizeof (float), where _Cn_ is the number of clusters. **membership** is an array indicating whether each point has changed to a new cluster in each iteration of the algorithm, it is of size _Pn_ × sizeof (int), where sizeof (int) is the number of bytes to represent an integer value. Tereby the working kernel memory, in KiB, is: 



size (feature) + size (membership) + size (cluster)
1024
(1)

Using this equation, we can determine the largest problem size that will fit in each level of cache. Te tiny problem size is defined to have 256 points and 30 features; from Equation 1 the total size of the main arrays is 31.5 KiB, slightly smaller than the 32 KiB L1 cache. Te number of points is increased for each larger problem size to ensure that the main arrays fit within the lower levels of the cache hierarchy, measuring the total execution time and respective caching events. Te **tiny** , **small** and **medium** problem sizes in the first row of Table 2 correspond to L1, L2 and L3 cache respectively. Te **large** problem size is at least four times the size of the last-level cache – in the case of the Skylake, at least 32 MiB – to ensure that data are transferred between main memory and cache. 

For brevity, cache miss results are not presented in this paper but were used to verify the selection of suitable problem sizes for each benchmark. Te procedure to select problem size parameters is specific to each benchmark, but follows a similar approach to k-means. 

_4.4.2 lud, fft, srad, crc, nw._ Te LU-Decomposition lud, Fast Fourier Transform fft, Speckle Reducing Anisotropic Diffusion srad, Cyclic Redundancy Check crc and Needleman-Wunsch nw benchmarks did not require additional data sets. Where necessary these benchmarks were modified to generate the correct solution and run on modern architectures. Correctness was examined either by directly comparing outputs against a serial implementation of the codes (where one was available), or by adding utilities to compare norms between the experimental outputs. 

_4.4.3 dwt._ Two-Dimensional Discrete Wavelet Transform is commonly used in image compression. It has been extended to support loading of Portable PixMap (.ppm) and Portable GrayMap (.pgm) image format, and storing Portable GrayMap images of the resulting DWT coefficients in a visual tiled fashion. Te input image dataset for various problem sizes was generated by using the resize capabilities of the ImageMagick application. Te original gum leaf image is the large sample size has the ratio of 3648 × 2736 pixels and was down-sampled to 80 × 60. 

_4.4.4 gem, nqueens, hmm._ For three of the benchmarks, we were unable to generate different problem sizes to properly exercise the memory hierarchy. 

Gemnoui gem is an n-body-method based benchmark which computes electrostatic potential of biomolecular structures. Determining suitable problem sizes was performed by initially browsing the National Center for Biotechnology Information’s Molecular Modeling Database (MMDB) [17] and inspecting the corresponding Protein Data Bank format (pdb) files. Molecules were then selected based on complexity, since the greater the complexity the greater the number of atoms required for the benchmark and thus the larger the memory footprint. **tiny** used the Prion Peptide 4TUT [29] and was the simplest structure, consisting of a single protein (1 molecule), it had the device side memory usage of 31.3 KiB which should fit in the L1 cache (32 KiB) on the Skylake processor. **small** used a Leukocyte Receptor 2D3V [27] also consisting of 1 protein molecule, with an associated memory footprint of 252KiB. **medium** used the nucleosome dataset originally provided in the OpenDwarfs benchmark suite, using 7498 KiB of device-side memory. **large** used an X-Ray Structure of a Nucleosome Core Particle [6], consisting of 8 protein, 2 nucleotide, and 18 chemical molecules, and requiring 10 970.2 KiB of memory when executed by gem. Each pdb file was converted to the pqr atomic particle charge and radius format using the pdb2pqr [8] tool. Generation of the solvent excluded molecular surface used the tool msms [26]. Unfortunately, the molecules used for the **medium** and **large** problem sizes contain uninitialized values only noticed on CPU architectures and as such further work is required to ensure correctness for multiple problem sizes. Te datasets used for gem and all other benchmarks can be found in this paper’s associated GitHub repository [13]. 

Te nqueens benchmark is a backtrack/branch-and-bound code which finds valid placements of queens on a chessboard of size n×n, where each queen cannot be atacked by another. For this code, memory footprint scales very slowly with increasing number of queens, relative to the computational cost. Tus it is significantly compute-bound and only one problem size is tested. 

Te Baum-Welch Algorithm Hidden Markov Model hmm benchmark represents the Graphical Models dwarf and did not require additional data sets, however validation of the correctness of results has not occurred apart from over the **tiny** problem size, as such, it is the only size examined in the evaluation. 

_4.4.5 Summary of Benchmark Setings._ Te problem size parameters for all benchmarks are presented in Table 2. 

Each **Device** can be selected in a uniform way between applications using the same notation, on this system **Device** comprises of -p 1 -d 0 -t 0 for the Intel Skylake CPU, where p and d are the integer identifier of the platform and device to respectively use, 

4 



**Table 2: OpenDwarfs workload scale parameters** Φ 

|**Benchmark**|**tiny**|**small**|**medium**|**large**|
|---|---|---|---|---|
|kmeans|256|2048|65600|131072|
|lud|80|240|1440|4096|
|csr|736|2416|14336|16384|
|fft|2048|16384|524288|2097152|
|dwt|72x54|200x150|1152x864|3648x2736|
|srad|80,16|128,80|1024,336|2048,1024|
|crc|2000|16000|524000|4194304|
|nw|48|176|1008|4096|
|gem|4TUT|2D3V|nucleosome|1KX5|
|nqueens|18|–|–|–|
|hmm|8,1|900,1|1012,1024|2048,2048|



**Table 3: Program Arguments** 

|**Benchmark**|**Arguments**|
|---|---|
|kmeans|-g -f 26 -pΦ|
|lud|-sΦ|
|csr†|-iΨ|
||Ψ=createcsr -nΦ-d 5000△|
|fft|Φ|
|dwt|-l 3 Φ-gum.ppm|
|srad|Φ1 Φ2 0 127 0 127 0.5 1|
|crc|-i 1000<br>~~Φ~~.txt|
|nw|Φ10|
|gem|Φ80 1 0|
|n-queens|Φ|
|hmm|-n Φ1-s Φ2-v s|



- Te -d 5000 indicates density of the matrix in 

- this instance 0.5% dense (or 99.5% sparse). 

- Te csr benchmark loads a file generated by createcsr according to the workload size parameter Φ; this file is represented by Ψ. 

and -p 1 -d 0 -t 1 for the Nvidia GeForce GTX 1080 GPU. Each application is run as **Benchmark Device** -- **Arguments** , where **Arguments** is taken from Table 3 at the selected scale of Φ. For reproducibility the entire set of Python scripts with all problem sizes is available in a GitHub repository [13]. Where Φ is substituted as the argument for each benchmark, it is taken as the respective scale from Table 2 and is inserted into Table 3. 

## **5 RESULTS** 

Te primary purpose of including these time results is to demonstrate the benefits of the extensions made to the OpenDwarfs Benchmark suite. Te use of LibSciBench allowed high resolution timing measurements over multiple code regions. To demonstrate the portability of the Extended OpenDwarfs benchmark suite, we 

present results from 11 varied benchmarks running on 15 different devices representing four distinct classes of accelerator. For 12 of the benchmarks, we measured multiple problem sizes and observed distinctly different scaling paterns between devices. Tis underscores the importance of allowing a choice of problem size in a benchmarking suite. 

## **5.1 Time** 

We first present execution time measurements for each benchmark, starting with the Cyclic Redundancy Check crc benchmark which represents the Combinational Logic dwarf. 

Figure 1 shows the execution times for the crc benchmark over 50 iterations on each of the target architectures, including the KNL. Te results are colored according to accelerator type: red for CPU devices, green for consumer GPUs, blue for HPC GPUs, and purple for the KNL MIC. Execution times for crc are lowest on CPU-type architectures, probably due to the low floating-point intensity of the CRC computation [14, Ch. 6]. Excluding crc, all the other benchmarks perform best on GPU type accelerators; furthermore, the performance on the KNL is poor due to the lack of support for wide vector registers in Intel’s OpenCL SDK. We therefore omit results for KNL for the remaining benchmarks. 

Figures 2 and 3 shows the distribution of kernel execution times for the remaining benchmarks. Some benchmarks execute more than one kernel on the accelerator device; the reported iteration time is the sum of all compute time spent on the accelerator for all kernels. Each benchmark corresponds to a particular dwarf: Figure 2a (kmeans) represents the MapReduce dwarf, Figure 2b (lud) represents the Dense Linear Algebra dwarf, Figure 2c (csr) represents Sparse Linear Algebra, Figure 2d (dwt) and Figure 2e (fft) represent Spectral Methods, Figure 3a (srad) represents the Structured Grid dwarf and Figure 3b (nw) represents Dynamic Programming. 

Finally, Figure 4 presents results for the three applications with restricted problem sizes and only one problem size is shown. Te N-body Methods dwarf is represented by (gem) and the results are shown in Figure 4a, the Backtrack & Branch and Bound dwarf is represented by the (nqueens) application in Figure 4b and (hmm) results in Figure 4c represent the Graphical Models dwarf. 

Examining the transition from tiny to large problem sizes (from lef to right) in Figure 3a shows the performance gap between CPU and GPU architectures widening for srad – indicating codes representative of structured grid dwarfs are well suited to GPUs. 

In contrast, Figure 3b shows Dynamic Programming problems have performance results tied to micro-architecture or OpenCL runtime support and can not be explained solely by considering accelerator type. For instance, the Intel CPUs and NVIDIA GPUs perform comparably over all problem sizes, whereas all AMD GPUs exhibit worse performance as size increases. 

For most benchmarks, the coefficient of variation in execution times is much greater for devices with a lower clock frequency, regardless of accelerator type. While execution time increases with problem size for all benchmarks and platforms, the modern GPUs (Titan X, GTX1080, GTX1080Ti, R9 Fury X and RX 480) performed relatively beter for large problem sizes, possibly due to their greater second-level cache size compared to the other platforms. A notable 

5 



![Figure](assets/figure_0001_page_0006.svg)**Figure 1: Kernel execution times for the crc benchmark on different hardware platforms** 

exception is k-means for which CPU execution times were comparable to GPU, which reflects the relatively low ratio of floating-point to memory operations in the benchmark. 

Generally, the HPC GPUs are older and were designed to alleviate global memory limitations amongst consumer GPUs of the time. (Global memory size is not listed in Table 1.) Despite their larger memory sizes, the clock speed of all HPC GPUs is slower than all evaluated consumer GPUs. While the HPC GPUs (devices 7-9, in blue) outperformed consumer GPUs of the same generation (devices 10-13, in green) for most benchmarks and problem sizes, they were always beaten by more modern GPUs. Tis is no surprise since all selected problem sizes fit within the global memory of all devices. 

A comparison between CPUs (devices 1-3, in red) indicates the importance of examining multiple problem sizes. Medium-sized problems were designed to fit within the L3 cache of the i7-6700K system, and this conveniently also fits within the L3 cache of the Xeon E5-2697 v2. However, the older i5-3550 CPU has a smaller L3 cache and exhibits worse performance when moving from small to medium problem sizes, and is shown in Figures 2b, 2d, 2e and 3a, 

Increasing problem size also hinders the performance in certain circumstances for GPU devices. For example, Figure 3b shows a widening performance gap over each increase in problem size between AMD GPUs and the other devices. 

Predicted application properties for the various Berkeley Dwarfs are evident in the measured runtime results. For example, Asanovic´ et al. [1] state that applications from the Spectral Methods dwarf is memory latency limited. If we examine dwt and fft – the applications which represent Spectral Methods – in Figure 2d and Figure 2e respectively, we see that for medium problem sizes the execution times match the higher memory latency of the L3 cache of CPU devices relative to the GPU counterparts. Te trend only increases with problem size: the large size shows the CPU devices frequently accessing main memory while the GPUs’ larger memory ensures a lower memory access latency. It is expected if had we extended this study to an even larger problem size that would not fit on GPU global memory, much higher performance penalties would be experienced over GPU devices, since the PCI-E interconnect has a higher latency than a memory access to main memory from the CPU systems. As a further example, Asanovic et al. [´ 1] 

state that the Structured Grid dwarf is memory bandwidth limited. Te Structured Grid dwarf is represented by the srad benchmark shown in Figure 3a. GPUs exhibit lower execution times than CPUs, which would be expected in a memory bandwidth-limited code as GPU devices offer higher bandwidth than a system interconnect. 

## **5.2 Energy** 

In addition to execution time, we are interested in differences in energy consumption between devices and applications. We measured the energy consumption of benchmark kernel execution on the Intel Skylake i7-6700k CPU and the Nvidia GTX1080 GPU, using PAPI modules for RAPL and NVML. Tese were the only devices examined since collection of PAPI energy measurements (with LibSciBench) requires superuser access, and these devices were the only accelerators available with this permission. Te distributions were collected by measuring solely the kernel execution over a distribution of 50 runs. RAPL CPU energy measurements were collected over all cores in package 0 rapl:::PP0 ~~E~~ NERGY:PACKAGE0. NVML GPU energy was collected using the power usage readings nvml:::GeForce ~~G~~ TX ~~1~~ 080:power for the device and presents the total power draw (+/-5 wats) for the entire card – memory and chip. Measurements results converted to energy J from their original resolution nJ and mW on the CPU and GPU respectively. 

From the time results presented in Section 5.1 we see the largest difference occurs between CPU and GPU type accelerators at the **large** problem size. Tus we expect that the **large** problem size will also show the largest difference in energy. 

Figures 5a and 5b show the kernel execution energy for several benchmarks for the **large** size. All results are presented in joules. Te box plots are coloured according to device: red for the Intel Skylake i7-6700k CPU and blue for the Nvidia GTX1080 GPU. Te logarithmic transformation has been applied to Figure 5b to emphasise the variation at smaller energy scales (< 1 J), which was necessary due to small execution times for some benchmarks. In future this will be addressed by balancing the amount of computation required for each benchmark, to standardize the magnitude of results. 

All the benchmarks use more energy on the CPU, with the exception of crc which as previously mentioned has low floating-point 

6 



![Figure](assets/figure_0002_page_0007.svg)**Figure 2: Benchmark kernel execution times on different hardware platforms** 



![Figure](assets/figure_0003_page_0008.svg)**Figure 4: Single problem sized benchmarks of kernel execution times on different hardware platforms** 

intensity and so is not able to make use of the GPU’s greater floatingpoint capability. Variance with respect to energy usage is larger on the CPU, which is consistent with the execution time results. 

## **6 CONCLUSIONS** 

We have performed essential curation of the OpenDwarfs benchmark suite. We improved coverage of spectral methods by adding a new Discrete Wavelet Transform benchmark, and replacing the previous inadequate fft benchmark. All benchmarks were enhanced to allow multiple problem sizes; in this paper we report results for 

four different problem sizes, selected according to the memory hierarchy of CPU systems as motivated by Marjanovic’s findings [´ 18]. Tese can now be easily adjusted for next generation accelerator systems using the methodology outlined in Section 4.4. 

We ran many of the benchmarks presented in the original OpenDwarfs [15] paper on current hardware. Tis was done for two reasons, firstly to investigate the original findings to the stateof-the-art systems and secondly to extend the usefulness of the benchmark suite. Re-examining the original codes on range of modern hardware showed limitations, such as the fixed problem sizes along with many platform-specific optimizations (such as local 

8 



![Figure](assets/figure_0004_page_0009.svg)**Figure 5: Benchmark kernel execution energy (large problem size) on Core i7-6700K and Nvidia GTX1080** 

work-group size). In the best case, such optimizations resulted in sub-optimal performance for newer systems (many problem sizes favored the original GPUs on which they were originally run). In the worst case, they resulted in failures when running on untested platforms or changed execution arguments. 

Finally a major contribution of this work was to integrate LibSciBench into the benchmark suite, which adds a high precision timing library and support for statistical analysis and visualization. Tis has allowed collection of PAPI, energy and high resolution (sub-microsecond) time measurements at all stages of each application, which has added value to the analysis of OpenCL program flow on each system, for example identifying overheads in kernel construction and buffer enqueuing. Te use of LibSciBench has also increased the reproducibility of timing data for both the current study and on new architectures in the future. 

## **7 FUTURE WORK** 

We plan to complete analysis of the remaining benchmarks in the suite for multiple problem sizes. In addition to comparing performance between devices, we would also like to develop some notion of ‘ideal’ performance for each combination of benchmark and device, which would guide efforts to improve performance portability. Additional architectures such as FPGA, DSP and Radeon Open Compute based APUs – which further breaks down the walls between the CPU and GPU – will be considered. 

Each OpenCL kernel presented in this paper has been inspected using the Architecture Independent Workload Characterization (AIWC). Analysis using AIWC helps understand how the structure 

of kernels contributes to the varying runtime characteristics between devices that are presented in this work, and will be published in the future. 

Certain configuration parameters for the benchmarks, e.g. local workgroup size, are amenable to auto-tuning. We plan to integrate auto-tuning into the benchmarking framework to provide confidence that the optimal parameters are used for each combination of code and accelerator. 

Te original goal of this research was to discover methods for choosing the best device for a particular computational task, for example to support scheduling decisions under time and/or energy constraints. Until now, we found the available OpenCL benchmark suites were not rich enough to adequately characterize performance across the diverse range of applications and computational devices of interest. Now that a flexible benchmark suite is in place and results can be generated quickly and reliably on a range of accelerators, we plan to use these benchmarks to evaluate scheduling approaches. 

## **ACKNOWLEDGEMENTS** 

We thank our colleagues at Te University of Bristol’s High Performance Computing Research group for the use of “Te Zoo” Research cluster for experimental evaluation. 

## **REFERENCES** 

> [1] Krste Asanovic, Ras Bodik, Bryan Christopher Catanzaro, Joseph James Gebis,´ Parry Husbands, Kurt Keutzer, David A Paterson, William Lester Plishker, John Shalf, Samuel Webb Williams, and Kathy Yelick. 2006. _Te landscape of parallel computing research: A view from Berkeley_ . Technical Report UCB/EECS-2006-183. EECS Department, University of California, Berkeley. 

9 



- [2] David H Bailey, Eric Barszcz, John T Barton, David S Browning, Robert L Carter, Leonardo Dagum, Rod A Fatoohi, Paul O Frederickson, Tomas A Lasinski, Rob S Schreiber, et al. 1991. Te NAS parallel benchmarks. _Internat. J. Supercomputing Applic._ 5, 3 (1991), 63–73. 

- [3] Eric Bainville. 2010. OpenCL Fast Fourier Transform. (2010). htp://www.bealto. com/gpu-f.html 

- [4] Taylor Barnes, Brandon Cook, Jack Deslippe, Douglas Doerfler, Brian Friesen, Yun He, Torsten Kurth, Tuomas Koskela, Mathieu Lobet, Tareq Malas, et al. 2016. Evaluating and optimizing the NERSC workload on Knights Landing. In _International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)_ . IEEE, 43–53. 

- [5] Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W Sheaffer, Sang-Ha Lee, and Kevin Skadron. 2009. Rodinia: A benchmark suite for heterogeneous computing. In _IEEE International Symposium on Workload Characterization (IISWC)_ . IEEE, 44–54. 

- [6] Curt A Davey, David F Sargent, Karolin Luger, Armin W Maeder, and Timothy J Richmond. 2002. Solvent mediated interactions in the structure of the nucleosome core particle at 1.9<sup>˚</sup> A resolution. _Journal of Molecular Biology_ 319, 5 (2002), 1097– 1113. 

- [7] Tina Declerck, Katie Antypas, Deborah Bard, Wahid Bhimji, Shane Canon, Shreyas Cholia, Helen Yun He, Douglas Jacobsen, and Nicholas J Wright Prabhat. 2016. Cori - A system to support data-intensive computing. _Proceedings of the Cray User Group_ (2016), 8. 

- [8] Todd J Dolinsky, Jens E Nielsen, J Andrew McCammon, and Nathan A Baker. 2004. PDB2PQR: an automated pipeline for the setup of Poisson–Boltzmann electrostatics calculations. _Nucleic Acids Research_ 32, suppl 2 (2004), W665– W667. 

- [9] Michael Feldman. 2017. Cray to Deliver ARM-Powered Supercomputer to UK Consortium. _TOP500 Supercomputer Sites_ (Jan 2017). htps://www.top500.org/ news/cray-to-deliver-arm-powered-supercomputer-to-uk-consortium/ 

nvlink-takes-gpu-acceleration-next-level/ 

   - [23] Timothy Morgan. 2017. Te Power9 Rollout Begins With Summit And Sierra Supercomputers. _Te Next Platform_ (Sep 2017). htps://www.nextplatform.com/ 2017/09/19/power9-rollout-begins-summit-sierra/ 

   - [24] Philip J Mucci, Shirley Browne, Christine Deane, and George Ho. 1999. PAPI: A portable interface to hardware performance counters. In _Proceedings of the Department of Defense HPCMP users group conference_ , Vol. 710. 

   - [25] OpenDwarfs 2017. OpenDwarfs (base version). htps://github.com/vtsynergy/ OpenDwarfs/commit/31c099aff5343e93ba9e8c3cd42bee5ec536aa93. (26 Feb 2017). 

   - [26] Michel F Sanner, Arthur J Olson, and Jean-Claude Spehner. 1996. Reduced surface: an efficient way to compute molecular surfaces. _Biopolymers_ 38, 3 (1996), 305–320. 

   - [27] Mitsunori Shiroishi, Mizuho Kajikawa, Kimiko Kuroki, Toyoyuki Ose, Daisuke Kohda, and Katsumi Maenaka. 2006. Crystal structure of the human monocyteactivating receptor, “Group 2” leukocyte Ig-like receptor A5 (LILRA5/LIR9/ILT11). _Journal of Biological Chemistry_ 281, 28 (2006), 19536–19544. 

   - [28] Yifan Sun, Xiang Gong, Kavyan Amir, Amir Kavyan Ziabari, Leiming Yu, Xiangyu Li, Saoni Mukherjee, Carter McCardwell, Alejandro Villegas, and David Kaeli. 2016. Hetero-Mark, a benchmark suite for CPU-GPU collaborative computing. In _IEEE International Symposium on Workload Characterization (IISWC)_ . 

   - [29] Lu Yu, Seung-Joo Lee, and Vivien C Yee. 2015. Crystal structures of polymorphic prion protein _β_ 1 peptides reveal variable steric zipper conformations. _Biochemistry_ 54, 23 (2015), 3640–3648. 

- [10] Michael Feldman. 2017. TOP500 Meanderings: Supercomputers Take Big Green Leap in 2017. _TOP500 Supercomputer Sites_ (Sep 2017). htps://www.top500.org/ news/top500-meanderings-supercomputers-take-big-green-leap-in-2017/ 

- [11] Juan Gomez-Luna,´ Izzat El Hajj, Victor Chang, Li-Wen Garcia-Flores, Simon Garcia de Gonzalo, Tomas Jablin, Antonio J Pena, and Wen-mei Hwu. 2017. Chai: Collaborative heterogeneous applications for integrated-architectures. In _IEEE International Symposium on Performance Analysis of Systems and Sofware (ISPASS)_ . IEEE. 

- [12] Torsten Hoefler and Roberto Belli. 2015. Scientific benchmarking of parallel computing systems: Twelve ways to tell the masses when reporting performance results. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ . ACM, 73. 

- [13] Beau Johnston. 2017. OpenDwarfs. htps://github.com/BeauJoh/OpenDwarfs. (2017). 

- [14] Anagha S Joshi. 2016. _A performance focused, development friendly and model aided parallelization strategy for scientific applications_ . Master’s thesis. Clemson University. 

- [15] Konstantinos Krommydas, Wu-Chun Feng, Christos D Antonopoulos, and Nikolaos Bellas. 2016. OpenDwarfs: Characterization of dwarf-based benchmarks on fixed and reconfigurable architectures. _Journal of Signal Processing Systems_ 85, 3 (2016), 373–392. 

- [16] M Graham Lopez, Jeffrey Young, Jeremy S Meredith, Philip C Roth, Mitchel Horton, and Jeffrey S Veter. 2015. Examining recent many-core architectures and programming models using SHOC. In _International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)_ . ACM, 3. 

- [17] Tomas Madej, Christopher J Lanczycki, Dachuan Zhang, Paul A Tiessen, Renata C Geer, Aron Marchler-Bauer, and Stephen H Bryant. 2013. MMDB and VAST+: tracking structural similarities between macromolecular complexes. _Nucleic Acids Research_ 42, D1 (2013), D297–D303. 

- [18] Vladimir Marjanovic, Jos´ e Gracia, and Colin W Glass. 2016.´ HPC benchmarking: problem size maters. In _International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)_ . IEEE, 1–10. 

- [19] Mat Martineau, Simon McIntosh-Smith, Carlo Bertolli, Arpith C Jacob, Samuel F Antao, Alexandre Eichenberger, Gheorghe-Teodor Bercea, Tong Chen, Tian Jin, Kevin O’Brien, et al. 2016. Performance analysis and optimization of Clang’s OpenMP 4.5 GPU support. In _International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)_ . IEEE, 54–64. 

- [20] Gaurav Mitra, Eric Stotzer, Ajay Jayaraj, and Alistair P Rendell. 2014. Implementation and optimization of the OpenMP accelerator model for the TI Keystone II architecture. In _International Workshop on OpenMP_ . Springer, 202–214. 

- [21] Timothy Morgan. 2016. Inside Japan’s Future Exascale ARM Supercomputer. (Jun 2016). htps://www.nextplatform.com/2016/06/23/ inside-japans-future-exaflops-arm-supercomputer/ 

- [22] Timothy Morgan. 2016. NVLink Takes GPU Acceleration To Te Next Level. _Te Next Platform_ (May 2016). htps://www.nextplatform.com/2016/05/04/ 

10 

