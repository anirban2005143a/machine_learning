# Benchmarking the Performance and Energy Efficiency of AI Accelerators for AI Training 

Yuxin Wang, Qiang Wang, Shaohuai Shi, Xin He, Zhenheng Tang, Kaiyong Zhao, and Xiaowen Chu<sup>_∗_</sup> 

High Performance Machine Learning Lab 

Department of Computer Science 

Hong Kong Baptist University 

## **ABSTRACT** 

Deep learning has become widely used in complex AI applications. Yet, training a deep neural network (DNNs) model requires a considerable amount of calculations, long running time, and much energy. Nowadays, many-core AI accelerators (e.g., GPUs and TPUs) are designed to improve the performance of AI training. However, processors from different vendors perform dissimilarly in terms of performance and energy consumption. To investigate the differences among several popular offthe-shelf processors (i.e., Intel CPU, NVIDIA GPU, AMD GPU, and Google TPU) in training DNNs, we carry out a comprehensive empirical study on the performance and energy efficiency of these processors<sup>2</sup> by benchmarking a representative set of deep learning workloads, including computationintensive operations, classical convolutional neural networks (CNNs), recurrent neural networks (LSTM), Deep Speech 2, and Transformer. Different from the existing end-to-end benchmarks which only present the training time, We try to investigate the impact of hardware, vendor’s software library, and deep learning framework on the performance and energy consumption of AI training. Our evaluation methods and results not only provide an informative guide for end users to select proper AI accelerators, but also expose some opportunities for the hardware vendors to improve their software library. 

**_K_ eywords** AI Accelerator, Deep Learning, CPU, GPU, TPU, Computation-intensive Operations, Convolution Neural Networks, Recurrent Neural Networks, Transformer, Deep Speech 2 

## **1 Introduction** 

Recent years have witnessed the fast development of deep neural networks (DNNs) [1] in many AI applications, such as image recognition [2], object detection [3], speech to text tasks [4], etc. However, training DNN models requires a considerable amount of computational resources that usually involve Graphics Processing Units (GPUs) [1, 5]. 

Graphics Processing Units (GPUs) [6] serve as one of the most popular accelerators for efficient DNN training. Different from conventional CPUs, a GPU is typically equipped with thousands of cores and large memory bandwidth [6], which significantly accelerates the training and reasoning speed of DNNs. Since 2016, the new generation of computing device - the TPU (Tensor Processing Units) [7], has been launched by Google with better performance than many other AI accelerators, followed by Cloud TPU v2 and Cloud TPU v3 [8] with improved performance and memory capacity. With its extraordinary parallel computing capability, the cloud service of TPU greatly facilitated research and applications of artificial intelligence. 

Meanwhile, the development of optimized software keeps pace with advancement of the hardware. For the CPU processors, there exist highly optimized libraries like MKL and MKLDNN [9]. For the NVIDIA GPUs, researchers and 

- _∗_ {yxwang, qiangwang, csshshi, csxinhe, zhtang, kyzhao, chxw}@comp.hkbu.edu.hk 

- 2The benchmark results include performance of CPU, GPUs, TPUs; and energy of GPUs. 



industry developed cuDNN [10], cuBLAS, and other CUDA based libraries, enabling peak performance of GPU cards for AI training. For the AMD GPUs, ROCm [11] is being actively developed for high performance deep learning. Also, for TPUs, TensorFlow [12] is highly optimized under a large development community. 

However, generations of AI accelerators from different vendors have diverse performance and energy consumption. For example, the performance of throughput could be different on GPUs from NVIDIA and AMD with similar raw capacity. In terms of performance, there exist some benchmarks including software comparison [13], hardware comparison [14, 15, 16, 17, 18] and a combination of software and hardware comparison [19][20] in training DNNs. In addition, vendors provide their own benchmarks to demonstrate the performance with their own highly optimized libraries or configurations; however, these results could be unfairly compared. 

For server deployment, it is vital to reduce energy consumption for DNN training in order to reduce the long-term electric bills. There exist several techniques to balance and improve the performance and energy efficiency, including characterization of performance and energy for traditional algorithms [21]; dynamic voltage and frequency scaling (DVFS) techniques [22, 23, 24]; job scheduling algorithms [25] for saving energy while preserving the computing efficiency of tasks. 

In summary, existing benchmarks consider either only performance or energy for particular accelerators and algorithms. Furthermore, there is little study on AMD GPUs, while AMD researchers have also developed a deep learning ecosystem ROCm for users. An overview of recent AI benchmark comparison can be found in Table 1. In this paper, we make wide benchmarks on many popular AI accelerators, including the Intel CPU, NVIDIA GPUs, the AMD GPU, and Google TPUs in terms of performance and energy efficiency. Our work covers multiple workloads ranging from computation-intensive operations to real-world tasks, including image recognition, speech recognition, and translation tasks. To make the evaluation thorough, we first evaluate the performance on low-level operations on the above aforementioned accelerators, and then we evaluate the performance and energy consumption of the end-to-end DNN training from different AI areas, including CNNs [26], [2], [27], LSTM [28], Deep Speech 2 [4] and Transformer [29]. 

Our major findings are shown in Table 2. On the one hand, our evaluation results provide a guide for end-users on how to choose proper AI accelerators to train their own DNN models under different considerations. For example, end-users can compare the budgets of cloud-based GPUs and TPUs for a specific model, and choose a cheaper one to train the model. On the other hand, the problems revealed in the evaluation results could be helpful for hardware or software design for further optimization. For example, GPU library engineers can have an insight into the performance with the roofline model [30] and kernel-level optimization of better GPU utilization. The results of energy efficiency can redound to develop job scheduling algorithms for energy conservation [25][31], in which one should expect the task to be finished in expected time (related to performance) with less power (related to energy). 

Table 1: AI Benchmark Comparison 

|Training|Benchmarks|Our work|<br>MLPer|f<br>DAWNBenc|h<br>AIBench|Fathom|TBD|
|---|---|---|---|---|---|---|---|
||CPU|✓|✓|✓|✓|✓|✓|
|AI|NVIDIA GPU|✓|✓|✓|✓|✓|✓|
|Accelerators|AMD GPU|✓|x|x|x|x|x|
||TPU|✓|✓|✓|x|x|x|
||PERF|✓|✓|✓|✓|✓|✓|
|Metrics|NRG|✓|x|x|x|x|x|
||IC|✓|✓|✓|✓|✓|✓|
||SR|✓|✓|x|✓|✓|✓|
|Workloads|TR|✓|✓|x|✓|✓|✓|
||TNR|✓|✓|x|✓|x|✓|
||CIO|✓|x|x|✓|✓|x|
|Notes:|PERF: Perform<br>IC: Image Clasi<br>TR: Translatio<br>CIO: Computa|ance. NR<br>sification<br>n-Recurre<br>tion-inten|G: Ener<br>i; SR: Sp<br>nt; TNR<br>sive Op|gy.<br>i  eech Recogni<br>: Translation-<br>erations.|i   tion.<br>Non-recur|rent.||



The rest of this paper is organized as follows. Section 2 introduces some background knowledge related to DNNs, AI accelerators, as well as training pipeline and algorithms. Section 3 describes our experimental designs and setups, including hardware configurations and DNNs. Section 4 demonstrates our experimental results and analysis of AI accelerators with different training tasks. Related benchmarks are introduced in Section 5. We finally conclude the paper in Section 6. 

2 



Table 2: Summary of Main Findings about PERF (performance) and NRG (energy) 

|Section<br>AI|Accelerat|or<br>Metric<br>Main Findings|
|---|---|---|
|4.1.1|CPU|1.1 Powered by AVX optimization, the CPU utilization scales linearly by threads and FLOPs of computation-intensive operations.<br>1.2 Matrix multiplication is better optimized than 2d convolution with regards to the FLOPS utilization.|
|4.2.1||1.3 Theparallel computation intensityand executionpattern, peak FLOPS,and data-fetchingare three facts that bottleneck the training performance.<br>|
|||2.1 GPUs from NVIDIA and AMD perform the same the maximum utilization of peak FLOPS with high-throughput kernels.|
|4.1.2||2.2 There exists limitations on the utilization of Tensor Core. the GPU utilization with Tensor Core is 9% - 31% belows that of the FP32 core<br>under the tested computation intensity.|
||GPU|2.3 The model property (including size, FLOPs, kernel type, parameters, etc.) casts comprehensive influence on the effects of mini-batch size<br>during DNN training, determining the highest throughput.|
|4.2.2||PERF<br>2.4 The actual utilization of Tensor Core is low during end-to-end Training, ranging from 3% -9%. Compared with FP32, 2_×_is the maximum<br>performance improvement by Tensor Core on the CNN. Meanwhile, the benefits of Tensor Core on NLP models are small.<br>2.5 The Titan X(Pascal) prevails Radeon VII in training performance with inferior hardware configuration and superior software optimization.<br>The software supportingof NVIDIA GPUs alsoprevails.|
|4.1.2||3.1 The TPU v2 can reach peak FLOPS with the TensorFlow implementation of the maximum computation intensity in the test. Operations<br>with larger sizes benefits more frompulsation matrix calculation inside of the TPU.|
|4.2.4|TPU|3.2 From TPU v2 to TPU v3, the peak FLOPS updates 2.3_×_, whereas the memory bandwidth updates only 1.5_×_. The update ratios lead to an<br>unbalance in the performance improvement among DNNs with different parameters and FLOPs.<br>3.3 The benefit ofperformance byswtchingfrom V100 to TPU is 1.1_×_-1.7_×_on the non-recurrent translation model and 1.5_×_-2.7_×_on CNNs.|
|||4.1 Server level GPUs, including P100 and V100 demostrate lower energy consumption. Especially, the usage of Tensor Core make the<br>energy consumption lower for 2_×_. V100 has at most 5.2_×_lower energy consumption than other GPUs.|
|4.3|GPU|NRG<br>4.2 The energy consumption benefits from larger batch size in most cases, if only the performance has about the same benefits.<br>4.3 The power management of the AMD GPU is inferior to that of the NVIDIA GPU, especially for recurrent models.<br>4.4 AmongNVIDIA GPUs,Titan X(Pascal)has the largest energyconsumption for all end-to-end DNNs.|



## **2 Preliminaries** 

### **2.1 Deep Models** 

In different areas of AI applications, there exist various types of deep architectures that achieve state-of-the-art results. In image classification and object detection tasks, convolutional neural networks (CNNs) are the main architectures to extract the image features automatically, among which VGG [26], ResNet [2] and Inception [27] architectures are widely used. These CNNs also achieve excellent results in image classification and object detection, tasks of popular ImageNet challenge [32]. In the area of natural language processing(NLP), recurrent neural network (RNN) was one of the successful models with a long developing history, especially LSTM [33]. In recent years, Deep Speech 2 [4] was proposed with state-of-the-art results on speech recognition tasks, and the attention-based model - Transformer [29] has achieved outstanding scores in machine translation tasks. Fig. 1 shows some of the DNN architectures. 

### **2.2 AI Accelerators** 

There are many newly developed AI accelerators. In this paper, we focus on the widely used processors, including CPU, GPU, and TPU. We will investigate FPGAs in the future. 

**CPU** CPUs are traditional processors that used in computers, while it was not good at doing highly parallel and computing-intensive tasks. In the era of deep learning, the main CPU vendor designs its many-core CPUs for these kinds of tasks. For example, the Intel Xeon processor [34] is a powerful CPU with high computing FLOPS<sup>3</sup> among Intel CPUs. among Intel CPUs. The scalable processor was reported that it outperforms NVIDIA GPU in deep learning inference on the ResNet-50 model<sup>4</sup> . 

**NVIDIA and AMD GPUs** GPUs are designed for highly parallel applications in terms of the number of computing cores and the memory access speed. The peak FLOPS has increased rapidly in the last ten years. In Table 2, we list the configurations of four recent GPUs, three of which from NVIDIA (Tesla V100, P100, and Titan X(Pascal)) and one from AMD (Radeon VII). It can be seen that the peak FP32 computing FLOPS is more than 10 TFLOPs, which is around 5 _×_ higher than CPUs. 

**Google TPUs** Tensor Processing Units (TPUs) are Google’s custom-designed machine learning application-specific integrated circuits (ASICs). Each TPU device has 4 chips and each consists of 2 cores, so a TPU device contains 8 cores. Each core has scalar, vector and matrix units (MXU) and is connected with the on-chip high bandwidth memory (HBM). As shown in Fig. 2, there are two types of TPUs: TPU v2 and TPU v3. For TPU v2, the amount of HBM of each core is 8 GB. Especially, one MXU is allocated to a core. While for TPU v3, each core has two MXUs and is connected with 16 GB of HBM. TPUs support the bfloat16 format which has a wider range of values than float16 with 

> 3In this paper, FLOPS is a performance metric indicated by <u>FLOating Point Operations per Second, while FLOPs is a workload</u> metric that represent the total number of FLOating Point Operations. 

> 4 `https://intel.ly/2k4Bxh2` 

3 



![Figure](assets/figure_0001_page_0004.svg)Figure 1: Different Architecture of DNNs. 

Table 3: The Technical Specifications of GPUs 

|Product Name|Tesla V100|Tesla P100|Titan X(Pascal)|Radeon VII|
|---|---|---|---|---|
|GPU|GV100|GP100|GP108|Vega 20|
|GPU Cores|5120|3584|3584|3840|
|Tensor Cores|640|-|-|-|
|Core Clock|1245 MHz|1190 MHz|1417MHz|1400 MHz|
|Boost Clock|1380MHz|1329 MHz|1531 MHz|1750 MHz|
|MemoryClock|877 MHz|715 MHz|1251MHz|1000 MHz|
|MemoryBus Width|4096 bit|4096 bit|384 bit|4096 bit|
|MemoryBandwidth|897.0 GB/s|732.2 GB/s|480.4 GB/s|1 TB/s|
|MemoryType|HBM2|HBM2|GDDR5X|HBM2|
|FP16 Computing|28.26 TFLOPS|19.05 TFLOPS|-|26.88 TFLOPS|
|FP32 Computing|14.13 TFLOPS|9.526 TFLOPS|10.97 TFLOPS|13.44 TFLOPS|
|TDP|250w|250w|250w|295w|



the same 16-bit storage. TPU v2 with 8 cores (TPU v2-8) and TPU v3 with 8 cores (TPU v3-8) have peak bfloat16 computing capacity of 180 Tera bfloat16 per second and 420 Tera bfloat16 per second respectively. Additionally, TPU v2 Pod is assembled by 64 TPU v2 devices, containing 512 TPU v2 cores. TPU v3 Pod provides a maximum of 256 TPU v3 devices and consists of a total 2048 TPU v3 cores. 

![Figure](assets/figure_0002_page_0004.svg)Figure 2: The structures of TPU v2 (left) and TPU v3 (right). 

**bfloat16 and float16** The MXU in each TPU core is used to execute 16K multiply-accumulate operations in each cycle. Besides, MXU supports mixed precision training, i.e. its input and output are 32-bit floating point values and it 

4 



can use bfloat16 for activation and gradients. Compared with IEEE half-precision floating point (fp16), bfloat16 has a wider range of values because it has one sign bit, eight exponent bits, and seven mantissa bits plus one implicit mantissa bit, as shown in Fig. 3. Using bfloat16 can help reduce the size of data in memory, making larger models available for the same size of memory, while ensuring no degradation of converged accuracy. 

![Figure](assets/figure_0003_page_0005.svg)Figure 3: The comparison between bfloat16 and float16. 

### **2.3 Training Pipeline** 

**Mini-batch SGD** Mini-batch SGD [35] is a derivative algorithm of the Stochastic Gradient Descent method (SGD) , which divides the entire data set into multiple subsets, and iteratively update the model parameters according to the first-order gradients at current mini-batch of data. The training process during a single iteration can be divided into the following steps. As is shown in Fig. 4, a single iteration starts with the process of reading data from the computer’s disk to the CPU’s memory, and it ends with updates of parameters. The training process is to repeat the iteration until some terminating criteria. We generally use the average iteration time to measure the performance of training on particular software and hardware. 

**Mixed Precision Training** The mixed precision<sup>5</sup> training technique is a very successful training algorithm that uses only low-bit floating points to do the computation of forward and backward during training such that the hardware resource can be better utilized. Typically, in mixed precision, FP32 master weights and loss scaling are adopted to avoid the instability that FP16/bfloat16 precision might trigger. The training process is also shown in Fig. 4. 

![Figure](assets/figure_0004_page_0005.svg)Figure 4: The training pipeline of deep neural networks. Note that FP16 will be replaced by bfloat16 when training DNNs on TPUs. 

5The mixed precision mainly exploits FP16(on Tensor Cores)/bfloat16(on TPU cores) as computation during the forward and backward passes. 

5 



## **3 Methodology** 

In this section, we introduce the methodology of our evaluation for demonstrating a comparison of performance and energy among multiple accelerators. We first present the selected hardware settings and DNNs from different AI areas. Then we illustrate our evaluation methods. 

### **3.1 Hardware Setup** 

As we would like to evaluate the most commonly used accelerators for DNN training, we select many-core processors from four vendors including Intel<sup>6</sup> , NVIDIA, AMD, and Google. For each vendor, we select one to three processors for evaluation. The details of the selected accelerators are listed in Table 4, which presents the key parameters related to the performance of accelerators. 

Table 4: Hardware Setup 

|Vendor|Accelerator Model|Memory|Theoretical FLOPS|MemoryBdw|MemoryType|CPU|
|---|---|---|---|---|---|---|
|Intel|An Alibaba Cloud CPU,12physical cores.|48GB|3.84 T(FP32)|119.21 GB/s|DDR4|Xeon Platinum 8163|
||Titan X(Pascal)|12GB|11 T(FP32)|480.4 GB/s|GDDR5X|i7-7820X|
|NVIDIA|Tesla P100<br>Tesla V100|16GB<br>16GB|9.5 T(FP32)<br>112 T(Tensor Core)|732.2 GB/s<br>897.0 GB/s|HBM2<br>HBM2|i7-6800K<br>i7-6800K|
|AMD|Radeon VII|16GB|13.44 T(FP32)|1 TB/s|HBM2|i7-4790|
|Goole|TPU v2-8|64GB|180 T (bfloat16)|600 GB/s|HBM|-|
|g|TPU v3-8|128GB|420 T(bfloat16)|900 GB/s|HBM|-|



### **3.2 Evaluated Workloads** 

### **3.2.1 Computation-intensive Operators** 

DNN models are mainly stacked by many layers, which generally invoke two main resource-consuming operators (ops), including the matrix multiplication (Matmul) and convolution in 2d dimension (Conv2d). During training, highthroughput kernels will conduct computation-intensive calculations for parallel operations like Matmul and Conv2d. Low-throughput kernels require serial operations and longer training time. 

To evaluate the performance of ops, the input data are synthetic tensors of different FLOPs. For the Matmul operator, tensor dimensions range from 2048 _×_ 2048 to 8192 _×_ 8192, a typical workload of Fully Connected DNN model (For example, the Transformer model); for the Conv2d operator, inputs and filters are selected from Resnet50 under different batch sizes, including 128 and 256. To ensure the utilization of accelerators, we select to show results of small, medium, and large sizes of input tensors, which are listed in Table 5. The F, K, and S in Table 5 refer to input size, kernel size, and strides, respectively. 

**CUDA C++ Setting** We adopted the CUDA C++ codes in DeepBench<sup>7</sup> and made two major revisions to it. Firstly, we erase the unnecessary kernels launched during the training iterations to ensure the best performance of cuBLAS. Secondly, we have the forward algorithm fixed at IMPLICIT_PRECOMP_GEMM while input Tensors vary, which eliminates the differences in real forwarding FLOPs for ops in the experiment. 

**TensorFlow Setting** TensorFlow has provided users with the necessary tools for profiling. By printing the timeline of training processes, we get the accurate calculation time of GPU kernels. 

Table 5: Input Tensor Sizes of Ops 

|Matmul Shae|Matmul I|Matmul II|Matmul III|
|---|---|---|---|
|p|(2048,2048)|(4096,4096)|(8192,8192)|
|FLOPs|1.72E+10|1.37E+11|1.10E+12|
||Conv2d I|Conv2d II|Conv2d III|
|Conv2d Shape|F(256,224,224,3)|F(128,112,112,64)|F(256,56,56,128)|
||K(7,7,3,64)S(2,2)|K(3,3,64,128)S(1,1)|K(3,3,128,256)S(1,1)|
|FLOPs|5.72E+10|2.28E+11|4.40E+11|



> 6Note that the CPU instance used in our experiments has 24 vCPUs on a rented Alibaba Cloud ECS Computing Type c5, which is, in fact, a package of 12 physical cores, serving only half of peak theoretical FLOPS: 1.92T FLOPS. 

> 7https://github.com/baidu-research/DeepBench.git 

6 



### **3.2.2 DNNs** 

To cover comprehensive AI applications, we choose DNN models from image classification with CNNs, language models with LSTM, speech recognition with Deep Speech 2 and the state-of-the-art Transformer language model. For CNNs, we choose ResNet-50 [2], Inception V3 [27], and VGG16[26] on the ImageNet [32] dataset. For LSTM, we selected the typical 2-Layer LSTM on the PTB dataset. For the Deep Speech 2 architecture, we train the model on the AN4 dataset. For Transformer, we test the model on the WMT14 EN-DE dataset. The details of DNN configurations are shown in Table 6. 

Table 6: Model and Dataset Setting 

|Networks|# Params<br>(million)<br>|Theoretical<br>MACs(GFLOPs)|Datasets|# Samples|Input<br>Sizes|
|---|---|---|---|---|---|
|VGG16|138|86.66|||224*|
|Resnet50|26|32.79|ImageNet|1.2 m|224|
|Inception V3|27|23.84|||*3|
|2-Layer LSTM|51|6.91|PTB|42 k|20|
|DeepSpeech 2|87|40.99|AN4|948|100-400|
|Transformer|65|46.08|WMT14<br>EN-DE|36 m|512|
|Note:|For nlpm|odels,input sizes|refer to the|sequence|lengths here.|



### **3.3 Evaluation Methods** 

**Evaluation Metrics** In order to present the readers a comprehensive scope of different tasks and AI accelerators, we use performance and energy<sup>8</sup> as the evaluation metrics. For the performance, we report the throughput of the accelerators in terms of samples per second (Samples/s).<sup>9</sup> For the energy, we sample the system power (in Watt) in every 2ms during the training process using the built-in interfaces provided by NVIDIA Management Library [36] on NVIDIA GPUs and ROCm System Management Library [11] on the AMD GPU. The energy is directly derived from the evaluated performance and power as _performancepower_<sup>.The metric details are defined in Table 7.</sup> 

Table 7: Definition of Metrics 

|Index|Definition|Unit|
|---|---|---|
||Time duration for the computingdevice toprocess||
|Performance|a certain mini batch over an iteration.|Second|
||The electrical energycost of the computingdevice||
|Energy|toprocess a sample.|Jper sample|



Table 8: Software Setup 

||||Workloads|||
|---|---|---|---|---|---|
|Accelerators|Ops|Transformer|CNNs<br>LSTM|DeepSpeech 2|Libraries|
|Intel CPU|||||MKL-2019.4-243|
|NVIDIA GPUs|ab||c||CUDA-10.0|
|AMD GPU||a|||ROCm-v2.4|
|Google TPUs||||/|-|
|Software|a: Tel|nsorflow 1.1|l 4.0;b: CUDA|C++;c: PyTorch|1.1;/: no resultsyet|



**Software Tools for Measurement** For the measurement of operations and DNNs, we make evaluations with the software listed in Table 8. As TPU mainly supports TensorFlow, we measure the TPU training performance with TensorFlow for the best performance. Note that we report the average data collected from the experiments. Data collected from both op level benchmarking and end-to-end training on different platforms are repeated for 1000 rounds(iterations), leaving out the 10 rounds that report the value of the highest and lowest metric. 

> 8For the CPU and TPUs, performance is the key metric for benchmarking. For GPUs, the above two metrics are adopted for analysis. 

> 9The units in pictures of samples are images for CNNs, sentences for LSTM, utterances for Deep Speech 2, and token for Transformer, respectively. 

7 



## **4 Experimental Results** 

In this section, we present the experimental results and discussions, including the performance of low-level mathematical operators (Subsection 4.1), performance of end-to-end training (Subsection 4.2), and energy consumption of end-to-end training (Subsection 4.3). 

### **4.1 Performance of Computation-intensive Operators** 

We evaluate the AI Accelerators on the two major operators (i.e., matrix multiplication and 2D convolution) that are computation-intensive and widely used in DNN training. We test the operators with small, medium, and large FLOPs, which represent different computation intensity for accelerators, as shown in Table 5. 

![Figure](assets/figure_0005_page_0008.svg)Figure 5: The results of CPU’s performance and utilization on two computation intensive operators. The lines refer to utilization, and the bars refer to TFLOPS. 

### **4.1.1 CPU Results** 

Multi-threading and AVX (Advanced Vector Extension) are two main techniques to exploit the many-core and SIMD capabilities of modern CPUs. To evaluate the performance of operators using an Intel Optimization version of TensorFlow with AVX512 enabled, we enumerate the number of threads from 1 to 12. The results of the utilization and performance of Intel Xeon 8163 are shown in Fig. 5. The maximum utilization is up to 83% and 68% on Matmul and Conv2d, respectively. For the matrix multiplication, the computing performance is almost linear to the number of threads (shown in Fig. 5 as Thrd). 

Regarding the Conv2d operator, the Conv2d ops achieve overall lower utilization than Matmul ops with even higher FLOPs (Conv2d I, II compared with Matmul I, II). Since there are no memory bottlenecks for low-level operations, the results on Matmul and Conv2d indicate that there exist further opportunities to optimize the performance of Conv2d with multi-core and AVX techniques. 

8 



![Figure](assets/figure_0006_page_0009.svg)Figure 6: The results of GPU and TPU’s performance (FLOPS) and utilization of Matmul and Conv2d operators. 

### **4.1.2 GPU and TPU Results** 

On the GPU and TPU, the performance and utilization of operations that have been transformed into FLOPS and percentage are shown in Fig. 6. Among GPUs, P100 has the lowest utilization under all workloads. V100 has achieved up about 97% utilization for Matmul ops and 92% for Conv2d ops under FP32/FP16 Precision. The utilization of V100 with Tensor Core is 9%-31% and 22%-37% lower than V100 (FP32) on Matmul and Conv2d respectively (shown in Fig. 6 as V100 (TC) and V100 (FP32)). 

Computation-intensive operations call high-throughput kernels for calculating to achieve the highest FLOPS (throughput), as can be seen from Fig. 6. The gap between CUDA C++ and TensorFlow implementation of performance on Tensor Core is larger on Conv2d than Matmul. Among GPUs, V100, Titan X, and Radeon VII achieve the same best FP32 performance and utilization on both operators, which proves their ability of high-throughput calculation. TPU V2 achieves nearly optimal utilization on both the operators, especially about 99% on the Conv2d operator - much better than Tesla V100 with Tensor Core. We may safely draw the conclusion that there remain performance bottlenecks for V100 with Tensor Core enabled. 

### **4.2 Performance of End-to-end Training** 

### **4.2.1 CPU Results** 

The evaluated virtual CPU instance contains 12 physical cores and supports up to 8-way multiprocessing. We first evaluate how the number of threads affects training performance, which is shown in Fig. 7. It can be seen that on the 12-core CPU, the 2-way multiprocessing (24 threads) generally achieves the best performance on CNNs and LSTM. Transformer and Deep Speech 2 are two exceptions. When _threads_ = 1, the CPU utilization is high enough for Transformer, which limits the effect of multi-threading. For Deep Speech 2, the best number of threads on Deep Speech 2 of Fig. 7 is 8. The thread for pre-processing data could lack CPU resources to do computations such that the overall performance would be even worse. 

Note that with doubled number of threads (not larger than the number of physical cores), the increasing rate of performance drops due to the limitation of total CPU FLOPS and Amdahl’s law - the effect of multi-threading is bottlenecked by the percentage of serial operators in the DNNs. For small threads, as we analyzed in Section 4.1, the 

9 



![Figure](assets/figure_0007_page_0010.svg)Figure 7: The results of CPU’s performance of end-to-end training. Note that ’bs’ refers to the batch size in this paper. 

performance improvement of the parallel operator of high computation intensity is around 90%-100% with the double threads. 

### **4.2.2 GPU Results** 

**Performance vs Mini-batch Size** There exist thousands of cores in current GPUs. When training DNNs, the GPU workload linearly increases with respect to the mini-batch size. Therefore, we first select a representative GPU (i.e., Tesla V100) in Mixed and FP32 Precision to demonstrate the relationship between performance and the mini-batch size, which is displayed in Fig. 8. As can be seen from Fig. 8, the improvement in performance by increasing batch size is greater for Mixed Precision than FP32 Precision. Smaller mini-batch sizes may not fully utilize the computing resource of GPUs. Though, the benefit of doubling the batch size from medium to the largest for Transformer and Deep Speech 2 drops in 30% and 5% respectively in FP32 Precision, which indicates that one should set proper mini-batch size for the particular DNN and accelerator to achieve maximum performance. 

![Figure](assets/figure_0008_page_0010.svg)Figure 8: The result of performance on V100 GPU with different mini-batch sizes using FP32 and Mixed Precision. 

![Figure](assets/figure_0009_page_0010.svg)Figure 9: The results of performance comparison on GPUs with CNNs. Note that the Titan X GPU runs out of memory when training inception v3 and _batchsize_ = 128. 

10 



The performance of end-to-end training of different DNNs on different GPUs (including NVIDIA and AMD) is shown in Fig. 9 and Fig. 10. We compare their performance in two applications (i.e., Image Classification and NLP models). 

Table 9: The Average Utilization of Tensor Core 

|DNNs|Resnet50|Inception v3|Vgg16|2-Layer LSTM|DeepSpeech 2|Transformer|
|---|---|---|---|---|---|---|
|Tenser Core Util|9.2%|6.8%|7.3%|3.9%|5.1%|4.4%|



**CNNs** It can be seen from Fig. 9 that Tesla V100 has the best performance with both FP32 and Mixed Precision training among the tested GPUs. Under the same numerical precision, NVIDIA V100 generally achieves 1.5 _×_ -2 _×_ higher performance than the AMD Radeon VII GPU. Among NVIDIA GPUs, Tesla P100 and Titan X (Pascal) have close performance as these two GPUs have similar peak FLOPS and memory bandwidth, as shown in Table 4. When the benefit of switching from FP32 Precision to Mixed Precision is up to only 2 _×_ higher throughput for V100, the utilization of Tensor Core is 9% at most on Resnet50 in Table 9 - far from the peak FLOPS. For two desktop-level GPUs between NVIDIA Titan X (Pascal) and AMD Radeon VII, the peak FLOPS, and memory bandwidth of Radeon VII is about 22% and 52% higher than that of Titan X (Pascal) respectively, whereas Titan X achieves slightly higher performance than Radeon VII on Resnet 50. The phenomenon indicates there exist opportunities for future optimizations on the AMD software ecosystem to better utilize the hardware ability in various deep learning workloads. 

**NLP Models** The result<sup>10</sup> of performance of NLP models are shown in Fig. 10. NVIDIA GPUs achieve overall higher performance than the AMD GPU. In particular, Titan X (Pascal) is nearly 1.9 _×_ and 1.5 _×_ faster than Radeon VII on recurrent and non-recurrent models, respectively. Among NVIDIA GPUs, Tesla V100 always has the best performance; however, with even slighter improvement in Mixed Precision compared to the FP32 counterpart. The margin improvement of Tesla V100 with Mixed Precision indicates that the software library should be further optimized for NLP models. 

![Figure](assets/figure_0010_page_0011.svg)Figure 10: The results of performance comparison on GPUs with NLP models. Deep Speech 2 cannot be successfully run on the Radeon VII GPU as some operators are not supported. 

### **4.2.3 TPU Results** 

The performance of TPUs of two generations is shown in Fig. 11. On three evaluated DNNs, TPU v3-8 is about 1.5 _×_ -1.7 _×_ faster than TPU v2-8. However, the peak FLOPS of TPU v3-8 is around 2.3 _×_ higher than TPU v2-8, as shown in Table 4, which indicates that FLOPS utilization on TPU v3-8 is much lower than that on TPU v2-8 and partially bottlenecked by the 1.5 _×_ improvement in memory bandwidth. The experimental results indicate that there still exist hard-ware level optimization for better performance of TPU v3-8 to fit modern deep learning networks. 

### **4.2.4 Discussion** 

**Comparison Between GPU and TPU** V100 and TPU represent two of the fastest AI accelerators in the world. It can be seen from Fig. 11 that TPUs outperform Tesla V100 GPU in the three evaluated models. TPU V2 and TPU v3 have 1.6 _×_ and 3.75 _×_ more FLOPS, but 0.66 _×_ and the same memory bandwidth compared with V100 respectively. On CNNs (ResNet-50 and Inception v3), TPU V2-8 and TPU V3-8 achieve more than 1.5 _×_ and 2.7 _×_ higher throughput 

> 10We present the results with the largest batch size to explore the best performance. As the connectionist temporal classification (CTC) loss function of the Deep Speech 2 is not supported on the AMD GPU, we exclude the AMD result for this case. 

11 



than Tesla V100 with Tensor Cores respectively. However, on the Transformer architecture, TPU V2-8 is very close to Tesla V100 GPU, and TPU V3-8 achieves around 1.7 _×_ faster than Tesla V100. 

![Figure](assets/figure_0011_page_0012.svg)Figure 11: The results of performance comparison between Tesla V100 GPU and TPUs. 

**The Limitation of Tensor Core** The Volta architecture transforms the convolutional parts in a neural network into matrix calculations before Tensor Cores conduct FP16 precision parallel calculations. Tesla V100(PCIe), in end-to-end Mixed Precision training, outperforms itself in FP32 training over two times with only 9% utilization of the peak FLOPS of Tensor Core. However, the intensive calculations in models like Resnet50 and Transformer account for over 40% of the total calculations. There are several possible reasons to list for this mismatch: the remaining FP32 low-throughput calculations in Mixed Precision training, the inherent low utilization of Tensor Core, and the limited memory bandwidth [30]. 

### **4.3 Energy Efficiency of End-to-end Training** 

![Figure](assets/figure_0012_page_0012.svg)Figure 12: Comparison of energy consumption on GPUs. 

We compare the energy consumption of different GPUs among different DNN models in Fig. 12. Server-level GPUs, Tesla V100 and P100, are mostly the top-two in energy efficiency. Especially V100 has at most 5.2 _×_ lower energy consumption than other GPUs since it performs a remarkably higher training throughput with negligible power sacrifice. For Titan X (Pascal) and Radeon VII, increasing the processed batch size generally leads to the improvement in the resource utilization and training throughput, which consequently improves energy efficiency. 

However, for CNNs and Deep Speech 2, the larger batch size of those CNNs on P100 and V100 do not help conserve energy consumption and even lead to marginal increments, since server GPUs achieve similar training throughput on models under two batch sizes, as shown in Fig. 8. 

For recurrent models, Radeon VII should be further optimized due to high energy consumption. As can be seen from Fig. 12, the maximum energy efficiency of V100 on 2-Layer LSTM compared with Radeon VII is 4 _×_ . Meanwhile, NVIDIA GPUs achieve similar energy efficiency under different batch sizes, while the Radeon VII only achieve better energy efficiency when the batch size increases. 

12 



At last, Transformer, the non-recurrent translation model, launches GPU kernels that require low training power during training. The low utilization of GPU leads to promotions in the training throughput while increasing the batch sizes, which results in better energy efficiency for all the GPUs. 

For ease of reference to the experimental results, all raw numbers of end-to-end training are shown in Table 10. 

|||||T<br>|able 10<br>|: End<br>|-to-en<br>|d Exp<br>|erimen<br>|tal Res<br>|ults<br>||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||DNN<br>|Resn<br>|et50|Incept<br>|on V3|VG<br>|G16||-Layer LST|M<br>|D|eepSpeec<br>|2||Transformer||
|Model|Batch Size|64|128|64|128|64|128|64|128|256|8|16|32|1024|2048|4096|
||Titan X|189.25|187.43|131.25|OOM|132.46|94.62|622.36|669.18|789.21|25.81|32.00|35.96|9,359.36|10,690.56|13,148.16|
||Radeon VII|206.45|166.23|133.33|110.34|125.49|126.73|145.45|261.22|419.67|-|-|-|5,521.00|7,658.00|8,722.00|
||P100|202.47|207.61|147.69|151.23|135.58|138.12|591.25|656.29|705.28|27.59|32.00|36.36|6,973.44|11,632.64|11,796.48|
|PERF|V100|311.25|334.58|230.53|243.59|213.32|216.25|817.44|898.21|1,055.87|38.10|43.24|44.44|9,431.04|13,783.04|18,350.08|
|(samples/s)|V100(MP)|589.39|680.45|372.21|406.88|**385.43**|**418.59**|**889.21**|**1,019.55**|**1,096.57**|**42.11**|**47.06**|**46.38**|9,236.48|14,950.40|20,152.32|
||TPU V2|1,066.67|1,163.64|800.00|914.29|-|-|-|-|-|-|-|-|16,000.00|19,033.46|21,333.33|
||TPU V3|1,280.00|**1,882.35**|**1,488.37**|**1,523.81**|-|-|-|-|-|-|-|-|**25,600.00**|**29,383.07**|**35,772.93**|
||CPU|6.35|11.77|6.44|12.33|8.84|16.64|2.69|3.22|4.72|3.41|5.17|11.44|300.12|-|-|
||Titan X|260.15|263.21|265.79|OOM|266.72|263.15|276.08|275.25|278.21|128.74|136.58|141.55|212.64|236.35|259.23|
||Radeon VII|275.50|285.50|**141.00**|**142.00**|271.00|260.00|**134.00**|**141.00**|**169.50**|-|-|-|126.51|149.21|172.55|
|POW|P100|**169.64**|**176.51**|162.38|178.03|**166.76**|**163.45**|195.32|197.35|201.47|109.65|119.52|127.87|105.70|136.39|151.32|
|(watt)|V100|193.87|206.61|191.56|198.68|190.71|201.19|226.61|234.32|240.21|107.54|121.64|131.77|115.67|144.61|162.26|
||V100(MP)|171.56|188.87|168.71|190.38|183.73|187.41|186.37|199.41|215.58|**98.21**|**115.05**|**114.77**|**90.43**|**107.04**|**132.21**|
||Titan X|1.37|1.40|2.03|OOM|2.01|2.78|0.44|0.41|0.35|4.99|4.27|3.94|0.0227|0.0221|0.0197|
||Radeon VII|1.33|1.72|1.06|1.29|2.16|2.05|0.92|0.54|0.40|-|-|-|0.0229|0.0195|0.0198|
|NRG|P100|0.84|0.85|1.10|1.18|1.23|1.18|0.33|0.30|0.29|3.97|3.74|3.52|0.0152|0.0117|0.0128|
|(J/sample)|V100|0.62|0.62|0.83|0.82|0.89|0.93|0.28|0.26|0.23|2.82|2.81|2.96|0.0123|0.0105|0.0088|
||V100(MP)|**0.29**|**0.28**|**0.45**|**0.47**|**0.48**|**0.45**|**0.21**|**0.20**|**0.20**|**2.33**|**2.44**|**2.47**|**0.0098**|**0.0072**|**0.0066**|
|Note: ’-’ me|ans the item is|currentlyun|supported.||||||||||||||



## **5 Related Work** 

Benchmarks are key methods to make the hardware and software move forward to better targets (e.g., performance and/or energy). In the era of deep learning, training tasks are computationally intensive and resource-consuming. The running time performance and energy consumption of AI accelerators are the two major concerns for practitioners. Started from 2016, deep learning frameworks are rapidly developed to support many types of processors and accelerators, like CPUs, GPUs, and TPUs. 

Researchers [13, 19] started to evaluate the performance among different deep learning frameworks and different GPUs. However, these works mainly focused on software-level evaluation in terms of performance. Later, Stanford DAWN deep learning benchmark [15] and MLPerf [20] were developed for the evaluation of training and inference performance under different software and hardware platforms. These two open benchmark platforms have attracted many submissions from vendors, but they mainly focused on the end-to-end training performance without reasoning on the results. Several new benchmarks including ParaDnn [14], AIBench [16], Fathom [17], and TBD [18] conduct better analysis, whereas lack comprehensive scope of AI accelerators or the energy efficiency. We list some benchmarks in Table. 1 for comparison. Recently Wang et al. [14] proposed ParaDnn to measure the performance of various hardware including Intel CPU, NVIDIA GPU and Google TPU, which was the closest to our work. 

The energy consumption is of great importance to the servers that run resource-intensive tasks, but there is little study measuring the power and energy consumption of DNN training tasks. One related work is [24] which studied the impact of GPU dynamic voltage and frequency scaling (DVFS) on the training performance and energy. 

## **6 Conclusion** 

In this paper, we made a comprehensive evaluation of the training performance and energy consumption on various modern AI accelerators including Intel CPU, AMD/NVIDIA GPUs, and Google TPUs, covering a representative set of deep learning workloads (CNNs for image recognition, recurrent and non-recurrent neural networks for translation, and Deep Speech 2 for speech recognition). Our benchmark results provide several levels of comparison for end-users and hardware/software designers, including hardware performance, software utilization, energy consumption. In the future, we will extend our evaluation to more AI accelerators such as FPGAs and more training categories. Another direction is to benchmark the performance and energy efficiency of deep learning inference tasks on both server devices and edge/mobile devices. 

13 



## **Acknowledgements** 

The research was supported by Hong Kong RGC GRF grant HKBU 12200418. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X (Pascal) used for this research. We also gratefully acknowledge Google for providing TPUs to support our research in the TensorFlow Research Cloud (TFRC) program. 

## **References** 

- [1] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. _Nature_ , 521(7553):436–444, 2015. 

- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. _arXiv preprint arXiv:1512.03385_ , 2015. 

- [3] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In _the IEEE conference on computer vision and pattern recognition_ , pages 779–788, 2016. 

- [4] Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In _International conference on machine learning_ , pages 173–182, 2016. 

- [5] etc Jeffrey Dean. Large scale distributed deep networks. In _Advances in Neural Information Processing Systems 25_ . 2012. 

- [6] David Luebke, Mark Harris, Naga Govindaraju, Aaron Lefohn, Mike Houston, John Owens, Mark Segal, Matthew Papakipos, and Ian Buck. GPGPU: general-purpose computation on graphics hardware. In _Proceedings of the 2006 ACM/IEEE conference on Supercomputing_ , page 208. ACM, 2006. 

- [7] Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In _2017 ACM/IEEE 44th Annual International Symposium on Computer Architecture (ISCA)_ , pages 1–12. IEEE, 2017. 

- [8] Chris Ying, Sameer Kumar, Dehao Chen, Tao Wang, and Youlong Cheng. Image classification at supercomputer scale. _arXiv preprint arXiv:1811.06992_ , 2018. 

- [9] Scott Cyphers et al. Intel ngraph: An intermediate representation, compiler, and executor for deep learning. _arXiv preprint arXiv:1801.08058_ , 2018. 

- [10] Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cuDNN: Efficient primitives for deep learning. _arXiv preprint arXiv:1410.0759_ , 2014. 

- [11] AMD. ROCm System Management Library . [Online] `https://rocm-documentation.readthedocs.io/ en/latest/ROCm_System_Managment/ROCm-System-Managment.html` . 

- [12] Martın Abadi et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. _Software available from tensorflow. org_ , 2015. 

- [13] I. Paul et al. Harmonia: Balancing compute and memory power in high-performance gpus. In _2015 ACM/IEEE 42nd Annual International Symposium on Computer Architecture (ISCA)_ , pages 54–65, June 2015. 

- [14] Yu Emma Wang, Gu-Yeon Wei, David Brooks, et al. Benchmarking TPU, GPU, and CPU platforms for deep learning. _arXiv preprint arXiv:1907.10701_ , 2019. 

- [15] Cody Coleman, Deepak Narayanan, Daniel Kang, Tian Zhao, Jian Zhang, Luigi Nardi, Peter Bailis, Kunle Olukotun, Chris Ré, and Matei Zaharia. Dawnbench: An end-to-end deep learning benchmark and competition. _Training_ , 100(101):102, 2017. 

- [16] Wanling Gao, Fei Tang, Lei Wang, Jianfeng Zhan, Chunxin Lan, Chunjie Luo, Yunyou Huang, Chen Zheng, Jiahui Dai, Zheng Cao, Daoyi Zheng, Haoning Tang, Kunlin Zhan, Biao Wang, Defei Kong, Tong Wu, Minghe Yu, Chongkang Tan, Huan Li, Xinhui Tian, Yatao Li, Junchao Shao, Zhenyu Wang, Xiaoyu Wang, and Hainan Ye. Aibench: An industry standard internet service ai benchmark suite, 2019. 

- [17] Robert Adolf, Saketh Rama, Brandon Reagen, Gu-Yeon Wei, and David M. Brooks. Fathom: Reference workloads for modern deep learning methods. _CoRR_ , abs/1608.06581, 2016. 

- [18] Hongyu Zhu et al. TBD: benchmarking and analyzing deep neural network training. _CoRR_ , abs/1803.06905, 2018. 

- [19] Shaohuai Shi, Qiang Wang, Pengfei Xu, and Xiaowen Chu. Benchmarking state-of-the-art deep learning software tools. In _2016 7th International Conference on Cloud Computing and Big Data (CCBD)_ , pages 99–104. IEEE, 2016. 

14 



- [20] MLPerf. MLPerf. `https://mlperf.org` , 2019. Accessed: 2019-09. 

- [21] Qiang Wang et al. EPPMiner: An extended benchmark suite for energy, power and performance characterization of heterogeneous architecture. In _Proceedings of the Eighth International Conference on Future Energy Systems_ , e-Energy ’17. ACM, 2017. 

- [22] Xinxin Mei, Qiang Wang, and Xiaowen Chu. A survey and measurement study of GPU DVFS on energy conservation. _Digital Communications and Networks_ , 3(2):89–100, 2017. 

- [23] Qiang Wang and Xiaowen Chu. GPGPU performance estimation with core and memory frequency scaling. In _2018 IEEE 24th International Conference on Parallel and Distributed Systems (ICPADS)_ , pages 417–424. IEEE, 2018. 

- [24] Zhenheng Tang et al. The impact of GPU DVFS on the energy and performance of deep learning: an empirical study. In _Proceedings of the Tenth ACM International Conference on Future Energy Systems_ , pages 315–325. ACM, 2019. 

- [25] Vincent Chau, Xiaowen Chu, Hai Liu, and Yiu-Wing Leung. Energy efficient job scheduling with DVFS for CPU-GPU heterogeneous systems. In _Proceedings of the Eighth International Conference on Future Energy Systems_ , pages 1–11. ACM, 2017. 

- [26] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 

- [27] Christian Szegedy et al. Rethinking the inception architecture for computer vision. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2818–2826, 2016. 

- [28] Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. Learning to forget: Continual prediction with LSTM. 1999. 

- [29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing systems_ , pages 5998–6008, 2017. 

- [30] Samuel Williams, Andrew Waterman, and David A. Patterson. Roofline: An insightful visual performance model for floating-point programs and multicore architectures*. 2008. 

- [31] Xinxin Mei, Xiaowen Chu, Hai Liu, Yiu-Wing Leung, and Zongpeng Li. Energy efficient real-time task scheduling on CPU-GPU hybrid clusters. In _IEEE INFOCOM 2017-IEEE Conference on Computer Communications_ , pages 1–9. IEEE, 2017. 

- [32] Jia Deng et al. ImageNet: A large-scale hierarchical image database. In _Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on_ , pages 248–255. IEEE, 2009. 

- [33] Ofir Press and Lior Wolf. Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_ , 2016. 

- [34] Greg Regnier, Dave Minturn, Gary McAlpine, Vikram A Saletore, and Annie Foong. ETA: Experience with an Intel Xeon processor as a packet processing engine. _IEEE Micro_ , 24(1):24–31, 2004. 

- [35] Mu Li, Tong Zhang, Yuqiang Chen, and Alexander J Smola. Efficient mini-batch training for stochastic optimization. In _Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining_ , pages 661–670. ACM, 2014. 

- [36] NVIDIA. NVIDIA Management Library . [Online] `https://developer.nvidia.com/ nvidia-management-library-nvml` , 2018. 

15 

