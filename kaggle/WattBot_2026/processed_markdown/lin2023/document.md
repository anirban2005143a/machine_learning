# Neural Network Methods for Radiation Detectors and Imaging 

S. Lin<sup>1,2</sup> , S. Ning<sup>2</sup> , H. Zhu<sup>2</sup> , T. Zhou<sup>3</sup> , C. L. Morris<sup>1</sup> , S. Clayton<sup>1</sup> , M. Cherukara<sup>4</sup> , R. T. Chen<sup>2,5,6,*</sup> , and Z. Wang<sup>1,*</sup> 

> 1Los Alamos National Laboratory, Los Alamos, NM 87545, USA 

> 2Department of Electrical and Computer Engineering, The University of Texas at Austin, Austin , TX 78705, USA 

> 3Center for Nanoscale Materials, Argonne National Laboratory, Lemont , IL 60439, USA 

> 4Advanced Photon Source, Argonne National Laboratory, Lemont , IL 60439, USA 

> 5Microelectronics Research Center, The University of Texas at Austin, Austin , TX 78758, 

USA 

> 6Omega Optics, Inc., Austin , TX 78757, USA 

> *Correspondence: R. T. C. (chenrt@austin.utexas.edu), Z. W. (zwang@lanl.gov) 

##### **Abstract** 

Recent advances in image data processing through machine learning and especially deep neural networks (DNNs) allow for new optimization and performance-enhancement schemes for radiation detectors and imaging hardware through data-endowed artificial intelligence. We give an overview of data generation at photon sources, deep learning-based methods for image processing tasks, and hardware solutions for deep learning acceleration. Most existing deep learning approaches are trained offline, typically using large amounts of computational resources. However, once trained, DNNs can achieve fast inference speeds and can be deployed to edge devices. A new trend is edge computing with less energy consumption (hundreds of watts or less) and real-time analysis potential. While popularly used for edge computing, electronic-based hardware accelerators ranging from general purpose processors such as central processing units (CPUs) to application-specific integrated circuits (ASICs) are constantly reaching performance limits in latency, energy consumption, and other physical constraints. These limits give rise to next-generation analog neuromorhpic hardware platforms, such as optical neural networks (ONNs), for high parallel, low latency, and low energy computing to boost deep learning acceleration. (LA-UR-23-32395) 

1 



## **1 Introduction** 

X-rays produced by synchrotrons and free electron lasers (XFELs), together with high-energy photons above 100 keV, which are often generated using high-current (kA) electron accelerators and lately high-power lasers, are widely used as radiographic imaging and tomography (RadIT) tools to examine material properties and their temporal evolution. Spatial resolution ( _δ_ ) down to atomic dimensions is possible by using diffractionlimited X-rays, _δ ∼ λ/_ 2, corresponding to Abbe’s diffraction limit for X-ray wavelength _λ_ . The overall object size that X-rays can probe readily reaches a length ( _L_ ) greater than 1 mm, which is limited by the X-ray attenuation length and is X-ray energy dependent. In room-temperature water, for example, _L_ = 0.19, 1.2, 5.9, and 14.1 cm for 1 _/e_ -attenuation length of 10 keV, 20 keV, 100 keV, and 1 MeV X-rays, respectively. The temporal resolution has now approached a few femtoseconds by using XFELs, where an XFEL experiment can be repeated for many hours in a pump-probe configuration. In other words, the spatial dynamic range (i.e. for 10 keV X-rays, L _∼_ 1 mm) is 2 _L/λ >_ 10<sup>7</sup> and temporal dynamic range is _>_ 10<sup>18</sup> . Such ultra-wide-dynamic-range abilities of X-ray and photon techniques to connect elementary atomic and molecular processes, which are described by quantum physics and happen ultra fast (sub-nanosecond), with emergent macroscopic material properties and functions, which are usually treated classically through continuum approximations, make them extremely valuable in a wide range of applications, such as medicine (i.e. new drug discovery), high-energy density battery development, and applications in materials exposed to high-temperature, high radiation, and other harsh or ‘extreme’ conditions. Additional applications include the optimization of chemical catalysis and the development of new superconductors and other quantum materials for information technology, accelerated computing, and artificial intelligence (AI). 

The enormous spatial and temporal dynamic ranges give rise to ‘big data’ in X-ray imaging, tomography, and photon science. Theoretically, 1 mm<sup>3</sup> of water contains about 5.6 _×_ 10<sup>_−_5</sup> mole of water molecules ( _N_ = 3.3 _×_ 10<sup>19</sup> ). If the position of every molecule were recorded, the memory size would be _N_ log2 _N_ (log2 _N_ is the bit length for a binary data system) or 2.2 _×_ 10<sup>21</sup> bits. In experiments, explosive data growth in X-ray and other forms of RadIT is built upon steady progress for more than 120 years in X-ray and radiation sources, detectors, computation, and lately data science. The fourth generation synchrotrons such as APS-U [1] and PETRA IV [2] will have a significant reduction in emittance and a brilliance increase by a factor about 10<sup>3</sup> over the parameters of the third generation synchrotrons such as APS and PETRA III. XFELs, which are many of orders of magnitude brighter than synchrotrons, will run at a higher repetition rate up to 1 MHz [3]. LCLS, in comparison, operates at 120 Hz. High-speed detectors with frame rate frequencies above 1 MHz are commercially available. The combination of high-repetition-rate experiments with a mega-pixel and larger recording system leads to high data rates, exceeding 1 TB/s (1 TB = 10<sup>12</sup> bytes), as we discuss further in Sec. 2.1. 

Big data not only presents a significant challenge to data handling in terms of computing speed, computing power, short- and long-term computer memory, and computer energy consumption, which all together is called ‘computational resources’, but also offer a transformative approach to process and interpret data, _i.e._ machine learning (ML) and artificial intelligence (AI) through data-enabled algorithms. Such algorithms, including deep learning (DL) [4, 5], are distinctive from traditional physics, statistical, and other forward-model- or domain-knowledge-driven algorithms. Traditional algorithms are based on the domain knowledge, such as physics and statistics, and applicable to both small or large ensembles of data. In contrast, data-driven models may only rely on data explicitly for model training (tuning), model validation and use, with no domain knowledge required. In practice, domain knowledge always helps, partly due to the fact that some aspects of data models, such as the model architecture and other hyper-parameters, are chosen pragmatically and do 

2 



not depend on the data. The amount of data required for data model training depends on the number of model parameters such as weights, activation functions, the number of nodes, _etc._ It is not uncommon that a deep neural network (DNN) may contain billions of tunable free parameters, which require a commensurate amount of data for training. Hybrid approaches to ML and AI, which merge data and domain knowledge, are increasingly popular. Hybrid models not only supplement data-driven models with domain knowledge and reduce the amount of data required for training, but also accelerate the computational speed of traditional forward models by 10 to more than 100 times by bypassing some detailed and time-consuming computations. 

We may differentiate two approaches to ML and AI by the computational resources involved and how the resources are distributed. In the _centralized_ approach, data are collected from distributed locations or different data acquisition instruments through the internet. The data are then stored in a data center, and processed by high-performance computers or mainframes. Very large traditional, ML, or hybrid algorithms can be deployed in the data center, which also requires correspondingly large memory, energy and power consumption. Estimated global data center electricity consumption in 2022 was 240 - 340 TWh [6, 7], or around 1-1.3% of global final electricity demand. Cloud computing and data centers are now widely used to process ‘big data’ in industry, health care, and research institutions. Through the cloud computing and data center approach, data generation and data processing tasks can be separated, which can mitigate the computation and data processing burden on people who generate data. In the _distributed_ or _edge_ approach, ML and AI, together with the computing hardware, are deployed at the individual device or instrument level. Distributed computing now pairs with distributed data. Through an internet of ML/AI-enhanced instruments, each ML/AI-enhanced instrument can be optimized for a specific purpose such as data reduction and real-time data processing. The large volumes, varieties, and generation rate of X-ray data motivate automated processing and reduction in light sources, such as synchrotrons and XFELs, to reduce the memory requirement, minimize latency related to data transmission and processing, and lower energy and power consumption. 

Centralized approaches and large models are commonly executed by a large team of people. A Meta AI research team recently introduced the model called Segment Anything Model (SAM) and a dataset of more than 1 billion masks on 11 Million images [8]. Nvidia unveiled Project Clara at its recent GTC conference, showing early results using DL post-processing to dramatically enhance existing, often grainy and indistinct echocardiograms (sonograms of the heart). Clara motivates acceleration in research being done on several fronts that exploits explosive growth in DL computational capability to perform analysis that was previously impossible or far too costly. One technique is called 3D volumetric segmentation that can accurately measure the size of organs, tumors or fluids, such as the volume of blood flowing through arteries and the heart. NVIDIA claims that a recent implementation, an algorithm called V-Net, “would’ve needed a computer that cost $10 million and consumed 500 kW of power”. 

There are some other successes using ML and AI in areas such as HEP experiments (i.e. Higgs boson discovery) and electron microscopy. The discovery of the Higgs boson is a major challenge in HEP and can be setup as a classification problem. Many ML methods such as decision trees, logistic regression, and DL algorithms have been applied to solve the signal separation problem [9, 10]. Meanwhile, ML and AI in electron microscopy are proposed to enable autonomous experimentation. Specifically, the automation of routine operations including but not limited to probe conditioning, guided exploration of large images, optimized spectroscopy measurements, and time-intensive and repetitive operations [11]. Edge ML and edge AI have already attracted a lot of attention in medicine. The fusion of DL and medical images creates dramatic improvements [12]. The concept is similar to techniques like high-dynamic range (HDR) photography, digital remastering of recordings or even film colorization in that one or more original sources of data are 

3 



![Figure](assets/figure_0001_page_0004.svg)Figure 1: Evolution of digital image sensor technology, which started with the introduction of the chargecoupled device (CCD) in the late 1960s. The latest trend is smart multi-functional CMOS image sensors enabled by three-dimensional (3D) integration in fabrication, innovations in heterogeneous materials and structures, neural networks, and edge computing. The upper left plot shows the increasing growth of sales for CMOS image sensors from 2016 to a projected value in 2026 [13]. 

post-processed and enhanced to bring out additional detail, remove noise or improve aesthetics. 

We will give an overview of DL methods for real-time radiation image analysis as well as hardware solutions for DL acceleration at the edge. Specifically, this paper is organized as follows. In Section 2, we discuss different radiation detectors and imaging devices, the resulting big data generation at photon sources, and the motivations for edge computing and DL. In Section 3, we present an overview of popular neural network architectures and several image processing tasks that have potential to be performed on edge devices. In addition, we discuss examples of DL-based methods for each. In Section 4, we present on overview of hardware solutions for DL acceleration and recent works that have applied them for computing at the edge. Lastly, Section 5 concludes this paper. 

## **2 Experimental data generation at photon sources** 

Data science at light sources is centered around scientific data generation and processing. Scientific data at synchrotrons and XFEL sources consist of experimental data, simulation and synthetic data, and meta data, such as detector calibration data, material properties of objects and sensors, and point spread functions of the detectors. Methods (imaging modalities) and detectors to collect experimental data are driven by the light sources, which continue to improve in source brightness, repetition rate, source coherence, photon energy, and spectral tunability. Computing hardware and algorithms are used to process experimental data and for data visualization. Computing hardware and algorithms are also used to simulate the experiments and produce synthetic data as close to the experimental data as possible for experimental data interpretation. Diversity of the materials to be integrated and imaged, together with the photon source and detector improvement have demanded continued improvements in computing hardware and algorithms towards real-time data processing, reductions in data transmission over long distances, and reducing data storage volumes. 

4 



### **2.1 Radiation Detectors and Imaging for Photon Science** 

Complementary metal-oxide semiconductor (CMOS) pixelated detectors are now widely used in photon science, replacing charge-coupled devices (CCDs) as the primary digital imaging technology, see Figure 1. Particle nature of photons motivates digitized detectors for photon counting. However, several factors complicate photon counting implementation in high-luminosity X-ray sources. The intensity of the sources can be too high to count individual photons one by one. The amount of X-ray photon-induced charge in CMOS detectors, which is the basis of X-ray photon counting, is not constant for the same X-ray energy. The source energy is not monochromatic, especially in imaging applications. Inelastic scattering of mono-chromatic X-rays can result in a broad distribution of X-ray photon energies after scattering by the object. When an optical camera is used together with a X-ray scintillator, see Table 1, the energy resolution of individual X-rays based on the photon detection is worse than direct detection when the X-ray directly deposits its energy in a silicon photo-diode. 

### **2.2 Imaging Modalities** 

X-ray microscopy uses X-ray lenses, zone plates, mirrors and other optics to modulate the X-ray field to form images [14]. As the X-ray intensities generated by synchrotrons and XFELs continue to increase, the advances in computational imaging modalities and lens-less X-ray modalities are increasingly used in synchrotrons and XFELs. In some cases, lens-less modalities may be preferred to avoid damages to X-ray lenses and mirrors. Lens-less modalities may also avoid aberration, diffraction due to imperfect X-ray lens, defects in zone plates, and other optics. The simplest lens-less X-ray imaging setup is radiography or projection imaging, pioneered by R¨ontgen. R¨ontgen’s lens-less radiographic imaging modality directly measures attenuated X-ray intensity due to absorption. Synchrotrons and XFELs also allow a growing number of phase contrast imaging, see Ref. [15] and references therein. Other modalities include in-line holography [16] and coherent diffractive imaging [17]. Additional phase and intensity modulation using pinholes, coded apertures, and kinoforms are also possible. Combinatorial X-ray modalities have also been introduced. For example, X-ray ptychography microscopy combines raster scanning X-ray microscopy with coherent diffraction imaging [18]. Compton scattering, usually ignored in the synchrotron and XFEL setting, may offer some additional information about the samples and potentially reduce the dose required [19]. The versatility of modalities requires different off-line and real-time data processing techniques. Background reduction is a common issue for all X-ray modalities. Real-time data processing, including energy-resolving detection, is highly desirable to distinguish different sources of X-rays since the detector pixel may simultaneously collect X-ray photons from different sources of X-ray attenuation and scattering. 

### **2.3 Real Time In-pixel Data-processing** 

When an X-ray photon is detected directly or indirectly through the use of a scintillator, charge-hole pairs are created through photo-to-electric conversion, or the photoelectric effect, within pixels of a camera or a pixelated array. CCD cameras, CMOS cameras, and low-gain avalanche detector (LGAD) arrays are now available for synchrotron and XFEL applications. Unlike a CCD camera, a CMOS image sensor collects charge and stores it in capacitors in pixels in parallel. Parallel charge collection and capacitor voltage digitization, which turns analog voltage signals into digitized signals, allow CMOS image sensors to operate at a much higher frame rate than CCDs. Charge and voltage amplification, in LGADs and sometimes in CMOS image sensors, are also used to improve signal-to-noise ratio. Any source of charge or voltage modulation not related 

5 



![Figure](assets/figure_0002_page_0006.svg)Figure 2: Integrated neural network and image sensor for real-time data processing and reductions. 

to the photoelectric effect is a potential source of noise. The photoelectric effect itself can lead to so-called Poisson noise due to the probabilistic process of photo-to-electric conversion. Other sources of noise include thermal noise or dark current, salt’n’pepper noise (due to charge migration in and out of pixel defects and traps), and readout noise. 

Automated real-time in-pixel signal and data processing are therefore required in CMOS and other pixelated array sensors for noise rejection, noise reduction, and noise correction for charge and voltage amplification controls, and for charge sharing corrections. Figure 2 illustrates an example of an integrated neural network and image sensor for real-time data processing. If uncorrected, noise can corrupt the image information and make it hard for post processing or misleading for data interpretation. Charge and voltage amplification may lead to nonlinear distortion between the X-ray flux and voltage signal. When the X-ray flux is too high, the so-called plasma effect may also need correction. Charge-sharing happens when an X-ray photon arrives at a pixel border and the electron-hole pairs created are spread across multiple neighboring pixels. 

By using transistor circuits, correlated double sampling (CDS) is an extremely successful example in noise reduction. Adaptive gain control circuits have been implemented in the AGIPD high-speed camera [20]. While real-time pixel-level signal processing by novel transistor circuits is important, there is also room for novel data-processing approaches that do not require hardware modifications to the pixels. As a recent example [21], a physics-informed neural network was demonstrated to improve spatial resolution of neutron imaging. Other novel applications of neural networks and their integration with hardware, see Figure 2, may offer new possibilities in noise reduction and image corrections. Integrated hardware and software (neutral networks are emphasized here) approaches for optimal performance also need to take into account of the complexity of the workflow [22, 23, 24], or computational cost, power consumptions, constrained by the frame rate and other metrics. For example, the computational cost of an _n × n_ matrix is _O_ ( _n_<sup>3</sup> ) [25]. 

## **3 Deep Learning for Image Processing** 

In recent years, deep learning (DL) has contributed significantly to the progress in computer vision, especially in different areas of image processing tasks including but not limited to image denoising, segmentation, super-resolution, and classification. DL is a sub-field of ML and AI that utilize neural networks (NNs) 

6 



Table 1: A comparison of different camera data rates, additional details and examples may be found in [15]. The state of the art is _>_ 10 Gpixel/s in continuous mode imaging. The burst mode imaging is _>_ 1 Tpixel/s [26]. 

|Detector|Facility|Array format|frame-rate|data|Data rate|
|---|---|---|---|---|---|
|(camera)|(particle/<br>photon)|(voxel size, _µ_m<sup>3 </sup>/<br>pixel size, _µ_m<sup>2 </sup>)|(fps/<br>Hz)<br>|bits|(GB/s)|
|AGIPD [27]|Eu-XFEL<br>(12.4 keV)|512 _×_ 128 <sup>a</sup><br>(200<sup>2</sup>_×_ 500 )|16 k/6.5 M <sup>b</sup>|14|1.85|
|CS-PAD [28]|LCLS<br>(8.3 keV)|194 _×_ 370 <sup>c</sup><br>(110<sup>2</sup>_×_ 500 )<br>|120|14|0.02|
|ePix100 [29]|LCLS|384 _×_ 352 <sup>d</sup>|120|14|0.03|
||(8.3 keV)|(50<sup>2</sup>_×_ 500)|(_≤_240)|||
|ePix10k|LCLS|384 _×_ 352 <sup>e</sup>|120|14|0.03|
||(8.3 keV)|(100<sup>2</sup>_×_ 500)|(_≤_10<sup>3</sup>)|||
|EIGER2|APS & others|1028 _×_ 512 <sup>f</sup><br>|2.25 k|16|2.37|
|(Dectris)||(75<sup>2</sup>_×_ 450)|(4.5 k)|(8)||
|HEXITEC [30]|DIAMOND<br>(2-200 keV)|80<sup>2</sup><br>(250<sup>2</sup>_×_ 1000 <sup>g</sup>)|6.3 - 8.9 k|14|0.07 - 0.1|
|Icarus [31]|NIF, Z|1024 _×_ 512<br>|_≥_250 M <sup>h</sup>|10|163,840|
|(Advanced|(0.7 - 10 keV)|(25<sup>2</sup>_×_ 25)||||
|hCMOS Sys.)||||||
|JUNGFRAU [32]|PSI<br>(12 keV)|1024 _×_ 512|2.2k|16|2.31|
|MM-PAD [33]|CHESS|128<sup>2</sup>|10 k/100 M <sup>i</sup>|14|0.3 - 2867|
||(_>_20 keV) <sup>j</sup>|(150<sup>2</sup>_×_ 500)||||
|SOPHIAS|SACLA|891 _×_ 2157<br>(30<sup>2</sup>_×_ 500)|60|12|0.17|
|HPV-X2 [34]|APS & others|400 _×_ 250|7.8 k/5 M <sup>k</sup>|10|0.98 - 625|
|(Shimadzu)|(10-40 keV)|(32<sup>2</sup>)||||
|Kraken [35]|NNSS|800 _×_ 800<br>(30<sup>2</sup>)|20 M <sup>l</sup>|12|19,200|
|MX170-HS|LCLS|3840<sup>2</sup>|2.5 <sup>m</sup>|16|0.07|
|(Rayonix)|(8-12 keV)|(44<sup>2</sup>)<br>||||
|PI MAX 4|APS|1024<sup>2</sup><br>|26 <sup>n</sup>|16|0.05|
|(Teledyne)|(10-40 keV)|(12.8<sup>2</sup>)||||



> a AGIPD is deployed as mega-pixel/voxel cameras through tiling. 

> b Burst mode for 352 stored frames. 

> c CS-PAD is deployed as tiled 2, 8, and 32 modules with up to 2.3 M voxels. 

> d ePix100 is deployed as tiled 4 modules with about 0.5 M voxels. 

- e ePix10K replaces CS-PAD, and is deployed as a single, or tiled 16 modules with about 2.2 M voxels. 

> f Eiger2 is deployed as a single, or tiled modules with more than 10 M voxels. 

> g Also 2 mm CdZnTe 

- h In burst mode for 4 frames. 

- i In burst mode for 8 frames. 

- j When using 750 _µ_ m CdTe as sensor 

- k In burst mode for 128 stored frames; or 10 M Hz frame rate and 256 stored frames possible by reducing the number of pixels by half. 

- l In burst mode for 8 frames. Read noise 157 _e−_ , Full Well 4.0 _×_ 105 _e−_ . Buttable to larger array 2 _×_ 2. 

- m Higher frame rate can be obtained through pixel binning, at 10 _×_ 10 binning, the frame rate increases to 120 Hz n Higher frame rate can be obtained through pixel binning, at 4 _×_ 4 binning, the frame rate increases to 95 Hz 

7 



![Figure](assets/figure_0003_page_0008.svg)Figure 3: Basic neural network architectures for (a) CNNs, (b) LSTMs, (c) encoder-decoders, and (d) GANs. 

and their superior nonlinear approximation capabilities to learn underlying structures and patterns within high-dimensional data [4]. In other words, DL aims to learn multiple levels of representation, corresponding to a hierarchy of features or concepts, where higher-level features are defined from lower-level ones and lower-level ones can help build up higher-level features. 

### **3.1 Neural Network Architectures** 

This section provides an overview of different popular deep neural network (DNN) architectures used for image processing tasks. These widely used architectures include but are not limited to convolutional neural networks (CNNs), long short term memory (LSTM), encoder-decoder networks, and generative adversarial networks (GANs). Due to space limitations, other DNN architectures such as transformers [36], restricted boltzamann machines [37], and extreme learning [38] will not be covered here. 

#### **3.1.1 Convolutional Neural Networks (CNNs)** 

CNNs are one of the most widely used architectures in DL, especially for image processing tasks, due to its inherenet spatial invariance property. The built-in convolutional layers allow the network to naturally reduce the high dimensionality of the input data, i.e. images, without information loss. Figure 3(a) shows the basic architecture of CNNs, which usually consists of 3 types of layers: i) convolutional layers, ii) pooling layers, and iii) fully connected layers. The convolutional layer uses various kernels to convolve the entire input image, including intermediate feature maps, and generate new feature maps. There are 3 major advantages of the convolutional operation [39]: i) the number of parameters is reduced by using weight sharing mechanisms, ii) the correlation among neighboring pixels are easily learned through local connectivity, and iii) the location of objects are fixed due to spatial invariance. Generally following a convolutional layer, the pooling layer is used to further reduce the dimensions of feature maps and network parameters. The average pooling and max pooling methods are commonly used, and their theoretical performances have been evaluated by [40] and [41], where max pooling is shown to achieve faster convergence and improved CNN performance. Lastly, the fully connected layers follows the last pooling or convolutional layer to convert the 2D feature maps into a 1D vector for additional feature mapping, i.e. labels. A few of the well known CNN models are the AlexNet [42], VGG [43], GoogLeNet [44], and ResNet [45], where all the models were top 3 finishers in the ImageNet Large 

8 



Scale Visual Recognition Challenge (ILSVRC). 

#### **3.1.2 Long-Short Term Memory (LSTM)** 

LSTMs [46] are a special type of recurrent neural network (RNN) that is commonly used to process sequential datasets, such as audio recordings, videos, and time-series data. Figure 3(b) shows the basic structure of a LSTM block, which consists of 3 gates (input, output, and forget gates) that regulate the stored memory and information flow within the block. The multiple gate architecture of LSTMs is specifically designed to capture long-term dependencies in the data as well as to avoid the vanishing gradient problem of vanilla RNNs [46]. Other LSTM architectures are derived from the basic architecture in Figure 3(b) such as LSTM without a forget gate, LSTM with peephole connections, the gated recurrent unit (GRU), and other variants [46]. 

#### **3.1.3 Encoder-Decoders** 

Encoder-decoder neural networks, also known as sequence-to-sequence networks, are a type of network that learns to map the input domain to a desired output domain [5]. As shown in Figure 3(c), the network consists of two main components: an encoder network which uses an encoder function _h_ = _f_ ( _x_ ) to compress the input _x_ into a latent-space representation _h_ , and a decoder network _y_ = _g_ ( _h_ ) that produces a reconstruction _y_ from _h_ . The latent-space representation _h_ prioritizes learning the important aspects of the input _x_ which are useful in reconstructing the output _y_ . A special case of encoder-decoder models, autoencoders are networks in which the input and output domains are the same. These networks are popularly used in DL applications involving sequence-to-sequence modeling such as natural language processing [47], image captioning [48], and speech recognition [49]. 

#### **3.1.4 Generative Adversarial Networks (GANs)** 

GANs [50] are increasingly popular DL frameworks for generative AI models. Classical GANs consist of 2 different networks, a generator and a discriminator, as shown in Figure 3(d). The generator network _G_ aims to generate data that is indistinguishable from the real data by learning a mapping from an input noise distribution _z_ to a target distribution _y_ . Meanwhile, the discriminator network _D_ takes as input the real and generated data, and aims to correctly classify them as “real” or “fake” (generated). The GAN learning objective takes on a game-theoretic approach as a two player minimax game between _G_ and _D_ . Let _L_ denote the loss function and the GAN objective as min _G_ max _D L_ ( _G, D_ ). Intuitively, _D_ aims to minimize its own classification error, which maximizes _L_ ( _G, D_ ). Meanwhile, _G_ aims to maximize the classification error of _D_ , which minimizes _L_ ( _G, D_ ). This adversarial loss function allows both models to be trained simultaneously and in competition with each other. Other GAN architectures are derived from the basic architecture in Figure 3(d) such as conditional GANs, GANs with inference models, and adversarial autoencoders [51]. 

### **3.2 Image Processing Techniques** 

This section provides an overview of several image processing tasks that have potential to be performed on edge devices. In addition, this section surveys different works that have applied the DL-based image processing techniques to radiographic image processing. 

9 



#### **3.2.1 Restoration** 

Image restoration is the process of adjusting the quality of digital images such that the enhanced image can facilitate further image analysis. Common enhancement operations include histogram-based equalization, brightness, and contrast adjustment. However, these operations are very elemental and advanced operations are necessary to further improve the perceptual quality. These advanced operations include image denoising, deblurring, and super-resolution (SR). 

**Denoising.** One of the fundamental challenges in image processing, image denoising aims to estimate the ground-truth image by suppressing internal and external noise factors such as sensor and environmental noise, as discussed in Section 2.3. Conventional methods including but not limited to adaptive nonlinear filters, Markov random field (MRF), and weighed nuclear norm minimization (WNNM), have achieved good performance in image denoising [52], however, they suffer from several drawbacks [53]. Two major drawbacks are the need to manually set parameters as the proposed methods are non-convex and the high computational cost for the optimization problem for the test phase. To overcome these challenges, DL methods are applied for image denoising problems to learn the underlying noise distribution. Various neural network architectures, such as CNNs, encoder-decoders, and GANs, have been proposed for image denoising in recent years; see [52] for details. 

On example application that uses image denoising is in X-ray computed tomography (CT). X-ray CT imaging is a common noninvasive imaging technique that allows for reconstructing the internal structure of objects by using 3D reconstruction from 2D projection images; see Section 3.2.5 on 3D reconstruction. The spatial resolution of CT images can range from tens of microns to a few nanometers, while higher resolutions can be obtained by using higher radiation doses. However, some experiments may require short exposure times or low radiation dosage to avoid damaging the sample. The low-dose image conditions results in noisy 2D projection images, which in turn impacts the quality of the 3D reconstructed image. To address this issue, [54] developed a GAN-based image denoising method called TomoGAN. TomoGAN is a conditional GAN model where the generator _G_ conditionally uses the noisy reconstruction as input and outputs enhanced (denoised) reconstructions. Furthermore, the generator network architecture adopts a modified U-Net [55] architecture, popularly used for image segmentation. Meanwhile, the discriminator _D_ is trained to classify reconstructions of the enhanced reconstructions and reconstructions of normal dose projections. [54] evaluates the effectiveness of TomoGAN on two experimental (shale sample) datasets. TomoGAN outperforms conventional methods in noise reduction and reports a higher structural similarity (SSIM) value. In addition, TomoGAN is demonstrated to be robust to images with dynamic features from faster experiments, e.g. collecting fewer projections and/or using shorter exposure times. 

Denoising has also been applied to synchrotron radiation CT (SR-CT) in a recent work by [56], which developed a CNN-based image denoising method called Sparse2Noise. Similar to the previous work for TomoGAN, this work presents a low-dose imaging strategy and utilizes paired normal-flux CT images (sparseview) and low-flux CT image (full-view) to train Sparse2Noise. In addition, Sparse2Noise also adopts a modified U-Net architecture for its performance of removing image degradation factors such as noise and ring artifacts. The Sparse2Noise network takes as input the normal-flux CT images into the modified U-Net architecture and outputs the enhanced image. During training, the network is trained in a supervised fashion using the low-flux CT images. The loss function to update the network weights is defined to minimize the difference between the enhanced image and the reconstructed low-flux CT image. [56] evaluates the effectiveness of Sparse2Noise on one simulated and two experimental datasets. Furthermore, Sparse2Noise is compared to simultaneous iterative reconstruction technique (SIRT), unsupervised deep imaging prior (DIP), and supervised training 

10 



algorithms Noise2Inverse [57] and Noise2Noise [58]. For the simulated dataset, Sparse2Noise outperforms all methods by achieving the highest SSIM and peak signal to noise ratio (PSNR) values, and in terms of removing image degradation factors such as noise and ring artifacts. For the experimental datasets, Sparse2Noise also achieves the best performance in terms of noise and ring artifact removal. Most importantly, however, Sparse2Noise can achieve excellent performance for low-dose experiments (0.5 Gy per scan). 

**Deblurring.** Image deblurring aims to recover a sharp image from a blurred image by suppressing blur factors such as lack of focus, camera shake, and target motion. Some blur factors are application specific such as multiple Coulomb scattering and chromatic aberration in proton radiography [59]. A blurred image can be modeled mathematically as _B_ = _K ∗ I_ + _N_ , where _B_ denotes the blurred image, _K_ the blur kernel, _I_ the sharp image, _N_ the additive noise, and _∗_ the convolution operation. The blur kernel _K_ is typically modeled as a convolution of blur kernels that are spatially invariant or varying [60]. Conventional methods aim to solve the inverse filtering problem to estimate _K_ , however, this is an ill-posed problem as the sharp image _I_ needs to be estimated as well. To address this issue, prior-based optimization approaches, also known as maximum a posteriori (MAP)-based approaches, have been proposed to define priors for _K_ and _I_ [61]. While these approaches are shown to achieve good results for image deblurring, deep learning approaches can further improve the accuracy of the blur kernel estimation or even skip the kernel estimation process altogether by using end-to-end methods. Various neural network architectures, such as CNNs, LSTMs, and GANs, have been proposed for image deblurring; see [60, 61] for details. 

One example application that uses image deblurring is in neutron imaging restoration (NIR), a nondestructive imaging method. However, the neutron images suffer from noise and blur artifacts due to the neutron source and the digital image system. The low quality of raw neutron images limits their applications in research, and thus image denoising and deblurring techniques are necessary to produce sharp images. To address these issues, [62] proposes a fast and lightweight neural network called DAUNet. DAUNet consists of three main blocks: a feature extraction block (FEB), multiple cascaded attention U-Net blocks (AUB), and a reconstruction block (RB). First, DAUNet takes as input a degraded neutron image and feeds it into the FEB to extract important underlying features. Next, the AUB inputs the extracted feature maps into a modified U-Net with an attention mechanism, which allows U-Net to focus on harder to address features such as texture and structure information, and outputs a restored image. Last, the RB block outputs the enhanced image by reconstructing the restored image. To evaluate DAUNet, its performance is compared with several popular DNN image restoration methods such as DnCNN [63] and RDUNet [64]. Due to the lack of available neutron imaging datasets, the networks are trained on X-ray images that are similar to the neutron imaging principle; specifically, the X-ray images are obtained from the SIXray dataset [65], where 4699 and 23 images are used as the training and test set respectively. In addition, seven clean neutron images are added to the test set. Results show that DAUNet can effectively improve the image quality by removing noise and blurring artifacts, while achieving quality close to the large network with faster running times and a smaller number of network parameters. 

**Super-resolution (SR).** Image SR is the process of reconstructing high-resolution images from lowresolution images. It has been widely applied in many real-world applications, especially in medical imaging [66] and surveillance [67], where the spatial resolution of captured images are not sufficient due to limitations such as hardware and imaging conditions. A variety of DL-based methods for SR have been explored, ranging from CNN-based methods (e.g SRCNN [68]) to more recent GAN-based methods (e.g. SRGAN [69]). In addition to utilizing different neural network architectures, DL-based SR algorithms also differ in other major aspects such as their loss functions and training approaches [70]. These differences result from various factors 

11 



that contribute to the degradation of image quality including but not limited to blurring, sensor noise, and compression artifacts. Intuitively, one can think of the low-resolution image as the output of a degradation function with an input high-quality image. In the most general case, the degradation function is unknown and an approximate mapping is learned through deep learning. These degradation factors influence the design of loss function, and thus training approaches. A detailed discussion of the various loss functions, SR network architectures, and learning frameworks is out of scope for this paper; however, see [70] for details. 

An example application that applies super resolution is for X-ray CT imaging. As mentioned earlier, CT imaging has many factors that impact the resulting image quality such as radiation dose and slice thickness. In addition, 3D image reconstruction may require heavy computational power due to the number of slices or projection views taken, where thicker slices results in lower image resolution, and slower operational speed, which increases with the number of slices. To address this issue, it is desirable to obtain higher-resolution (thin-slice) images from low-resolution (thick-slice) ones. [71] develops an end-to-end super-resolution method based on a modified U-Net. The network takes as input the low-resolution image and outputs the highresolution one. The network is trained on slices of brain CT images obtained from a 65 clinical positron emission tomography (PET)/CT studies for Parkinson’s disease. The low-resolution images are generated as the moving average of five high resolution slices and the ground-truth image is taken as the middle slice. The performance of the proposed method is compared with the Richardson-Lucy (RL) deblurring algorithm using the PSNR and normalized root mean square error (NRMSE) metrics. The results show that the proposed method achieves the highest PSNR and lowest NRMSE values compared to the RL algorithm. In addition, the noise level of the enhanced images are reported to be lower than that of the ground-truth. 

#### **3.2.2 Segmentation** 

Image segmentation is the process which segments an image or video frames into multiple regions or clusters, where each pixel can be represented by a mask or be assigned a class [72]. This task is essential in a broad range of computer vision problems, especially for visual understanding systems. A few applications that utilize image segmentation include but are not limited to medical imaging for organ and tumor localization [73], autonomous vehicles for surface and object detection, and video footage for object and people detection and tracking [74]. Numerous techniques for image segmentation have been proposed throughout the years, ranging from early techniques based on region splitting or merging such as thresholding and clustering algorithms, to newer algorithms based on active contours and level sets such as graph cuts and Markov random fields [72, 75]. Although these conventional methods have achieved acceptable performance for some applications, image segmentation still remains a challenging task due to various image degradation factors such as noise, blur, and contrast. To address these issue, numerous deep learning methods have been developed and have been shown to achieve remarkable performance. This is due to the powerful feature learning capabilities of DNNs, which allows DNNs to have reduced sensitivity to image degradation factors compared to the conventional methods. Popular neural network architectures used for DL-based segmentation includes CNNs, encoder-decoder models, and multiscale architectures; see [75] for details. Two popular DNN architectures used for image segmentation problems are U-Net [55] and SegNet [74]. 

Image segmentation is an important step in analyzing X-ray radiographs from, for example, inertial confinement fusion (ICF) experiments [76]. ICF experiments typically use single or double shell targets which are imploded as the laser energy or laser-induced X-rays rapidly compress the target surface. X-ray and neutron radiographs of the target provide insight to the shape of target shells during the implosion. Contour extraction methods are used to extract the shell shape to conduct shot diagnostics such as quantifying the 

12 



implosion and kinetic energy, identifying shell shape asymmetries, and determining instability information. [76] uses U-Net [55], a CNN architecture for image segmentation, to output a binary masked image of the outer shell in ICF images. The shell contour is then extracted from the masked image using edge detection and shape extraction methods. Due to the limited number of actual ICF images, a synthetic dataset consisting of 2000 experimental-like radiographs is used to train the U-Net. In addition, the synthetic dataset provides ground-truth ICF image-mask pairs, which are required to train U-Net. The trained U-Net is tested on experimental images and has successfully extracted the binary mask of high-signal-to-noise ratio ICF images. 

Another example of X-ray image segmentation is for the Magnetized Linear Inertial Fusion (MagLIF) experiments at Sandia National Laboratory’s Z-facility [77]. The MagLIF experiments compresses a cylindrical beryllium tube, also known as a liner, filled with pure deuterium fuel using a very large electric current on the order of _O_ (20 _MA_ ). Before compression, the deuterium fuel is pre-heated and an axially oriented magnetic field is applied. The electric current causes the liner to implode and compresses the deuterium fuel in a quasi-adiabatic implosion. The magnetic field flux is also compressed which aids in the trapping of charged fusion particles at stagnation. X-ray radiographs are taken during the implosion process for diagnostics and to analyze the resulting plasma conditions and liner shape. To better analyze the implosion, a CNN model is proposed to segment the captured X-ray images into fuel strand and background. The CNN is trained using synthetically generated and augmented dataset of 10,000 X-ray images and their corresponding binary masks. The trained CNN is tested on experimental images where the results generally demonstrate excellent fuel-background segmentation performance. The worst segmentation performance is due to factors such as excessive background noise and X-ray image plate damage. 

#### **3.2.3 Compression** 

Image compression is the process of reducing the file size, in bytes, without reducing the quality of the image below a threshold. This process is important in order to save memory storage space and to reduce the memory bandwidth to transmit data, especially for running image processing algorithms on edge devices. The fundamental principle of compression is to reduce spatial and visual redundancies in images, by exploiting interpixel, psycho-visual, and coding redundancies. Conventional methods commonly leverage various quantization and entropy coding techniques [78]. Popularly used conventional methods for lossy and lossless compression includes but are not limited to JPEG [79], JPEG2000, wavelet, and PNG. While conventional methods are widely used for both image and video compression, their performance is not the most optimal for all types of image and video applications. DL approaches can achieve improved compression results due to several factors. DNNs can learn non-linear mappings to capture the compression process as well as extract the important underlying features of the image through dimensionality reduction. For example, an encoder network or CNN can extract important features into a latent feature space for compact representation. In addition, DNNs can implement direct end-to-end methods using networks such as encoder-decoders to directly obtain the compressed image from an input sharp image. Furthermore, once a DNN is trained, the inference time is much faster. For DL-based image compression methods, the most commonly used neural network architectures are CNNs, encoder-decoders, and GANs [78]. 

#### **3.2.4 Sparse Sampling** 

A closely related process to image compression is sparse sampling. While compression aims to reduce the file size, sparse sampling, also known as compressed sensing (CS), aims to efficiently acquire and reconstruct a signal by solving underdetermined linear systems. It has been shown in CS theory that a signal can be 

13 



A Scanning probe positions BStitched PtychoNN inference individual probe intensity individual inference 

![Figure](assets/figure_0004_page_0014.svg)Figure 4: Sparse-sampled single-shot ptychography reconstruction using PtychoNN. (a) scanning probe positions with minimal overlap. (b) Single-shot PtychoNN predictions on 25 _×_ sub-sampled data compared to (c) ePIE reconstruction of the full resolution dataset. 

recovered from sampling fewer measurements than required by the Niquist-Shannon sampling theorem [80]. As a result, both memory storage space and data transmission bandwidth can be reduced. In conventional methods, CS algorithms need to overcome two main challenges: the design of the sampling and reconstruction matrices. Numerous methods have been proposed including but no limited to random and binary sampling matrices and reconstruction methods using convex-optimization and greedy algorithms [81]. However, these conventional methods suffer from long computational times or low quality reconstruction. DL approaches allow for fast inference (reconstruction) times for a trained network, as well as learning non-linear functions for higher quality signal reconstruction [81, 82]. 

Neural network (NN) models that learn to invert X-ray data have also been shown to significantly reduce the sampling requirements faced by traditional iterative approaches. For example, in ptychography, traditional iterative phase retrieval methods require at least 50% overlap between adjacent scan positions to successfully reconstruct sample images as required by Nyquist-Shannon sampling. In contrast, Figure 4(b) shows image reconstructions obtained from PtychoNN when sampled at 25 _×_ less than required for conventional phase retrieval methods [83]. Figure 4(a) shows the probe positions and intensities, there is minimal overlap between probes. Through use of inductive bias provided through online training of the network [84], PtychoNN is able to reproduce most of the features seen in the sample even when provided extremely sparse data. Figure 4(c) shows the same region reconstructed using an oversampled dataset and traditional iterative phase retrieval. Furthermore, [84] demonstrated live inference performance during a real experiment using an edge device and running the detector at its maximum frame rate of 2 kHz. 

In the previous example, DL is used to reduce sampling requirements but not to alter the sampling strategy. In other words, the scan proceeds using a conventional acquisition strategy, but using fewer points along that trajectory than traditionally required. In contrast, active learning approaches are being developed that use data-driven priors to direct the acquisition strategy. Typically, this is treated as a Bayesian optimization (BO) problem using Gaussian processes (GPs). This method has been applied to a variety of characterization modalities including scanning probe microscopy [86], X-ray scattering [87], and neutron characterization [88]. A downside to such approaches is that the computational complexity typically increases as _O_ ( _N_<sup>3</sup> ) with the action space [89], making real-time decision a challenge. To address these scaling limitations which are critical especially in fast scanning instruments, recent work has demonstrated the use of pre-trained NNs to make 

14 



![Figure](assets/figure_0005_page_0015.svg)Figure 5: FAST framework for autonomous experimentation. Top panel shows the workflow that enables real-time steering of scanning microscopy experiments. Bottom panel shows reconstructed images at 5%, 15% and 20% sampling along with the corresponding locations from which they were sampled. In addition, the full-grid pointwise scan and corresponding points sampled between 15-20% is also shown. Reproduced with permission from [85]. 

such control decisions [90, 85]. Figure 5 shows the workflow and results from the Fast Autonomous Scanning Toolkit applied to a scanning diffraction X-ray microscopy measurement of a WSe2 sample. Starting from some quasi-random initial measurements, FAST generates an estimate of the sample morphology, predicts the next batch of 50 points to sample from, triggers acquisition on the instrument, analyzes the image after the next set of points has been acquired and continues the process until the improvement in sample image is minimal. Bottom panel of Figure 5, A,C,and E show the predicted image after 5%, 10% and 15% sampling while Figure 5 B, D, and F shows the points preferentially selected by the AI. The AI has learned to prioritize acquisition where the expected information gain is maximum, e.g. around contrast features on the sample. 

#### **3.2.5 3D Reconstruction** 

Image-based 3D reconstruction is the process of inferring a 3D structure from a single or multiple 2D images, and is a common topic in the fields of computer vision, medical imaging, and virtual reality. This problem is well known to be an ill-posed inverse problem. Conventional methods attempt to formulate a mathematical formula for the 3D to 2D projection process, use prior models, 2D annotations, and other techniques [91, 92]. In addition, high quality reconstruction typically requires 2D projections from multiple views or angles, which may be difficult to calibrate (i.e. cameras) or time consuming to obtain (i.e. CT) depending on the application. DL techniques and the increasing availability of large datasets motivates new advances in 3D reconstruction by address challenges found in conventional methods. The popular networks used for image-based 3D reconstruction are CNNs, encoder-decoder, and GAN models [91]. 

X-ray phase information is now available for 3D reconstruction in the state-of-the-art X-ray sources such 

15 



![Figure](assets/figure_0006_page_0016.svg)Figure 6: Comparison of 3D sample images obtained by (a) phase retrieval, (b) AutoPhaseNN, and (c) AutoPhaseNN + phase retrieval. Reproduced with permission from [95]. 

as synchrotrons and XFELs. In contrast to iterative phase retrieval methods that incorporate NNs through a DIP or other means, single-shot phase retrieval NNs provide sample images from a single pass through a trained NN. The inference time on a trained NN is minimal and such methods are hundreds of times faster than conventional phase retrieval [93, 94]. Figure 6(b) and 6(a) compare AutoPhaseNN and traditional phase retrieval for 3D coherent image reconstruction, respectively [95]. AutoPhaseNN is trained to invert 3D coherently scattered data into sample image in a single shot. Once trained AutoPhaseNN is _>_ 100 _×_ faster that iterative phase retrieval with some reduction in accuracy. The prediction from AutoPhaseNN can also be used to seed phase retrieval, i.e. provide an initial estimate which can be refined by a few iterations of phase retrieval. This combined approach of NN + phase retrieval is shown to be both faster and more accurate than iterative phase retrieval. 

A recent work by [96] developed an adaptive CNN-based 3D reconstruction method for coherent diffraction imaging (CDI), a non-destructive X-ray imaging technique that provides 3D measurements of electron density with nanometer resolution. The CDI detectors record only the intensity of the complex diffraction pattern of the incident object. However, all phase information is lost in this detection method, and thus results in an ill-posed inverse Fourier transform problem to obtain the 3D electron density. Conventional methods encounter many challenges including expert knowledge, sensitivity to small variations, and heavy computation requirements. While DL methods currently cannot completely substitute conventional methods, they can speed up the 3D reconstruction speed given an initial guess, and can be fine-tuned using conventional methods to achieve better performances. For CDI 3D reconstruction, [96] proposes a 3D CNN architecture with model-independent adaptive feedback agents. The network takes in 3D diffracted intensities as inputs and outputs a vector of spherical harmonic coefficients, which describe the surface of the 3D incident object. The adaptive feedback agents take as input the spherical harmonics to adaptively adjust the intensities, positions, and decay rates of a collection of radial basis functions. The 3DCNN is trained using a synthetic dataset consisting of 500,000 training set of 49 sampling coefficients as well as the spherical surface and volume of each in order to perform a 3D Fourier transform. An additional 100 random 3D shapes and their corresponding 3D Fourier transforms are used to test the adaptive model-independent feedback algorithm, with the CNN 

16 



![Figure](assets/figure_0007_page_0017.svg)Figure 7: Basic models of the (a) temporal, (b) CPU, (c) GPU, (d) spatial, (e) FPGA, and (f) ASIC architectures. 

output as its initial guess. Last, the robustness of the trained 3DCNN is tested on the experimental data of a 3D grain from a polycrystalline copper sample measured using high-energy diffraction microscopy. Results show that the 3DCNN provides an initial guess that captures the average size and a rough estimate of the shape of the grain. The adaptive feedback algorithm uses the 3DCNN initial guess to fine-tune the harmonic coefficients to match and converge the generated and measured diffraction patterns of the grain. 

## **4 Hardware Solutions for Deep Learning** 

Deep neural networks (DNNs) have been implemented for many imaging processing tasks ranging from enhancement to generation as discussed above. To achieve good performance, these algorithms use very deep networks which requires heavy computational power during training and inference. For very large networks such as AlexNet, a single forward pass may require millions of multiply and accumulate (MAC) operations, thus making DNNs both computationally and energy costly. For real-time data processing in imaging devices, DNN algorithms need to be executed with low latency, limited energy, and other design constraints. Hence, there is a need to develop cost and energy efficient hardware solutions for DL applications. 

Interestingly, neural network algorithms are known to have at least two types of inherent parallelism, namely model and data parallelism [97]. Model parallelism refers to the partitioning of the neural network weights for MAC operations for parallel execution as there are no data dependencies. Data parallelism refers to processing the data samples in batches rather than a single sample at a time. Hardware accelerators can exploit these characteristics by implementing parallel computing paradigms. This section presents different hardware accelerators used for DL applications. Note that the best hardware solution is dependent on the application and corresponding design requirements. For example, edge computing devices such as cameras and sensors may require small chip area with limited power consumption. 

17 



### **4.1 Electronic-based Accelerators** 

The electronic-based hardware solutions for DL are broad, ranging from general purpose processors such as central processing units (CPUs) and graphical processing units (GPUs), field-programmable gate arrays (FPGAs), to application-specific integrated circuits (ASICs). The circuit architecture design typically follows either temporal or spatial architectures [98] as shown in Figure 7(a) and 7(d). The architectures are similar in using multiple processing elements (PEs) for parallel computing, however, there are differences in control, memory, and communication. The temporal architecture features a centralized control for simple PEs, consisting of only arithmetic logic units (ALUs), which can only access data from the centralized memory. Meanwhile, the spatial architecture features a decentralized control scheme with complex PEs, where each unit can have its own local memory or register file (RF), ALU, and control logic. The decentralized control scheme forms interconnections between neighboring PEs to exchange data directly, allowing for dataflow processing techniques. 

#### **4.1.1 Temporal Architectures: CPUs and GPUs** 

CPUs and GPUs are general purpose processors that typically adopt the temporal architecture as shown in Figures 7(b) and 7(c). Modern CPUs can be realized as vector processors, which adopt the single-instruction multiple-data (SIMD) model to process a single instruction on multiple ALUs simultaneously. In addition, CPUs are optimized for instruction-level parallelism in order to accelerate the execution time of serial algorithms and programs. Meanwhile, modern GPUs adopt the single-instruction multiple threads (SIMT) model to process a single instruction across multiple threads or cores. Different from CPUs, GPUs are made up of more specialized, parallel, and smaller cores than CPUs to efficiently process vector data with high performance and reduced latency. As a result, GPU optimization relies on software defined parallelism rather than instruction-level parallelism [99]. Both the SIMD and SIMT execution models for CPUs and GPUs, respectively, allow for parallel MAC operations for accelerated computations. 

Nonetheless, CPUs are not the most used processor for DNN training and inference. Compared to GPUs, CPUs have a limited number of cores, and thus a limited number of parallel executions. For example, one of Intel’s server-grade CPUs is the Intel Xeon Platinum 8280 processor which can have up to 28 cores, 56 threads, 131.12 GB/s maximum memory bandwidth, and 2190 Giga-floating point operations per second (GFLOPS) for single-precision compute power. In comparison, NVIDIA’s GeForce RTX 2080 Ti is a desktop-grade GPU with 4352 CUDA cores, 616.0 GB/s memory bandwidth, and 13450 GFLOPS single-precision compute power. 

For DL at the edge, the hardware industry has developed embedded platforms for AI. One popular platform is the NVIDIA Jetson for next generation embedded computing. The Jetson processor features a heterogeneous CPU-GPU architecture [100] where the CPU accelerates the serial instructions and the GPU accelerates the parallel neural network computation. Furthermore, the Jetson is designed with a small form factor, size, and power consumption. A broad survey by [101] presents different works using the Jetson platform for DL applications such as medical, robotics, and speech recognition. Several surveyed works have used the Jetson platform to implement imaging processing tasks including segmentation, object detection, and classification. 

Also using the NVIDIA Jetson platform, a work by [102] investigates the performance of the Jetson TX2 for edge deployment for TomoGAN [54], an image denoising technique using general adversarial networks (GANs) for low-dose X-ray images. The training and testing datasets consist of 1024 pairs of images of size 1024 _×_ 1024 with each image pair consisting of a noisy image and its corresponding ground truth. The pre-trained TomoGAN network is deployed and tested on the Jetson TX2 and a laptop with an Intel Core 

18 



i7-6700HQ CPU @2.60GHz with 32GB RAM. The laptop CPU achieves an average inference performance of 1.537 seconds per image, while the TX2 achieves an inference performance of 0.88 seconds per image, approximately 1 _._ 7 _×_ faster than the laptop CPU. 

A recent work by [103] investigates the classification accuracy of tuberculosis detection from chest X-ray images using MobileNet [104], ShuffleNet [105], SqueezeNet [106], and their proposed E-TBNet. In addition, they further investigate the inference time during testing of each network on the NVIDIA Jetson Xavier and a laptop with Intel Core i5-9600KF CPU and NVIDIA Titan V GPU. The dataset consists of 800 chest X-ray images scaled to size 512 _×_ 512 _×_ 3. The MobileNet network achieves the highest accuracy at 90% while their proposed E-TBNet achieves 85%. However, the inference time for E-TBNet is the fastest for all investigated networks with an inference time of 0.3 ms and 3 ms per image when deployed on the laptop with Titian GPU and Jetson Xavier, respectively. The slowest reported inference time for the Jetson Xavier is 6 ms per image for the ShuffleNet. Although the inference time for the Xavier is an order of magnitude slower, classification inference can be achieved in real-time with smaller hardware footprint for edge deployment. 

#### **4.1.2 Spatial Architectures: FPGAs and ASICs** 

Field-programmable gate arrays (FPGAs) and application-specific integrated circuits (ASICs) typically adopt the spatial architecture as shown in Figure 7(e) and 7(f). FPGAs and ASICs are specialized hardware that are tailored for specific applications due to their design process. FPGAs can be configured to perform any function as it is made up of programmable logic modules and interconnecting switches as shown in Figure 7(e). The FPGA software is used to directly build the logic and data flow directly into the chip architecture. On the other hand, ASICs are designed and optimized for a single application, and cannot be reconfigured. Nonetheless, the spatial architecture of FPGAs and ASICs makes them well suited for neural network computations as the mathematical operations of each layer are fixed and known a priori. As a result, FPGAs and ASICs can attain highly optimized performance. 

As shown in Figure 7(d), the spatial architecture consists of an array of PEs interconnected with a Network-on-Chip (NoC) design, allowing for custom data flow schemes. Although not shown in Figure 7(d), the memory hierarchy consists of three levels. The lowest level consists of the RF in each PE, which is used to locally store data for inter-PE data movement or local accumulation operations. The middle level consists of a global buffer (GB) that holds the neural network weights and inputs to feed the PEs. The highest level is the off-chip memory, usually a DRAM, to store the weights and activations of the whole network. MAC operations need to be performed on large data sets. Hence, the major bottleneck is the high latency and energy costs of DRAM accesses. A comparison between DianNao and Cambricon-X, two CNN accelerators, show that DRAM accesses consume more that 80% of the total energy consumption [107]. In addition, [108] reports that the energy cost of DRAM access is approximately 200 _×_ more than a RF access. Therefore, energy efficiency can be greatly improved through the reduction of DRAM accesses, commonly done by exploiting the idea of data reuse. 

The focus of data reuse is to utilize the data already stored in RFs and the GB as often as possible. This gives rise to the investigations of efficient data flow paradigms in both the spatial and temporal operations of PEs. For example, in fully connected layers, the input reuse scheme is popular since the input vector is dot multiplied by each row of the weight matrix to compute the layer output. For convolutional layers, the weight reuse scheme is popular as the weight kernel matrix is used for multiple subsets of the input feature map. In addition for convolutional layers, convolutional reuse can be applied by exploiting the overlapping region of the sliding window of kernel weights and the input feature map. Additional data reuse schemes are the weight 

19 



stationary, output stationary, row stationary, and no local reuse schemes. A detailed discussion of the data reuse schemes is out of scope for this paper. However, for a comprehensive review, see details in [98, 109, 110]. In summary, optimizing the data flow is crucial for FPGAs and ASICs to attain high energy efficiency. 

The energy efficiency and massive parallelism of FPGA and ASIC-based accelerators make them desirable for edge computing. A recent work [111] develops a lightweight CNN architecture called SparkNet for image classification tasks. SparkNet features approximately 3 _×_ less parameters compared with the SqueezeNet, and approximately 150 _×_ less parameters than AlexNet. In addition, a comprehensive design is presented to map all layers of the network onto an Intel Arria 10 GX1150 FPGA platform with each layer mapped to a its own hardware unit to achieve simultaneously pipelined work, increasing throughput. SparkNet is tested on 4 benchmark image classification datasets, i.e. MINIST, CIFAR-10, CIFAR-100 and SVHN. The performance and average time for the Intel FPGA, NVIDIA Titan X GPU, and Intel Xeon E5 CPU to process 10,000 32 _×_ 32 _×_ 3 is reported. The FPGA-based accelerator achieves a processing time of 11.18 µs, which is 41 _×_ and 9 _×_ faster than the CPU and GPU, respectively. Furthermore, the FPGA average power consumption is 7.58 W with a performance of 337.2 Giga operations per second (GOP/s), making the FPGA more energy and computationally efficient compared to the CPU (95 W, 8.2 GOP/s) and GPU (250 W, 39.4 GOP/s). 

Another recent work [112] uses FGPAs to deploy MobileNet for face recognition in a video-based face tracking system. The work further integrates the FPGA with CPUs and GPUs to build a heterogeneous system with a delay-aware energy-efficient scheduling algorithm to achieve reduced execution time, latency, and energy cost. The face tracking experiment is run using an Intel Gold 5118 CPU, NVIDIA Tesla P100 GPU, and the Intel Arria 10 GX 900 and Intel Stratix 10 GX1100 FGPAs. The reported experimental results evaluate the computing speed and power efficiency of the FPGA-based accelerator compared to the CPU and GPU, as well as the efficiency of the combined detection system with CPU/GPU/FPGA. The FPGA accelerators achieve a computational speed that is approximate to or better than the GPU, while achieving superior power efficiency in GOP/s/W. The difference in performance of the FPGAs is due to their hardware specifications, where the hardware richer Intel Stratix will out perform the Intel Arria. Lastly, the experimental results report that the CPU/GPU/FPGA system can achieve optimal performance in comparison to using only one or a combination of two different accelerators. This is due to the energy efficient scheduling algorithm to optimally pipeline tasks to the different accelerators. The idea of utilizing a heterogeneous system and scheduling algorithm to improve computational and energy efficiency can be explored to address challenges of edge computing. 

The Google Edge Tensor Processing Unit (TPU) platform [113] is a general purpose ASIC designed and built by Google for inference at the edge. One example product is the Dual Edge TPU which features an area footprint of 22 _×_ 30 mm<sup>2</sup> , peak perfrmance of 8 trillion operations per second (TOPS), and power consumption of 2 TOPS/W. Other hardware options are available for ASIC prototyping and deployment for edge devices. A survey by [114] presents works that use the Edge TPU platform for DL applications such as image classification, object detection, and image segmentation. 

The previously discussed work [102], which deployed TomoGAN on the NVIDIA Jetson platform for X-ray image denoising, also deployed it on the Edge TPU. The work presents a quantized model of TomoGAN to address limitations of the Edge TPU, such as output size. A fine tuning model is also presented to improve the output quality of the quantized model. The Edge TPU’s average inference time is 0.554 s per image, which is faster than the Jetson TX2 inference time of 0.88 s per image. In addition, the power consumption is reduced to 2 W compared to Jetson TX2’s 7.5 W. 

In addition, there is interest in the development of software and firmware for modular and scalable 

20 



![Figure](assets/figure_0008_page_0021.svg)Figure 8: Scalable CARIBOu architecture for data readout. Adopted from [115] with permission. 

implementation of energy efficient algorithms on FPGA platforms. One such example is for high-speed readout systems for pixel detectors. Oak Ridge National Laboratory (ORNL), through the support of the Department of Energy (DOE) in High Energy Physics (HEP) and Nuclear Physics (NP), is leading the design of a new generic readout system for pixel detectors based on the successful first-generation system, the CARIBOu 2.0 [115]. The CARIBOu 2.0 system, shown in Figure 8, will be the proposed architecture for the platform. The concept of the system is to provide a generic framework for the readout of ASIC detectors for research and development and scalable to larger detector arrays. CARIBou 2.0 shares knowledge and code to provide the community with a convenient platform that maximizes reusability and minimizes overhead when developing such systems. ORNL will initially implement the readout firmware and software specific to the Timepix4 or to commercial CMOS image sensors, SMALLGAD, Photon-to-Digital Converters (PDCs), and the interconnect for the assemblies. The hardware platform is based on Xilinx Ultrascale+ FPGA, that provides resources for CPU and FPGA side data processing at high speed. Using the resources of this modern FPGA, software and firmware will be developed to flexibly implement data processing and reduction, edge computing, by using conventional and ML algorithms running in the FPGA. For larger data rates, firmware will be developed to move the data to a FELIX card, which can handle up to 24 CARIBOu 2.0 systems and transmit data via a high-speed network interface to a data center or process them locally via GPU and CPU in the FELIX host machine. As a result, the system can be scaled up to the readout of large smart sensor stack arrays. 

Furthermore, as advancements in ASIC technology have enabled greater integration of digital functionalities for scientific applications, there has been growing interest in incorporating compression capabilities directly within ASIC detectors to enhance data processing speed. ASIC architectures capable of frame rates approaching 1 MHz have been designed, providing a viable solution for enhancing the speed of various diffraction techniques employed at X-ray light sources, including those relying on coherent imaging methodologies like ptychography [116]. Developing ASIC compression strategies that exploit the structure in detector data enables high compression performance while requiring lower computational complexity than commonly used lossless compression methods like LZ4 [116]. 

21 



![Figure](assets/figure_0009_page_0022.svg)Figure 9: (a) Summary of electronic-based hardware comparison. (b) 50 years of CPU processor trend data from [117]. 

#### **4.1.3 Summary and Limitations** 

We have presented an overview of 4 different electronic-based accelerators and a few works applying them to DL applications at the edge. Figure 9(a) shows that there is a clear trade-off between programmability and efficiency. To attain higher performance and power efficiency, FPGAs and ASICs require more design complexity to optimize data flow, while ASICs need further hardware optimization. Correspondingly, the time-to-market increases with design complexity. DL algorithms can be deployed at the edge using these existing electronic-based hardware accelerators. However, in recent years, these electronic-based accelerators are constantly reaching performance limits in latency, energy consumption, high interconnect cost, excessive heat, and other physical constraints [118]. Figure 9(b) illustrates the past 50 years of CPU trends in regards to the number of transistors, single-thread performance, frequency, typical power consumption, and number of cores. The trends show that the number of transistors and correspondingly the power consumption continues to grow. As of November 2022, Apple’s M1 Ultra chip has the largest number of transistors on a commercial processor at 114 billion. Furthermore, the trends indicate that CPU clock frequency has plateaued since around 2005 while single-thread performance and number of cores are slowly tapering. In addition, electronic accelerators are traditionally designed to follow von Neumann architecture where the processor and memory units are connected by buses [119], which inherently increases data transfer and power consumption during computation. [107] demonstrates that more than 75% of the energy utilized by processors comes from DRAM accesses. These limits in electronic based computing gives rise to a shift in focus to analog neuromorphic computing and non-von Neumann architectures such as optical neural networks and bio-inspired spiking nerual networks for high-speed, energy-efficient, and parallel computing [120, 121]. 

### **4.2 Optical Neural Networks** 

Optical neural networks (ONNs) have emerged as a promising avenue for achieving high-performance and energyefficient computing, given their compute-in-light speed, ultra-high parallelism, and near-zero computation energy [122, 123, 124, 125]. Series of photonic tensor cores (PTCs) are designed to enhance the execution of linear matrix operations, the fundamental operations in AI and signal processing, with coherent photonic integrated circuits [122], micro-ring resonators [126], photonic phase-change materials [127, 128, 129], and diffractive optics [130, 131, 132]. On the basis of the linear optical computing paradigms, ONNs have been 

22 



constructed for various machine learning tasks such as image classification [133, 134, 132], vowel recognition [122], and edge detection [135]. Photonic computing methods [136] also feature great potential for supporting advanced Transformer models. Furthermore, ONNs holds significant promise for real-time image processing, where they process image signals directly in light fields, as opposed to after digitalization [137, 138, 139, 140]. For instance, recent advancements include the proposal of an image sensor with an ONN encoder [137], which filters relevant information within a scene using an energy-efficient ONN decoder before detection by image sensors. 

Despite their advantages, PTCs face significant challenges related to cross-domain signal conversion energy overhead, specifically in analog-to-digital (A/D) and digital-to-analog (D/A) conversion, as well as scalability, particularly concerning chip area. For example, Mach-Zehnder interferometer (MZI)-based PTCs [122] require _O_ ( _m_<sup>2</sup> + _n_<sup>2</sup> ) bulky MZIs and approximately _∼_ ( _m_ + _n_ ) cascaded MZIs within a single optical path to implement an n-input, m-output layer. 

Efficient analog-to-digital conversion solution [141] and various hardware-software co-design methodologies [135] have been investigated to reduce signal conversion overhead by reducing precision and energy per conversion. 

In pursuit of enhancing the scalability and efficiency of ONNs, researchers have delved into innovative optimizations at both the architecture and device levels. One noteworthy approach at the architecture level is the introduction of optical subspace neural networks (OSNNs), which make a trade-off between weight representation universality and the reduction of optical component usage, area costs, and energy consumption. For example, a butterfly-style OSNN as shown in Figure 10(a), which achieved a remarkable reduction of 7 times in trainable optical components compared to GEMM-based ONNs, was reported and demonstrated a measured accuracy of 94.16% in image recognition tasks [133]. Without sacrificing much model expressiveness, OSNNs can reduce footprints, often ranging from one to several orders of magnitude less than previous MZI-based ONN [122]. 

At the device level, employing compact custom-designed PTCs, such as multi-operand optical neurons (MOON) [142, 143, 134], enables the consolidation of matrix operations into arrays of optical components. Figure 10(b) and 10(c) shows a customized multi-operand MZI-based and microring resonator-based PTCs, respectively. Instead of performing a single math operation (e.g., scalar product) per device, MOON fuses a tensor operation in the single device. Crucially, this approach retains the capability to represent general matrices while still maintaining an exceptionally compact layout, in contrast to prior compact tensor designs like star couplers and metasurfaces [144]. One specific achievement in MOON is the development of multi-operand MZI-based (MOMZI) ONN [143], which has realized a two-orders-of-magnitude reduction in propagation loss, delay, and total footprint without losing matrix expressivity. The customized ONN demonstrated an 85.89% measured accuracy in the street view house number (SVHN) recognition dataset with 4-bit control precision. The combined progress in architecture, device design, and optimization techniques is pivotal in advancing the capabilities of ONNs, making themselves efficient, scalable, and practical for AI applications. 

### **4.3 Spiking Neural Networks** 

In addition to photonic neuromorphic computing, extensive research has been done for other neuromorphic computing architectures. Due to the bottleneck seen in von Neumann architectures, these computing paradigms aim to greatly reduce data movement between memory and PEs to attain high energy efficiency and parallel processing. Taking a unique approach to improve energy efficiency, neuromorphic computing architectures are inspired by the human brain’s neurons and synapses. The human brain is extremely energy efficient, where 

23 



![Figure](assets/figure_0010_page_0024.svg)Figure 10: Integrated photonic chips for optical neural networks. (a) a butterfly-style PTCs to reduce the opitcal components from an architecture level [133]. (b) and (c) are customized multi-operand MZI-based and microring resonator-based PTCs, respectively, which improve scalability and efficiency at the device level [142, 143, 134]. 

![Figure](assets/figure_0011_page_0024.svg)Figure 11: A comparison among (a) a biological neuron, (b) artificial network neuron, and (c) spiking network neuron. 

in terms of computing terminology, it is estimated to have a computing power of 1 exaFLOPS while only consuming 20 W. In recent years, there is a rise in interest to explore brain-inspired neural network computing architectures, better known as spiking neural networks (SNNs) [145]. 

SNNs are a special type of artificial neural network (ANN) that closely mimics biological neural networks. While ANNs are traditionally modeled after the brain, there are still many fundamental differences between them such as neuron computation and learning rules. In addition, one major difference is the propagation of information between neurons. Biological neurons, shown in Figure 11(a), transmit information to downstream neurons using a spike train of signals, or a time-series of delta functions. The individual spikes (delta functions) are known to be sparse in time and have high information content. Therefore, SNNs are designed to convey information by utilizing the spike timings and spike rates [146, 147] as shown in Figure 11(c). Furthermore, the advantages of the spiking event sparsity can be exploited in special hardware to reduce energy consumption while maintaining the transmission of high information content [148]. 

The hardware industry as well as academia are striving to develop unique solutions for neuromorphic computing chips. Intel’s Loihi [149] features 128 neuromorphic cores with 1024 spiking neural units per core. 

24 



A recent work [150] surveys different works that utilize Loihi as a computing platform for applications such as event-based sensing and perception, odor recognition, closed-loop control for robotics, and simultaneous localization and mapping. For medical image analysis, [151] uses Loihi to implement a SNN for brain cancer MRI image classification. IBM developed TrueNorth [152], a neurmorphic chip featuring 4096 neuromorphic cores, 1 million spiking neurons and 256 million synapses. A work by [153] uses the TrueNorth computing platform to detect and count cars from input images by mapping CNNs, such as AlexNet and VGG-16, onto TrueNorth. A few other well-known SNN hardwares are Neurogrid [154], BrainScaleS [155], and SpiNNaker [156], which all adopt different solutions to emulate spiking neurons. For a comprehensive review, see details in [157, 158]. 

Due to its low power consumption, SNN hardware is a potential platform for edge computing. A work by [159] presents preliminary results for implementing SNN on a mixed analog digital memresistive hardware for classifying neutrino scattering data collected at Fermi National Accelerator Laboratory using the MINERvA detector [160]. Two different SNNs, the neuroscience-inspired dynamic architecture (NIDA) [161] and a memresistive dynamic adaptive neural network array (mrDANNA) [162], were trained and tested on the MINERvA dataset’s X view. The training and testing datasets consisted of 10,000 and 90,000 synthetic instances, respectively, generated by a Monte Carlo generator. The NIDA network was trained on the Oak Ridge Leadership Computing Facility’s Titan using 10,000 computing nodes, and achieved a classification accuracy of 79.11% on the training set. Meanwhile, the mrDANNA was trained on a desktop and achieved a classification accuracy of 76.14% and 73.59% on the training and combined training and testing dataset, respectively. Both networks can attain an accuracy close to the state-of-art CNN accuracy of 80.42% while using far less neurons and synapses. In addition, the energy consumption was computed for the mrDANNA network and is estimated to be 1.66 _µ_ J per calculation. Although there is an accuracy drop using the smaller SNN networks, the energy consumption per calculation is very small, and thus can be deployed in edge devices. 

A recent work [163] implemented an SNN algorithm for filtering data from edge electronics in high energy collider experiments conducted at the High Luminosity Large Hadron Collider (HL-LHC), in order to reduce large data transfer rate or bandwidth (on the order of a few petabytes per second) to downstream electronics. In collider experiments, the collision events of charged particles with energy greater than 2 GeV is of significant interest. However, the high energy charged particles only comprise of approximately 10% of all recorded collision events. Therefore, filtering out low energy particle track clusters will greatly reduce data collection rate at edge devices. A synthetic dataset is used to train and test the SNN. The full synthetic dataset consists of 4 million charged particle interactions in a silicon pixel sensor. The training dataset is limited to the particle interactions in a 13 _×_ 21 pixel sub-region of the silicon sensor, with binary classification labels indicating high or low energy. The SNN is realized on Caspian [164], a neuromorphic development platform, and achieved a signal classification accuracy of 91.89%, very close to a prototyped full-precision DNN accuracy of 94.8%. In addition to accuracy, the SNN achieves good performance using nearly half of the number of DNN parameters. The reduced size and improved power efficiency of the SNN model makes it a good candidate for deployment on edge devices which have limited memory and power constraints. 

## **5 Summary** 

Experimental data generation at photon sources are rapidly increasing due to the advancements in light sources, detectors, and more efficient methods or modalities to collect data. As tabulated in Table 1, detectors can achieve frame-rates on the order of millions of frames per second with at least 10-bit data resolution, and 

25 



thus can achieve data rates over 1 Gb/s in continuous mode, and orders of magnitude higher data rate in burst mode. The high data rate is very costly in terms of data storage and transmission over long distances. These issues motivates the use of edge computing on detectors for real-time data processing and for reducing data transmission latency and storage volumes. 

Deep learning approaches have achieved significant progress in image processing tasks including but not limited to restoration, segmentation, compression, and 3D reconstruction. Their superior nonlinear approximation capabilities allow them to learn complex underlying structures and patterns in high dimensional data. The state-of-the-art methods for each image processing task achieve superior performance compared to conventional methods, while also overcoming the issues of conventional methods such as computational burdens associated with explicit programming for each data processing steps. Furthermore, once trained, deep learning methods can achieve very fast inference speeds for real-time computation. 

While deep learning approaches are widely used for many applications, they require deep networks to achieve good performance, and thus require heavy computational power and high energy consumption. This is critical hurdle for edge computing devices which have design constraints such as latency and energy. To address this issue, hardware accelerators now exist that leverage the model and data parallelism characteristics of neural network algorithms to implement parallel computing paradigms. Electronic-based hardware accelerators such as CPUs, GPUs, FPGAs, and ASICs are popularly used platforms for deep learning. However, the electronic-based solutions are constantly reaching performance limitations in clock speed, energy consumption, and other physical constraints. This gives rise to research in analog neuromorphic computing paradigms such as ONNs and SNNs to achieve high-speed, energy-efficient, and high-parallel computing, with significant potential for radiation detection and applications in photon science. 

## **Conflict of Interest Statement** 

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest. 

## **Author Contributions** 

SL: Supervision, Writing - original draft, Writing - review & editing; SN: Writing - original draft, Writing - review & editing; HZ: Writing - original draft, Writing - review & editing; TZ: Writing - original draft, Writing - review & editing; CLM: Writing - review & editing; SC: Writing - review & editing; MC: Writing - original draft, Writing - review & editing; RTC: Conceptualization, Writing - review & editing; ZW: Conceptualization, Writing - original draft, Writing - review & editing 

## **Funding** 

LANL work was performed under the auspices of the U.S. Department of Energy (DOE) by Triad National Security, LLC, operator of the Los Alamos National Laboratory under Contract No. 89233218CNA000001, including LANL Laboratory Directed Research and Development (LDRD) Program. This work is also supported in part by AFOSR MURI research center on Energy-efficient Optical Interconnects and Computing (Cont. No. FA9550-17-1-0071 managed by Dr. Gernot Pomrenke). 

26 



## **Acknowledgments** 

SL and ZW wish to thank Dr. Alice Bean from University of Kansas for reviewing the hardware section. SL and ZW wish to thank Dr. Mathieu Benoit from Oak Ridge National Laboratory for the CARIBOu discussions. 

27 



## **References** 

- [1] J. Dooling, M. Borland, W. Berg, J. Calvey, G. Decker, L. Emery, K. Harkay, R. Lindberg, G. Navrotksi, V. Sajaev, S. Shoaf, Y. P. Sun, K. P. Wootton, A. Xiao, A. Grannan, and A. H. Lumpkin, “Collimator irradiation studies in the argonne advanced photon source at energy densities expected in next-generation storage ring light sources,” _Phys. Rev. Accel. Beams_ , vol. 25, p. 043001, 2022. 

- [2] C. G. Schroer, I. Agapov, W. Brefeld, R. Brinkmann, Y.-C. Chae, H.-C. Chao, M. Eriksson, J. Keil, X. N. Gavald`a, R. R¨ohlsberger, O. H. Seeck, M. Sprung, M. Tischer, R. Wanzenberg, and E. Weckert, 

- “PETRA IV: the ultralow-emittance source project at DESY,” _Journal of Synchrotron Radiation_ , vol. 25, pp. 1277–1290, 2018. 

- [3] N. Huang, H. Deng, B. Liu, D. Wang, and Z. Zhao, “Features and futures of x-ray free-electron lasers,” _The Innovation_ , vol. 2(2), p. 100097, 2021. 

- [4] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” _nature_ , vol. 521, no. 7553, pp. 436–444, 2015. 

- [5] I. Goodfellow, Y. Bengio, and A. Courville, _Deep Learning_ . MIT Press, 2016. `http://www. deeplearningbook.org` . 

- [6] E. Masanet, A. Shehabi, N. Lei, S. Smith, and J. Koomey, “Recalibrating global data center energy-use estimates,” _Science_ , vol. 367, no. 6481, pp. 984 – 986, 2020. 

- [7] IEA, “Data centres and data transmission networks,” 2023. `https://www.iea.org/energy-system/ buildings/data-centres-and-data-transmission-networks` . 

- [8] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick, “Segment anything,” _arXiv_ , vol. 2304, no. 02643, 2023. `https://segment-anything.com/` . 

- [9] C. Adam-Bourdarios, G. Cowan, C. Germain, I. Guyon, B. K´egl, and D. Rousseau, “The higgs boson machine learning challenge,” in _NIPS 2014 workshop on high-energy physics and machine learning_ , pp. 19–55, PMLR, 2015. 

- [10] M. Azhari, A. Abarda, B. Ettaki, J. Zerouaoui, and M. Dakkon, “Higgs boson discovery using machine learning methods with pyspark,” _Procedia Computer Science_ , vol. 170, pp. 1141–1146, 2020. 

- [11] S. V. Kalinin, M. Ziatdinov, J. Hinkle, S. Jesse, A. Ghosh, K. P. Kelley, A. R. Lupini, B. G. Sumpter, and R. K. Vasudevan, “Automated and autonomous experiments in electron and scanning probe microscopy,” _ACS nano_ , vol. 15, no. 8, pp. 12604–12627, 2021. 

- [12] K. Marko, “Ai-enhanced instrumentation - the fusion of deep learning and medical sensors creates dramatic improvements,” 2018. `https://diginomica. com/ai-enhanced-instrumentation-fusion-deep-learning-medical-sensors-\ creates-dramatic-improvements` . 

- [13] ICinsights, “Cmos image sensor market sees perfect storm,” 2022. 

- [14] B. Niemann, D. Rudolph, and G. Schmahl, “X-ray microscopy with synchrotron radiation,” _Applied Optics_ , vol. 15, no. 8, pp. 1883–1884, 1976. 

28 



- [15] Z. Wang, A. F. Leong, A. Dragone, A. E. Gleason, R. Ballabriga, C. Campbell, M. Campbell, S. J. Clark, C. Da Vi`a, D. M. Dattelbaum, _et al._ , “Ultrafast radiographic imaging and tracking: An overview of instruments, methods, data, and applications,” _Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment_ , p. 168690, 2023. 

- [16] P. Spanne, C. Raven, I. Snigireva, and A. Snigirev, “In-line holography and phase-contrast microtomography with high energy x-rays,” _Physics in Medicine & Biology_ , vol. 44, no. 3, p. 741, 1999. 

- [17] J. Miao, T. Ishikawa, I. K. Robinson, and M. M. Murnane, “Beyond crystallography: Diffractive imaging using coherent x-ray light sources,” _Science_ , vol. 348, no. 6234, pp. 530–535, 2015. 

- [18] F. Pfeiffer, “X-ray ptychography,” _Nature Photonics_ , vol. 12, no. 1, pp. 9–17, 2018. 

- [19] P. Villanueva-Perez, S. Bajt, and H. Chapman, “Dose efficient compton x-ray microscopy,” _Optica_ , vol. 5, no. 4, pp. 450–457, 2018. 

- [20] A. Allahgholi, J. Becker, A. Delfs, R. Dinapoli, P. Goettlicher, D. Greiffenberg, B. Henrich, H. Hirsemann, M. Kuhn, R. Klanner, _et al._ , “The adaptive gain integrating pixel detector at the european xfel,” _Journal of synchrotron radiation_ , vol. 26, no. 1, pp. 74–82, 2019. 

- [21] S. Lin, J. Baldwin, M. Blatnik, S. Clayton, C. Cude-Woods, S. Currie, B. Filippone, E. Fries, P. Geltenbort, A. Holley, W. Li, C.-Y. Liu, M. Makela, C. Morris, R. Musedinovic, C. O’Shaughnessy, R. Pattie, D. Salvat, A. Saunders, E. Sharapov, M. Singh, X. Sun, Z. Tang, W. Uhrich, W. Wei, B. Wolfe, A. Young, and Z. Wang, “Demonstration of sub-micron ucn position resolution using room-temperature cmos sensor,” _Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment_ , vol. 1057, p. 168769, 2023. 

- [22] P. Orponen, “Computational complexity of neural networks: A survey,” _Nordic J. Comp._ , vol. 1994, no. 1, pp. 94–110, 1994. 

- [23] P. Maji and R. Mullins, “On the reduction of computational complexity of deep convolution neural networks,” _Entropy_ , vol. 20, no. 305, 2018. 

- [24] P. J. Freire, S. Srivallapanondh, A. Napoli, J. E. Prilepsky, and S. K. Turitsyn, “Computational complexity evaluation of neural network applications in signal processing,” 2022. arXiv:2206.12191. 

- [25] G. H. Golub and C. F. Van Loan, _Matrix Computations_ . The Johns Hopkins University Press, 4 ed., 2013. 

- [26] R. Turchetta, “Towards gfps cmos image sensors,” in _In Workshop on Computational Image Sensors and Smart Cameras, WASC_ , vol. 2017, 2017. 

- [27] A. Allahgholi, J. Becker, A. Delfs, R. Dinapoli, P. Goettlicher, H. Graafsma, D. Greiffenberg, H. Hirsemann, S. Jack, A. Klyuev, _et al._ , “Megapixels@ megahertz–the agipd high-speed cameras for the european xfel,” _Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment_ , vol. 942, p. 162324, 2019. 

- [28] H. T. Philipp, M. Hromalik, M. Tate, L. Koerner, and S. M. Gruner, “Pixel array detector for x-ray free electron laser experiments,” _Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment_ , vol. 649, no. 1, pp. 67–69, 2011. 

29 



- [29] G. Carini, R. Alonso-Mori, G. Blaj, P. Caragiulo, M. Chollet, D. Damiani, A. Dragone, Y. Feng, G. Haller, P. Hart, _et al._ , “epix100 camera: Use and applications at lcls,” in _AIP Conference Proceedings_ , vol. 1741, AIP Publishing, 2016. 

- [30] M. Veale, P. Seller, M. Wilson, and E. Liotti, “Hexitec: A high-energy x-ray spectroscopic imaging detector for synchrotron applications,” _Synchrotron Radiation News_ , vol. 31, no. 6, pp. 28–32, 2018. 

- [31] L. Claus, T. England, L. Fang, G. Robertson, M. Sanchez, D. Trotter, A. Carpenter, M. Dayton, P. Patel, and J. Porter, “Design and characterization of an improved, 2 ns, multi-frame imager for the ultra-fast x-ray imager (uxi) program at sandia national laboratories,” in _Target Diagnostics Physics and Engineering for Inertial Confinement Fusion VI_ , vol. 10390, pp. 16–26, SPIE, 2017. 

- [32] F. Leonarski, A. Mozzanica, M. Br¨uckner, C. Lopez-Cuenca, S. Redford, L. Sala, A. Babic, H. Billich, O. Bunk, B. Schmitt, _et al._ , “Jungfrau detector for brighter x-ray sources: Solutions for it and data science challenges in macromolecular crystallography,” _Structural Dynamics_ , vol. 7, no. 1, 2020. 

- [33] D. Gadkari, K. Shanks, H. Hu, H. Philipp, M. Tate, J. Thom-Levy, and S. Gruner, “Characterization of 128 _×_ 128 mm-pad-2.1 asic: a fast framing hard x-ray detector with high dynamic range,” _Journal of Instrumentation_ , vol. 17, no. 03, p. P03003, 2022. 

- [34] Y. Tochigi, K. Hanzawa, Y. Kato, R. Kuroda, H. Mutoh, R. Hirose, H. Tominaga, K. Takubo, Y. Kondo, and S. Sugawa, “A global-shutter cmos image sensor with readout speed of 1-tpixel/s burst and 780-mpixel/s continuous,” _IEEE Journal of Solid-State Circuits_ , vol. 48, no. 1, pp. 329–338, 2012. 

- [35] A. Lewis, S. Baker, A. Corredor, L. Fegenbush, Z. Fitzpatrick, M. Jones, K. O’Flarity, K. Walters, L. Claus, and M. Sanchez, “New design yields robust large-area framing camera,” _Review of Scientific Instruments_ , vol. 92, no. 8, 2021. 

- [36] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,” _ACM computing surveys (CSUR)_ , vol. 54, no. 10s, pp. 1–41, 2022. 

- [37] N. Zhang, S. Ding, J. Zhang, and Y. Xue, “An overview on restricted boltzmann machines,” _Neurocomputing_ , vol. 275, pp. 1186–1199, 2018. 

- [38] J. Cao, Z. Lin, _et al._ , “Extreme learning machines on high dimensional and large data applications: a survey,” _Mathematical Problems in Engineering_ , vol. 2015, 2015. 

- [39] M. D. Zeiler, _Hierarchical convolutional deep learning in computer vision_ . PhD thesis, New York University, 2013. 

- [40] Y.-L. Boureau, J. Ponce, and Y. LeCun, “A theoretical analysis of feature pooling in visual recognition,” in _Proceedings of the 27th international conference on machine learning (ICML-10)_ , pp. 111–118, 2010. 

- [41] D. Scherer, A. M¨uller, and S. Behnke, “Evaluation of pooling operations in convolutional architectures for object recognition,” in _International conference on artificial neural networks_ , pp. 92–101, Springer, 2010. 

- [42] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” _Advances in neural information processing systems_ , vol. 25, 2012. 

30 



- [43] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” _arXiv preprint arXiv:1409.1556_ , 2014. 

- [44] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1–9, 2015. 

- [45] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- [46] Y. Yu, X. Si, C. Hu, and J. Zhang, “A review of recurrent neural networks: Lstm cells and network architectures,” _Neural computation_ , vol. 31, no. 7, pp. 1235–1270, 2019. 

- [47] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” _arXiv preprint arXiv:1409.0473_ , 2014. 

- [48] S. Herdade, A. Kappeler, K. Boakye, and J. Soares, “Image captioning: Transforming objects into words,” _Advances in neural information processing systems_ , vol. 32, 2019. 

- [49] C.-C. Chiu, T. N. Sainath, Y. Wu, R. Prabhavalkar, P. Nguyen, Z. Chen, A. Kannan, R. J. Weiss, K. Rao, E. Gonina, _et al._ , “State-of-the-art speech recognition with sequence-to-sequence models,” in _2018 IEEE international conference on acoustics, speech and signal processing (ICASSP)_ , pp. 4774–4778, IEEE, 2018. 

- [50] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” _Advances in neural information processing systems_ , vol. 27, 2014. 

- [51] A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta, and A. A. Bharath, “Generative adversarial networks: An overview,” _IEEE signal processing magazine_ , vol. 35, no. 1, pp. 53–65, 2018. 

- [52] C. Tian, L. Fei, W. Zheng, Y. Xu, W. Zuo, and C.-W. Lin, “Deep learning on image denoising: An overview,” _Neural Networks_ , vol. 131, pp. 251–275, 2020. 

- [53] A. Lucas, M. Iliadis, R. Molina, and A. K. Katsaggelos, “Using deep neural networks for inverse problems in imaging: beyond analytical methods,” _IEEE Signal Processing Magazine_ , vol. 35, no. 1, pp. 20–36, 2018. 

- [54] Z. Liu, T. Bicer, R. Kettimuthu, D. Gursoy, F. De Carlo, and I. Foster, “Tomogan: low-dose synchrotron x-ray tomography with generative adversarial networks: discussion,” _JOSA A_ , vol. 37, no. 3, pp. 422–434, 2020. 

- [55] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in _Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18_ , pp. 234–241, Springer, 2015. 

- [56] X. Duan, X. F. Ding, N. Li, F.-X. Wu, X. Chen, and N. Zhu, “Sparse2noise: Low-dose synchrotron x-ray tomography without high-quality reference data,” _Computers in Biology and Medicine_ , vol. 165, p. 107473, 2023. 

31 



- [57] A. A. Hendriksen, M. B¨uhrer, L. Leone, M. Merlini, N. Vigano, D. M. Pelt, F. Marone, M. Di Michiel, and K. J. Batenburg, “Deep denoising for multi-dimensional synchrotron x-ray tomography without high-quality reference data,” _Scientific reports_ , vol. 11, no. 1, p. 11895, 2021. 

- [58] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2noise: Learning image restoration without clean data,” _arXiv preprint arXiv:1803.04189_ , 2018. 

- [59] C. L. Morris, N. King, K. Kwiatkowski, F. Mariam, F. Merrill, and A. Saunders, “Charged particle radiography,” _Reports on progress in physics_ , vol. 76, no. 4, p. 046301, 2013. 

- [60] K. Zhang, W. Ren, W. Luo, W.-S. Lai, B. Stenger, M.-H. Yang, and H. Li, “Deep image deblurring: A survey,” _International Journal of Computer Vision_ , vol. 130, no. 9, pp. 2103–2130, 2022. 

- [61] S. A. Biyouki and H. Hwangbo, “A comprehensive survey on deep neural image deblurring,” _arXiv preprint arXiv:2310.04719_ , 2023. 

- [62] J. Yang, C. Zhao, S. Qiao, T. Zhang, and X. Yao, “Deep learning methods for neutron image restoration,” _Annals of Nuclear Energy_ , vol. 188, p. 109820, 2023. 

- [63] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,” _IEEE transactions on image processing_ , vol. 26, no. 7, pp. 3142–3155, 2017. 

- [64] J. Gurrola-Ramos, O. Dalmau, and T. E. Alarc´on, “A residual dense u-net neural network for image denoising,” _IEEE Access_ , vol. 9, pp. 31742–31754, 2021. 

- [65] C. Miao, L. Xie, F. Wan, C. Su, H. Liu, J. Jiao, and Q. Ye, “Sixray: A large-scale security inspection x-ray benchmark for prohibited item discovery in overlapping images,” in _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 2119–2128, 2019. 

- [66] Y. Li, B. Sixou, and F. Peyrin, “A review of the deep learning methods for medical images super resolution problems,” _Irbm_ , vol. 42, no. 2, pp. 120–133, 2021. 

- [67] J. Jiang, C. Wang, X. Liu, and J. Ma, “Deep learning-based face super-resolution: A survey,” _ACM Computing Surveys (CSUR)_ , vol. 55, no. 1, pp. 1–36, 2021. 

- [68] C. Dong, C. C. Loy, K. He, and X. Tang, “Image super-resolution using deep convolutional networks,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 38, no. 2, pp. 295–307, 2015. 

- [69] C. Ledig, L. Theis, F. Husz´ar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, _et al._ , “Photo-realistic single image super-resolution using a generative adversarial network,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4681–4690, 2017. 

- [70] Z. Wang, J. Chen, and S. C. Hoi, “Deep learning for image super-resolution: A survey,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 43, no. 10, pp. 3365–3387, 2020. 

- [71] J. Park, D. Hwang, K. Y. Kim, S. K. Kang, Y. K. Kim, and J. S. Lee, “Computed tomography super-resolution using deep convolutional neural network,” _Physics in Medicine & Biology_ , vol. 63, no. 14, p. 145011, 2018. 

- [72] R. Szeliski, _Computer vision: algorithms and applications_ . Springer Nature, 2022. 

32 



- [73] R. Wang, T. Lei, R. Cui, B. Zhang, H. Meng, and A. K. Nandi, “Medical image segmentation using deep learning: A survey,” _IET Image Processing_ , vol. 16, no. 5, pp. 1243–1267, 2022. 

- [74] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 39, no. 12, pp. 2481–2495, 2017. 

- [75] S. Minaee, Y. Boykov, F. Porikli, A. Plaza, N. Kehtarnavaz, and D. Terzopoulos, “Image segmentation using deep learning: A survey,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 44, no. 7, pp. 3523–3542, 2021. 

- [76] M. Falato, B. Wo lfe, N. Nguyen, X. Zhang, and Z. Wang, “Contour extraction of inertial confinement fusion images by data augmentation,” _arXiv preprint arXiv:2211.04597_ , 2022. 

- [77] W. E. Lewis, P. F. Knapp, E. C. Harding, and K. Beckwith, “Statistical characterization of experimental magnetized liner inertial fusion stagnation images using deep-learning-based fuel–background segmentation,” _Journal of Plasma Physics_ , vol. 88, no. 5, p. 895880501, 2022. 

- [78] D. Mishra, S. K. Singh, and R. K. Singh, “Deep architectures for image compression: a critical review,” _Signal Processing_ , vol. 191, p. 108346, 2022. 

- [79] G. K. Wallace, “The jpeg still picture compression standard,” _IEEE transactions on consumer electronics_ , vol. 38, no. 1, pp. xviii–xxxiv, 1992. 

- [80] D. L. Donoho, “Compressed sensing,” _IEEE Transactions on information theory_ , vol. 52, no. 4, pp. 1289– 1306, 2006. 

- [81] W. Shi, F. Jiang, S. Liu, and D. Zhao, “Image compressed sensing using convolutional neural network,” _IEEE Transactions on Image Processing_ , vol. 29, pp. 375–388, 2019. 

- [82] A. L. Machidon and V. Pejovi´c, “Deep learning for compressive sensing: a ubiquitous systems perspective,” _Artificial Intelligence Review_ , vol. 56, no. 4, pp. 3619–3658, 2023. 

- [83] M. J. Cherukara, T. Zhou, Y. Nashed, P. Enfedaque, A. Hexemer, R. J. Harder, and M. V. Holt, “Ai-enabled high-resolution scanning coherent diffraction imaging,” _Applied Physics Letters_ , vol. 117, no. 4, 2020. 

- [84] A. V. Babu, T. Zhou, S. Kandel, T. Bicer, Z. Liu, W. Judge, D. J. Ching, Y. Jiang, S. Veseli, S. Henke, _et al._ , “Deep learning at the edge enables real-time streaming ptychographic imaging,” _arXiv preprint arXiv:2209.09408_ , 2022. 

- [85] S. Kandel, T. Zhou, A. V. Babu, Z. Di, X. Li, X. Ma, M. Holt, A. Miceli, C. Phatak, and M. J. Cherukara, “Demonstration of an ai-driven workflow for autonomous high-resolution scanning microscopy,” _Nature Communications_ , vol. 14, no. 1, p. 5501, 2023. 

- [86] R. K. Vasudevan, K. P. Kelley, J. Hinkle, H. Funakubo, S. Jesse, S. V. Kalinin, and M. Ziatdinov, “Autonomous experiments in scanning probe microscopy and spectroscopy: choosing where to explore polarization dynamics in ferroelectrics,” _ACS nano_ , vol. 15, no. 7, pp. 11253–11262, 2021. 

33 



- [87] M. M. Noack, K. G. Yager, M. Fukuto, G. S. Doerk, R. Li, and J. A. Sethian, “A kriging-based approach to autonomous experimentation with applications to x-ray scattering,” _Scientific reports_ , vol. 9, no. 1, p. 11809, 2019. 

- [88] S. Venkatakrishnan, C. M. Fancher, M. Ziatdinov, R. Vasudevan, K. Saleeby, J. Haley, D. Yu, K. An, and A. Plotkowski, “Adaptive sampling for accelerating neutron diffraction-based strain mapping,” _Machine Learning: Science and Technology_ , vol. 4, no. 2, p. 025001, 2023. 

- [89] H. Liu, Y.-S. Ong, X. Shen, and J. Cai, “When gaussian process meets big data: A review of scalable gps,” _IEEE transactions on neural networks and learning systems_ , vol. 31, no. 11, pp. 4405–4423, 2020. 

- [90] M. Schloz, J. M¨uller, T. C. Pekin, W. Van den Broek, J. Madsen, T. Susi, and C. T. Koch, “Deep reinforcement learning for data-driven adaptive scanning in ptychography,” _Scientific Reports_ , vol. 13, no. 1, p. 8732, 2023. 

- [91] X.-F. Han, H. Laga, and M. Bennamoun, “Image-based 3d object reconstruction: State-of-the-art and trends in the deep learning era,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 43, no. 5, pp. 1578–1604, 2019. 

- [92] K. Fu, J. Peng, Q. He, and H. Zhang, “Single image 3d object reconstruction based on deep learning: A review,” _Multimedia Tools and Applications_ , vol. 80, pp. 463–498, 2021. 

- [93] Z. Guan, E. H. Tsai, X. Huang, K. G. Yager, and H. Qin, “Ptychonet: Fast and high quality phase retrieval for ptychography,” tech. rep., Brookhaven National Lab.(BNL), Upton, NY (United States), 2019. 

- [94] M. J. Cherukara, Y. S. Nashed, and R. J. Harder, “Real-time coherent diffraction inversion using deep generative networks,” _Scientific reports_ , vol. 8, no. 1, p. 16520, 2018. 

- [95] Y. Yao, H. Chan, S. Sankaranarayanan, P. Balaprakash, R. J. Harder, and M. J. Cherukara, “Autophasenn: unsupervised physics-aware deep learning of 3d nanoscale bragg coherent diffraction imaging,” _npj Computational Materials_ , vol. 8, no. 1, p. 124, 2022. 

- [96] A. Scheinker and R. Pokharel, “Adaptive 3d convolutional neural network-based reconstruction method for 3d coherent diffraction imaging,” _Journal of Applied Physics_ , vol. 128, no. 18, 2020. 

- [97] A. Gholami, A. Azad, P. Jin, K. Keutzer, and A. Buluc, “Integrated model, batch, and domain parallelism in training neural networks,” in _Proceedings of the 30th on Symposium on Parallelism in Algorithms and Architectures_ , pp. 77–86, 2018. 

- [98] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, “Efficient processing of deep neural networks: A tutorial and survey,” _Proceedings of the IEEE_ , vol. 105, no. 12, pp. 2295–2329, 2017. 

- [99] Intel, _Compare Benefits of CPUs, GPUs, and FPGAs for Different oneAPI Compute Workloads_ , 2022. `https://www.intel.com/content/www/us/en/developer/articles/technical/ comparing-cpus-gpus-and-fpgas-for-oneapi.html` , Last accessed on 2023-09-25. 

- [100] S. Mittal and J. S. Vetter, “A survey of cpu-gpu heterogeneous computing techniques,” _ACM Computing Surveys (CSUR)_ , vol. 47, no. 4, pp. 1–35, 2015. 

34 



- [101] S. Mittal, “A survey on optimized implementation of deep learning models on the nvidia jetson platform,” _Journal of Systems Architecture_ , vol. 97, pp. 428–442, 2019. 

- [102] V. Abeykoon, Z. Liu, R. Kettimuthu, G. Fox, and I. Foster, “Scientific image restoration anywhere,” in _2019 IEEE/ACM 1st Annual Workshop on Large-scale Experiment-in-the-Loop Computing (XLOOP)_ , pp. 8–13, IEEE, 2019. 

- [103] L. An, K. Peng, X. Yang, P. Huang, Y. Luo, P. Feng, and B. Wei, “E-tbnet: Light deep neural network for automatic detection of tuberculosis with x-ray dr imaging,” _Sensors_ , vol. 22, no. 3, p. 821, 2022. 

- [104] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4510–4520, 2018. 

- [105] X. Zhang, X. Zhou, M. Lin, and J. Sun, “Shufflenet: An extremely efficient convolutional neural network for mobile devices,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6848–6856, 2018. 

- [106] F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer, “Squeezenet: Alexnetlevel accuracy with 50x fewer parameters and¡ 0.5 mb model size,” _arXiv preprint arXiv:1602.07360_ , 2016. 

- [107] J. Li, G. Yan, W. Lu, S. Jiang, S. Gong, J. Wu, and X. Li, “Smartshuttle: Optimizing off-chip memory accesses for deep learning accelerators,” in _2018 Design, Automation & Test in Europe Conference & Exhibition (DATE)_ , pp. 343–348, IEEE, 2018. 

- [108] Y.-H. Chen, J. Emer, and V. Sze, “Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks,” _ACM SIGARCH computer architecture news_ , vol. 44, no. 3, pp. 367–379, 2016. 

- [109] M. Capra, B. Bussolino, A. Marchisio, G. Masera, M. Martina, and M. Shafique, “Hardware and software optimizations for accelerating deep neural networks: Survey of current trends, challenges, and the road ahead,” _IEEE Access_ , vol. 8, pp. 225134–225180, 2020. 

- [110] P. Dhilleswararao, S. Boppu, M. S. Manikandan, and L. R. Cenkeramaddi, “Efficient hardware architectures for accelerating deep neural networks: Survey,” _IEEE Access_ , 2022. 

- [111] M. Xia, Z. Huang, L. Tian, H. Wang, V. Chang, Y. Zhu, and S. Feng, “Sparknoc: An energy-efficiency fpga-based accelerator using optimized lightweight cnn for edge computing,” _Journal of Systems Architecture_ , vol. 115, p. 101991, 2021. 

- [112] X. Liu, J. Yang, C. Zou, Q. Chen, X. Yan, Y. Chen, and C. Cai, “Collaborative edge computing with fpga-based cnn accelerators for energy-efficient and time-aware face tracking system,” _IEEE Transactions on Computational Social Systems_ , vol. 9, no. 1, pp. 252–266, 2021. 

- [113] S. Cass, “Taking ai to the edge: Google’s tpu now comes in a maker-friendly package,” _IEEE Spectrum_ , vol. 56, no. 5, pp. 16–17, 2019. 

- [114] Y. Sun and A. M. Kist, “Deep learning on edge tpus,” _arXiv preprint arXiv:2108.13732_ , 2021. 

35 



- [115] H. Liu, M. Benoit, H. Chen, K. Chen, F. Di Bello, G. Iacobucci, F. Lanni, I. Peric, B. Ristic, M. V. B. Pinto, _et al._ , “Development of a modular test system for the silicon sensor r&d of the atlas upgrade,” _Journal of Instrumentation_ , vol. 12, no. 01, p. P01008, 2017. 

- [116] S. Strempfer, T. Zhou, K. Yoshii, M. Hammer, A. Babu, D. Bycul, J. Weizeorick, M. J. Cherukara, and A. Miceli, “A lightweight, user-configurable detector asic digital architecture with on-chip data compression for mhz x-ray coherent diffraction imaging,” _Journal of Instrumentation_ , vol. 17, no. 10, p. P10042, 2022. 

- [117] K. Rupp, “Microprocessor trend data.” `https://github.com/karlrupp/ microprocessor-trend-data/tree/master` , 2022. 

- [118] M. M. Waldrop, “More than moore,” _Nature_ , vol. 530, no. 7589, pp. 144–148, 2016. 

- [119] V. P. Heuring and M. J. Murdocca, “Principles of computer architecture,” 2021. 

- [120] A. Ganguly, R. Muralidhar, and V. Singh, “Towards energy efficient non-von neumann architectures for deep learning,” in _20th international symposium on quality electronic design (ISQED)_ , pp. 335–342, IEEE, 2019. 

- [121] X. Sui, Q. Wu, J. Liu, Q. Chen, and G. Gu, “A review of optical neural networks,” _IEEE Access_ , vol. 8, pp. 70773–70783, 2020. 

- [122] Y. Shen, N. C. Harris, S. Skirlo, M. Prabhu, T. Baehr-Jones, M. Hochberg, X. Sun, S. Zhao, H. Larochelle, D. Englund, _et al._ , “Deep learning with coherent nanophotonic circuits,” _Nature photonics_ , vol. 11, no. 7, pp. 441–446, 2017. 

- [123] B. J. Shastri, A. N. Tait, T. Ferreira de Lima, W. H. Pernice, H. Bhaskaran, C. D. Wright, and P. R. Prucnal, “Photonics for artificial intelligence and neuromorphic computing,” _Nature Photonics_ , vol. 15, no. 2, pp. 102–114, 2021. 

- [124] C. Feng, S. Ning, J. Gu, H. Zhu, D. Z. Pan, and R. T. Chen, “Integrated photonics for computing and artificial intelligence,” in _2023 IEEE Photonics Society Summer Topicals Meeting Series (SUM)_ , pp. 1–2, IEEE, 2023. 

- [125] J. Gu, C. Feng, H. Zhu, R. T. Chen, and D. Z. Pan, “Light in ai: Toward efficient neurocomputing with optical neural networks—a tutorial,” _IEEE Transactions on Circuits and Systems II: Express Briefs_ , vol. 69, no. 6, pp. 2581–2585, 2022. 

- [126] A. N. Tait, T. F. De Lima, E. Zhou, A. X. Wu, M. A. Nahmias, B. J. Shastri, and P. R. Prucnal, “Neuromorphic photonic networks using silicon photonic weight banks,” _Scientific reports_ , vol. 7, no. 1, p. 7430, 2017. 

- [127] J. Feldmann, N. Youngblood, M. Karpov, H. Gehring, X. Li, M. Stappers, M. Le Gallo, X. Fu, A. Lukashchuk, A. S. Raja, _et al._ , “Parallel convolutional processing using an integrated photonic tensor core,” _Nature_ , vol. 589, no. 7840, pp. 52–58, 2021. 

- [128] C. R´ıos, N. Youngblood, Z. Cheng, M. Le Gallo, W. H. Pernice, C. D. Wright, A. Sebastian, and H. Bhaskaran, “In-memory computing on a photonic platform,” _Science advances_ , vol. 5, no. 2, p. eaau5759, 2019. 

36 



- [129] H. Zhu, J. Gu, C. Feng, M. Liu, Z. Jiang, R. T. Chen, and D. Z. Pan, “Elight: Toward efficient and aging-resilient photonic in-memory neurocomputing,” _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , vol. 42, no. 3, pp. 820–833, 2022. 

- [130] X. Lin, Y. Rivenson, N. T. Yardimci, M. Veli, Y. Luo, M. Jarrahi, and A. Ozcan, “All-optical machine learning using diffractive deep neural networks,” _Science_ , vol. 361, no. 6406, pp. 1004–1008, 2018. 

- [131] T. Yan, J. Wu, T. Zhou, H. Xie, F. Xu, J. Fan, L. Fang, X. Lin, and Q. Dai, “Fourier-space diffractive deep neural network,” _Physical review letters_ , vol. 123, no. 2, p. 023901, 2019. 

- [132] F. Ashtiani, A. J. Geers, and F. Aflatouni, “An on-chip photonic deep neural network for image classification,” _Nature_ , vol. 606, no. 7914, pp. 501–506, 2022. 

- [133] C. Feng, J. Gu, H. Zhu, Z. Ying, Z. Zhao, D. Z. Pan, and R. T. Chen, “A compact butterfly-style silicon photonic–electronic neural chip for hardware-efficient deep learning,” _Acs Photonics_ , vol. 9, no. 12, pp. 3906–3916, 2022. 

- [134] C. Feng, R. Tang, J. Gu, H. Zhu, D. Z. Pan, and R. T. Chen, “Optically-interconnected, hardwareefficient, electronic-photonic neural network using compact multi-operand photonic devices,” in _Optical Interconnects XXIII_ , p. PC1242702, SPIE, 2023. 

- [135] J. Gu, Z. Zhao, C. Feng, H. Zhu, R. T. Chen, and D. Z. Pan, “Roq: A noise-aware quantization scheme towards robust optical neural networks with low-bit controls,” in _2020 Design, Automation & Test in Europe Conference & Exhibition (DATE)_ , pp. 1586–1589, IEEE, 2020. 

- [136] H. Zhu, J. Gu, H. Wang, Z. Jiang, Z. Zhang, R. Tang, C. Feng, S. Han, R. T. Chen, and D. Z. Pan, “Dota: A dynamically-operated photonic tensor core for energy-efficient transformer accelerator,” _arXiv preprint arXiv:2305.19533_ , 2023. 

- [137] T. Wang, M. M. Sohoni, L. G. Wright, M. M. Stein, S.-Y. Ma, T. Onodera, M. G. Anderson, and P. L. McMahon, “Image sensing with multilayer nonlinear optical neural networks,” _Nature Photonics_ , vol. 17, no. 5, pp. 408–415, 2023. 

- [138] T. Zhou, W. Wu, J. Zhang, S. Yu, and L. Fang, “Ultrafast dynamic machine vision with spatiotemporal photonic computing,” _Science Advances_ , vol. 9, no. 23, p. eadg4391, 2023. 

- [139] T. Yamaguchi, K. Arai, T. Niiyama, A. Uchida, and S. Sunada, “Time-domain photonic image processor based on speckle projection and reservoir computing,” _Communications Physics_ , vol. 6, no. 1, p. 250, 2023. 

- [140] L. Huang, S. Mukherjee, Q. Tanguy, J. Fr¨och, and A. Majumdar, “Photonic advantage of optical encoders,” in _2023 Conference on Lasers and Electro-Optics (CLEO)_ , pp. 1–2, IEEE, 2023. 

- [141] H. Zhu, K. Zhu, J. Gu, H. Jin, R. T. Chen, J. A. Incorvia, and D. Z. Pan, “Fuse and mix: Macamenabled analog activation for energy-efficient neural acceleration,” in _Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design_ , pp. 1–9, 2022. 

- [142] J. Gu, C. Feng, H. Zhu, Z. Zhao, Z. Ying, M. Liu, R. T. Chen, and D. Z. Pan, “Squeezelight: A multi-operand ring-based optical neural network with cross-layer scalability,” _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , vol. 42, no. 3, pp. 807–819, 2022. 

37 



- [143] C. Feng, J. Gu, H. Zhu, R. Tang, S. Ning, M. Hlaing, J. Midkiff, S. Jain, D. Z. Pan, and R. T. Chen, “Integrated multi-operand optical neurons for scalable and hardware-efficient deep learning,” _arXiv preprint arXiv:2305.19592_ , 2023. 

- [144] H. Zhu, J. Zou, H. Zhang, Y. Shi, S. Luo, N. Wang, H. Cai, L. Wan, B. Wang, X. Jiang, _et al._ , “Spaceefficient optical computing with an integrated chip diffractive neural network,” _Nature communications_ , vol. 13, no. 1, p. 1044, 2022. 

- [145] J. D. Nunes, M. Carvalho, D. Carneiro, and J. S. Cardoso, “Spiking neural networks: A survey,” _IEEE Access_ , vol. 10, pp. 60738–60764, 2022. 

- [146] A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, and A. Maida, “Deep learning in spiking neural networks,” _Neural networks_ , vol. 111, pp. 47–63, 2019. 

- [147] C. D. Schuman, S. R. Kulkarni, M. Parsa, J. P. Mitchell, P. Date, and B. Kay, “Opportunities for neuromorphic computing algorithms and applications,” _Nature Computational Science_ , vol. 2, no. 1, pp. 10–19, 2022. 

- [148] M. Pfeiffer and T. Pfeil, “Deep learning with spiking neurons: Opportunities and challenges,” _Frontiers in neuroscience_ , vol. 12, p. 774, 2018. 

- [149] M. Davies, N. Srinivasa, T.-H. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain, _et al._ , “Loihi: A neuromorphic manycore processor with on-chip learning,” _Ieee Micro_ , vol. 38, no. 1, pp. 82–99, 2018. 

- [150] M. Davies, A. Wild, G. Orchard, Y. Sandamirskaya, G. A. F. Guerra, P. Joshi, P. Plank, and S. R. Risbud, “Advancing neuromorphic computing with loihi: A survey of results and outlook,” _Proceedings of the IEEE_ , vol. 109, no. 5, pp. 911–934, 2021. 

- [151] N. Getty, T. Brettin, D. Jin, R. Stevens, and F. Xia, “Deep medical image analysis with representation learning and neuromorphic computing,” _Interface Focus_ , vol. 11, no. 1, p. 20190122, 2021. 

- [152] P. A. Merolla, J. V. Arthur, R. Alvarez-Icaza, A. S. Cassidy, J. Sawada, F. Akopyan, B. L. Jackson, N. Imam, C. Guo, Y. Nakamura, _et al._ , “A million spiking-neuron integrated circuit with a scalable communication network and interface,” _Science_ , vol. 345, no. 6197, pp. 668–673, 2014. 

- [153] R. Shukla, M. Lipasti, B. Van Essen, A. Moody, and N. Maruyama, “Remodel: Rethinking deep cnn models to detect and count on a neurosynaptic system,” _Frontiers in neuroscience_ , vol. 13, p. 4, 2019. 

- [154] B. V. Benjamin, P. Gao, E. McQuinn, S. Choudhary, A. R. Chandrasekaran, J.-M. Bussat, R. AlvarezIcaza, J. V. Arthur, P. A. Merolla, and K. Boahen, “Neurogrid: A mixed-analog-digital multichip system for large-scale neural simulations,” _Proceedings of the IEEE_ , vol. 102, no. 5, pp. 699–716, 2014. 

- [155] J. Schemmel, D. Br¨uderle, A. Gr¨ubl, M. Hock, K. Meier, and S. Millner, “A wafer-scale neuromorphic hardware system for large-scale neural modeling,” in _2010 IEEE International Symposium on Circuits and Systems (ISCAS)_ , pp. 1947–1950, IEEE, 2010. 

- [156] S. B. Furber, F. Galluppi, S. Temple, and L. A. Plana, “The spinnaker project,” _Proceedings of the IEEE_ , vol. 102, no. 5, pp. 652–665, 2014. 

38 



- [157] M. Bouvier, A. Valentian, T. Mesquida, F. Rummens, M. Reyboz, E. Vianello, and E. Beigne, “Spiking neural networks hardware implementations and challenges: A survey,” _ACM Journal on Emerging Technologies in Computing Systems (JETC)_ , vol. 15, no. 2, pp. 1–35, 2019. 

- [158] A. Basu, L. Deng, C. Frenkel, and X. Zhang, “Spiking neural network integrated circuits: A review of trends and future directions,” in _2022 IEEE Custom Integrated Circuits Conference (CICC)_ , pp. 1–8, IEEE, 2022. 

- [159] C. D. Schuman, T. E. Potok, S. Young, R. Patton, G. Perdue, G. Chakma, A. Wyer, and G. S. Rose, “Neuromorphic computing for temporal scientific data classification,” in _Proceedings of the Neuromorphic Computing Symposium_ , pp. 1–6, 2017. 

- [160] L. Aliaga, L. Bagby, B. Baldin, A. Baumbaugh, A. Bodek, R. Bradford, W. Brooks, D. Boehnlein, S. Boyd, H. Budd, _et al._ , “Design, calibration, and performance of the minerva detector,” _Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment_ , vol. 743, pp. 130–159, 2014. 

- [161] C. D. Schuman, J. D. Birdwell, and M. E. Dean, “Spatiotemporal classification using neuroscienceinspired dynamic architectures,” _Procedia Computer Science_ , vol. 41, pp. 89–97, 2014. 

- [162] N. C. Cady, S. P. C. of Nanoscale Science, and Engineering, “Development of a memristive dynamic adaptive neural network array (mrdanna),” 2019. 

- [163] S. R. Kulkarni, A. Young, P. Date, N. Rao Miniskar, J. Vetter, F. Fahim, B. Parpillon, J. Dickinson, N. Tran, J. Yoo, _et al._ , “On-sensor data filtering using neuromorphic computing for high energy physics experiments,” in _Proceedings of the 2023 International Conference on Neuromorphic Systems_ , pp. 1–8, 2023. 

- [164] J. P. Mitchell, C. D. Schuman, R. M. Patton, and T. E. Potok, “Caspian: A neuromorphic development platform,” in _Proceedings of the 2020 Annual Neuro-Inspired Computational Elements Workshop_ , pp. 1–6, 2020. 

39 

