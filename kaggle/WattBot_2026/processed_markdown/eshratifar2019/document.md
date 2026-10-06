# **BottleNet: A Deep Learning Architecture for Intelligent Mobile Cloud Computing Services** 

Amir Erfan Eshratifar Amirhossein Esmaili Department of Electrical Engineering Department of Electrical Engineering University of Southern California University of Southern California Los Angeles, California Los Angeles, California eshratif@usc.edu esmailid@usc.edu 

Massoud Pedram Department of Electrical Engineering University of Southern California Los Angeles, California pedram@usc.edu 

## **ABSTRACT** 

![Figure](assets/figure_0001_page_0001.svg)Recent studies have shown the latency and energy consumption of deep neural networks can be significantly improved by splitting the network between the mobile device and cloud. This paper introduces a new deep learning architecture, called BottleNet, for reducing the feature size needed to be sent to the cloud. Furthermore, we propose a training method for compensating for the potential accuracy loss due to the lossy compression of features before transmitting them to the cloud. BottleNet achieves on average 30× improvement in end-to-end latency and 40× improvement in mobile energy consumption compared to the cloud-only approach with negligible accuracy loss. 

**Figure 1: Overview of the proposed method.** 

## **KEYWORDS** 

deep learning, collaborative intelligence, mobile computing, cloud computing, feature compression 

## **1 INTRODUCTION** 

Mobile and Internet of Things (IoT) devices are increasingly relying on deep neural networks (DNNs) to provide state-of-the-art performance in various intelligent applications [1–5]. Due to limited computational and storage resources of mobile devices, which prohibits full deployment of advanced deep models on these devices (the _mobile-only_ method), the most common deployment approach of most of the DNN-based applications on mobile devices relies on using the cloud. In this approach, which is referred to as the _cloud-only_ approach, the deep network is fully placed on the cloud, and thus the input is sent from the mobile to cloud for performing the computations associated with the inference network, and the output is sent back to the mobile device. 

The cloud-only approach requires mobile devices to send vast amounts of data (e.g. images, audios and videos) over the wireless network to the cloud. This can give rise to considerable energy and latency overheads on the mobile device. Furthermore, pushing all computations toward the cloud can lead to congestion in a scenario where a large number of mobile devices simultaneously send data to the cloud. As a compromise between the mobile-only and the cloud-only approach, recently, a body of research work has been investigating the idea of splitting a deep inference network between the mobile and cloud [6–12]. In this approach, which is referred to as _collaborative intelligence_ , the computations associated with initial layers of the inference network are performed on the mobile device, and the feature tensor (activations) of the last computed layer is sent to the cloud for the remainder of computations. The main motivation for collaborative intelligence is the fact that in many 

applications which are based on convolutional neural networks (CNNs), the feature volume of layers will shrink in size as we go deeper in the model [6, 8, 12]. Therefore, computing a few layers on the mobile and then sending the last computed feature tensor to the cloud can reduce the latency and energy overheads of wireless transfer of the data to the cloud compared to sending the input of the model directly to the cloud. Furthermore, pushing a portion of computations onto the mobile devices can reduce the congestion on the cloud and hence increase its throughput. 

In research studies investigating collaborative intelligence, a given deep network is split between the mobile device and the cloud without any modification to the network architecture itself [6, 8– 12]. In this paper, we investigate altering the underlying deep model architecture to make it collaborative intelligence friendly. For this purpose, we mainly focus on altering the underlying deep model in a way that the feature data size needed to be transmitted to the cloud is reduced. This is because in the studies investigating collaborative intelligence, the latency and energy overheads of the wireless data transfer to the cloud yet play a major role in the total mobile energy consumption and the end-to-end latency [6]. Therefore, reducing the transmitted data size to cloud is generally beneficial. For this purpose, we add a non-expensive learnable _reduction unit_ after the layers assigned to be computed on the mobile device, and the output of this unit is then compressed using conventional compressors (e.g., JPEG) and sent to the cloud. Correspondingly, a decompressor and a learnable _restoration unit_ is added before the layers assigned on the cloud. The main components of reduction and restoration units are convolutional layers which their dimensions are determined in a way that the input of the reduction unit and the output of the restoration unit have the same dimensionality. 

An overview of the proposed method is shown in Fig. 1. The insertion location and size of the the reduction and restoration 



units in the underlying DNN are determined as explained in Section 2.3. Since by inserting the reduction unit, a data bottleneck is created in the model, the combination of the learnable reduction unit, compressor and decompressor, and the learnable restoration unit is referred to as the _bottleneck unit_ , and the new network architecture including the bottleneck unit is referred to as _BottleNet_ , which is trained end-to-end. For the reduction unit, we evaluate and compare dimension reductions along both the channel and spatial dimensions of intermediate feature tensors as explained in Section 2.1. 

As we see in Section 3, an obvious benefit of using the proposed bottleneck unit is in deep models where feature tensor sizes are relatively high, such as ResNet [2]. In such networks, the layer in which the feature size is less than the input size is either not present or lies very deep inside in the network. Therefore, if we want to merely split the network and send the intermediate feature tensor to the cloud as in [6, 8], we need to compute considerable number of layers on the mobile. This will push a major workload to the mobile which will likely result in higher latency and energy consumption compared to the cloud-only approach. This could be the main reason that previous work on collaborative intelligence usually has focused on deep architectures where the intermediate feature size is relatively small compared to the input size after computing only a few layers, such as AlexNet [1] and DeepFace [13]. 

Furthermore, since features of an intermediate layer in a deep model tend to exhibit statistical characteristics such as data redundancy and lower entropy compared to the input of the model, compressing the feature tensor before sending it to the cloud can potentially achieve considerable reductions in the data size needed to be sent over the wireless network. Therefore a major portion of the works studying collaborative intelligence also consider feature compression instead of direct transfer of the feature tensor to the cloud [6, 10–12, 14]. In this work, we consider lossy compression of the feature tensor. Lossy compression methods can lead to higher bit savings compared to the lossless compression approaches. However, they may adversely affect the achieved accuracy and thus lossy compression methods are less studied in the works using feature compression in collaborative intelligence framework, as they mostly use lossless compression techniques on the features before transmitting them to the cloud. In order to compensate for the reduced accuracy due to the lossy compressor, we propose a novel training method for the network which approximates the behavior of the lossy compressor as an identity function in backpropagation. The proposed training method is explained in detail in Section 2.2. In summary, the contributions of this paper are as follows: 

- We propose the bottleneck unit, in which by using a learnable reduction unit followed by a lossy compressor, the feature tensor size required to be transmitted to the cloud is significantly reduced. 

- For training our model, we propose a lossy compressionaware training method in order to compensate for the accuracy loss. 

- Using the proposed BottleNet architecture, we achieve on average 30× improvement in end-to-end latency and 40× 

![Figure](assets/figure_0002_page_0002.svg)**Figure 2: Learnable dimension reduction and restoration units along the (a) channel and (b) spatial dimension of features.** 

![Figure](assets/figure_0003_page_0002.svg)**Figure 3: The bottleneck unit embedded with a nondifferentiable lossy compression (e.g., JPEG).** 

improvement in mobile energy consumption compared to the cloud-only approach with negligible accuracy loss. 

The remainder of the paper is structured as follows: Section 2 provides a more detailed explanation of the bottleneck unit and training method. Section 3 provides the energy and latency improvements of the proposed bottleneck unit and discusses the efficacy of our training approach, and also elaborates on the flexibility of changing the partition point depending on the cloud server congestion and wireless network conditions. Finally, Section 4 concludes the paper. 

## **2 PROPOSED METHOD** 

In this section, first, we describe details of the bottleneck unit. Then, we explain our proposed training method when a non-differentiable lossy compression is applied to the intermediate feature tensor before transmitting it to the cloud. Finally, we explain our approach for finding the best insertion location and size of the bottleneck unit in the underlying deep model to achieve the lowest end-to-end latency and/or mobile energy consumption in different wireless settings. 

## **2.1 Bottleneck Unit** 

For dimension reduction in the feature tensor in the bottleneck unit, we evaluate and compare dimension reductions along either channel or spatial dimensions. The bottleneck unit, referred to as autoencoder in deep learning context, is responsible for learning a dense representation of the features in an intermediate layer. As depicted in Fig. 2, channel-wise reduction, shrinks the number of channels of the features, and spatial reduction shrinks the spatial 

2 



dimensions (width and height) of the features. More specifically, w Forward Propagation w channel-wise reduction unit takes a tensor of size ( _batch_  size_ , _w_ , Compressed _h_ , _c_ ) as input, and outputs a tensor of size ( _batch_  size_ , _w_ , _h_ , _c_<sup>′</sup> ) Features by applying a convolution filter of size (1, 1, _c_ , _c_<sup>′</sup> ) followed by Features h Features h normalization and non-linearity layers. The output tensor of the c Backpropagation c reduction unit is the reduced-order representation of its input ( _c_<sup>′</sup> ≪ ~~<mark>nd</mark>~~ Identity Function _c_ ). Spatial reduction unit takes a tensor of size ( _batch_  size_ , _w_ , _h_ , _c_ ) as input, and outputs a tensor of size ( _batch_  size_ , _w_<sup>′</sup> , _h_<sup>′</sup> , _c_ ) by applying **Figure 4: Embedding non-differentiable compression (e.g.,** a convolution filter of size ( _wf_ , _hf_ , _c_ , _c_ ) followed by normalization **JPEG) in DNN architecture. We approximate the pair of JPEG** and non-linearity layers. The output tensor of the reduction unit is **compressor and decompressor units by identity function to** the reduced-order representation of its input ( _w_<sup>′</sup> < _w_ , and _h_<sup>′</sup> < _h_ ). **make the model differentiable in backpropagation.** In both channel-wise and spatial reduction units, we use ReLU as the non-linearity function. For reduction in the spatial dimension, the stride step of the convolution should be more than one. It should features, _F_ , we use following quantizer: be noted that to cover each neuron during the convolution at least _F_ − _min_ <u>(</u> _F_ <u>)</u> once, the size of this filter should be more than the stride step size, _F_ ˜ = _round_ ( _max_ ( _F_ ) − _min_ ( _F_ )<sup>∗(2</sup><sup>_n_−1))</sup> (1) i.e., _wf_ > _<u>w</u>_<sup>_<u>w</u>_</sup><sup><u>′</u>, and</sup><sup>_hf_></sup> _h_<sup>_<u>h</u>_</sup><sup><u>′</u>. In this paper, we use the same reduction</sup> In addition, the input to the compressors should be reshaped into factor for both width and height, referred to as the spatial reduction 2-D tensors. The features, _F_ , with _C_ channels, are rearranged in factor of _s_ , i.e., _<u>w</u>_<sup>_<u>w</u>_</sup><sup><u>′</u>=</sup> _h_<sup>_<u>h</u>_</sup><sup><u>′</u>=</sup><sup>_s_.</sup> 

In addition, the input to the compressors should be reshaped into 2-D tensors. The features, _F_ , with _C_ channels, are rearranged in a tiled image with the width of 2<sup>_ceil_(</sup><sup><u>1</u></sup> 2<sup>_loд_2(</sup><sup>_C_))</sup> and the height of 2<sup>_f loor_(</sup><sup><u>1</u></sup> 2<sup>_loд_2(</sup><sup>_C_))</sup> to keep the aspect ratio as square as possible to achieve the maximum compression ratio. 

The bottleneck unit architecture uses both spatial and channelwise reduction units followed by a compressor unit on the mobile device to create a compressed representation of the feature tensor which is the tensor transmitted to the cloud. On the cloud, the bottleneck unit uses a decompressor followed by channel-wise and spatial restoration units to restore the dimension of the feature tensor. The detailed architecture of the bottleneck unit is depicted in Fig. 3. The choice of ReLU in reduction units can potentially lead to higher zero rates resulting in higher compression ratios. The bottleneck unit is inserted between two selected consecutive layers of the underlying deep model, where these two layers are selected by an algorithm as explained in Section 2.3. 

## **2.3 Architecture Selection** 

The proposed algorithm, for choosing the location of the bottleneck unit and the proper value of _c_<sup>′</sup> (reflecting the degree of reduction along the channel dimension), and _s_ (reflecting the degree of reduction along the spatial dimension) comprises of three main steps: 1) Training, 2) Profiling, and 3) Selection. We consider placing the bottleneck unit each time after an arbitrary selected layer of the underlying network, for total of _M_ different locations in the network, where _M_ is less than or equal to total _N_ layers of the network. In each of _M_ locations, we train different architectures associated with different degrees of dimension reduction along channel or spatial dimensions, and among those which result in acceptable accuracy levels, we select the one with the minimum bit requirement. We repeat this process for all of _M_ locations. At the end, among _M_ selected models associated with _M_ different partitioning solutions of the network, depending on our optimization target, we choose the best partitioning in terms of minimizing mobile energy consumption and/or end-to-end latency. We measure the latency and mobile energy consumption of computations assigned to the mobile (including reduction and compressor units), wireless transfer of dense compressed feature tensor to the cloud, and computations assigned to the cloud (including decompressor and restoration units). 

## **2.2 Non-differentiable Lossy Compression Aware Training** 

Lossy compression methods result in higher bit savings compared to lossless methods. However, lossy compression is inherently a non-differentiable function. Specifically, quantization is an integral part of the compression and is not differentiable. Introducing non-differentiable functions in a neural network disables the backpropagation because the gradients are not propagated to the layers before the non-differentiable function, resulting in the model not end-to-end trainable. To solve this issue, we introduce a new training method to enable the model to be end-to-end differentiable by defining a gradient for the pair of compressor and decompressor during the backpropagation. In other words, the pair of lossy compressor and decompressor is used as is during the forward propagation, while we treat this pair as an identity function during backpropagation (i.e., gradients passed without any change to the layers before the compressor). Therefore, the whole model will become end-to-end differentiable. The effectiveness of using this training method instead of simply training of the model without considering the compression will be explained in Section 3.3. 

The detailed algorithm for choosing the location of the bottleneck unit and the proper amount of reductions in channel and spatial dimensions is presented in Algorithm 1. 

## **3 EVALUATION** 

## **3.1 Experimental Setup** 

We evaluate our proposed method on NVIDIA Jetson TX2 board [15] equipped with NVIDIA Pascal™ GPU with 256 CUDA cores which fairly represents the communication and computation resources of mobile devices. Our server platform is equipped with a NVIDIA Geforce® GTX 1080 Ti GPU, which has almost 30x more computing 

The input to the compressors are typically quantized to unsigned n-bit numbers by an uniform quantizer. Similar to [11], to quantize 

3 



![Figure](assets/figure_0004_page_0004.svg)computations associated with the decompressor, restoration unit, and the remaining DNN layers. Upon the completion of the the execution of last DNN layer on the cloud, the inference result is sent back to the mobile device. For the choice of our lossy compressor, here, we use JPEG compression. 

For evaluating our proposed method, we use ResNet-50 as our underlying deep model, which is one of the widely used deep models allowing training very deep networks by using skip connections in residual blocks (RBs). ResNet architecture comes with flexible number of layers such as 34, 50, 101. There are 16 RBs in ResNet-50. Using algorithm 1 explained in Section 2.3, we obtain 16 different models where each model is associated with placing the bottleneck unit after one of the 16 available RBs. As presented in algorithm 1, the obtained _c_<sup>′</sup> and _s_ of the bottleneck unit, when it is placed after each of 16 RBs, corresponding to 16 different partitions, could be different, where _c_<sup>′</sup> and _s_ were corresponding to different degrees of reductions in channel and spatial dimensions, respectively. The architecture of ResNet-50 is presented in Fig. 5. The input image size of the model and the size of output feature tensor of each RB are presented in Fig. 6. As indicated in Fig. 6, the size of intermediate feature tensors in ResNet-50 are larger than the input size up until RB14, which is relatively deep in the model. Therefore, merely splitting this network between the mobile and cloud for collaborative intelligence may not perform better than the cloud-only approach in terms of latency and mobile energy consumption, since a large portion of the workload is pushed toward the mobile. 

**Figure 6: Input image size of the model and the size of output feature tensor of each residual block in ResNet-50.** 

power compared to our mobile platform. The detailed specifications of our mobile and server platforms are presented in Table 1 and Table 2, respectively. We measure the GPU power consumption on our mobile platform using INA226 power monitoring sensor with sampling rate of 500 KHz [16]. For the wireless network settings, the average upload speed of different wireless networks, 3G, 4G, and Wi-Fi, in the U.S. are used in our experiments [17, 18]. We use the transmission power models of [19] for wireless networks with estimation error rate of less than 6%. The power level for up-link is estimated by _Pu_ = _αutu_ + _β_ , where _tu_ is the up-link throughput, and _αu_ and _β_ are regression coefficients of power models. The values for our power model parameters are presented in Table 3. 

We prototype the proposed method by implementing the inference networks for both the mobile device and cloud server using NVIDIA TensorRT™ [20], which is the state-of-the-art platform for high-performance deep learning inference. It includes a deep learning inference optimizer and run-time that delivers low latency and high-throughput for deep learning inference applications. TensorRT is equipped with cuDNN[21], a GPU-accelerated library of primitives for deep neural networks. TensorRT supports three precision modes for creating the inference graph, namely FP32 (single precision), FP16 (half precision), and INT8 (8-bit integer). However, our mobile device does not support INT8 operations on its GPU for inference. Therefore, we use FP16 mode for creating the inference graph from the trained model graph, where for the training itself single precision mode is used. As demonstrated in [22], 8- bit quantization would be enough for even challenging tasks like ImageNet [23] classification. Therefore, we apply 8-bit quantization on FP16 data types, using the uniform quantizer presented in Section 2.2, before applying the lossy compression on them. We implement our client-server interface using Thrift [24], an open source flexible RPC interface for inter-process communication. Given a partition decision, execution begins on the mobile device and cascades through the layers of the DNN leading up to that partition point. Upon completion of that layer and the reduction unit and lossy compressor, mobile sends the reduced dense feature tensor from the mobile device to the cloud. Cloud server then executes the 

For our dataset, we use miniImageNet [25], a subset of ImageNet, which includes 100 classes and 600 examples per each class. 85% of whole dataset examples are used as the training set, and the rest as the test set. We randomly crop a 224×224 region from each sample for data augmentation. For training of our models, we use 90 epochs of data. 

## **3.2 Latency and Energy Improvements** 

The accuracy of ResNet-50 model for miniImageNet dataset without the bottleneck unit is 76%, which we refer to as the target accuracy. By assuming an acceptable accuracy loss of 2% compared to the target accuracy, using algorithm 1, placing the bottleneck unit after RBs 1-3, 4-7, 8-13, and 14-16, requires the reduced channel size _c_<sup>′</sup> of 1, 2, 5, and 10, respectively (placing the bottleneck unit after different RBs corresponds to different partitions). For all 16 partitions, the spatial factor reduction of _s_ is obtained as 2 by algorithm 1. According to our experiments, defining the acceptable accuracy loss of 2% allows channel-wise and spatial reduction units achieve significant dimension reductions and thus bit savings. For instance, when the bottleneck unit is placed after RB1, feature tensor of size 

4 



### **Algorithm 1:** The partitioning algorithm for BottleNet 

- **1 Inputs: 2** _N_ : number of layers in the DNN **3** _M_ : number of partitioning points in the DNN ( _M_ ≤ _N_ ) **4** _bottleneck_ ( _s_ , _c_<sup>′</sup> ): A bottleneck unit with the spatial reduction factor of _s_ and reduced channel size of _c_<sup>′</sup> 

- **5** _Smax_ : maximum allowable spatial reduction factor **6** _Cmax_<sup>′</sup> : maximum allowable reduced channel size **7** _Kmobile_ : current load level of mobile **8** _Kcloud_ : current load level of cloud **9** _tmobile_ , _pmobile_ ( _j_ , _Kmobile_ )| _j_ = 1.. _M_ : latency and power on the mobile corresponding to partition _j_ and load _Kmobile_ 

- **10** _tcloud_ ( _j_ , _Kcloud_ )| _j_ = 1.. _M_ : latency on the cloud corresponding to partition _j_ and load _Kcloud_ 

- **11** _NB_ : wireless network bandwidth **12** _PU_ : wireless network up-link power consumption **13 Outputs: 14** Best partitioned model **15 Variables: 16** { _Dj_ | _j_ = 1.. _M_ }: compressed feature size in each of _M_ partitioning solutions 

- **17 18** // Training phase **19 for** _j_ = 1; _j_ ≤ _M_ ; _j_ = _j_ + 1 **do 20 for** _c_<sup>′</sup> = 1; _c_<sup>′</sup> ≤ _Cmax_<sup>′</sup> ; _c_<sup>′</sup> = _c_<sup>′</sup> + 1 **do 21 for** _s_ = 1; _s_ ≤ _Smax_ ; _s_ = _s_ + 1 **do 22** Place _bottleneck_ ( _s_ , _c_<sup>′</sup> ) after j-th layer **23** Train() **24** Store the corresponding model and its accuracy **25 end 26 end 27 end 28 for** _j_ = 1; _j_ ≤ _M_ ; _j_ = _j_ + 1 **do 29** For those models that the bottleneck unit is placed after _j_ -th layer, among ones with acceptable accuracy, store the one with minimum compressed feature size and store its compressed feature size as _Dj_ 

- **30 end 31 32** // Profiling phase **33 for** _j_ = 1; _j_ ≤ _M_ ; _j_ = _j_ + 1 **do 34** _TMj_ = _tmobile_ ( _j_ , _Kmobile_ ) **35** _PMj_ = _pmobile_ ( _j_ , _Kmobile_ ) **36** _TCj_ = _tcloud_ ( _j_ , _Kcloud_ ) **37** _TUj_ = _Dj_ / _NB_ **38 end 39 40** // Selection phase **41 if** _target is min latency_ **then 42** return _arдminj_ =1.. _M_ ( _TMj_ + _TUj_ + _TCj_ ) **43 end 44 if** _target is min energy_ **then 45** return _arдminj_ =1.. _M_ ( _TMj_ × _PMj_ + _TUj_ × _PU_ ) **46 end** 

**Table 1: Mobile device specifications** 

|**Component**|**Specification**|
|---|---|
|System|NVIDIAJetson TX2 Developer Kit|
|GPU|NVIDIA Pascal™, 256 CUDA cores|
|CPU|HMP Dual Denver +Quad ARM® A57/2 MB L2|
|Memory|8 GB 128 bit LPDDR4 59.7 GB/s|



**Table 2: Server platform specifications** 

|**Component**|**Specification**|
|---|---|
|GPU|NVIDIA Geforce® GTX 1080 Ti, 12GB GDDR5|
|CPU|Intel® Xeon® CPU E7- 8837@2.67GHz|
|Memory|64 GB DDR4|



**Table 3: Wireless networks parameters** 

|**Param.**|**3G**|**4G**|**Wi-Fi**|
|---|---|---|---|
|_tu_ (Mbps)|1.1|5.85|18.88|
|_αu_ (mW/Mbps)|868.98|438.39|283.17|
|_β_ (mW)|817.88|1288.04|132.86|



(56, 56, 256) is reduced to (28, 28, 1) using the channel-wise and spatial reduction units. 

Table 4 presents the latency and mobile energy consumption by placing the bottleneck unit with obtained _c_<sup>′</sup> and _s_ values (for the accuracy loss less than 2%) after each residual block, for different wireless networks when there is no congestion in the mobile, cloud, and wireless network. The best partitions in terms of end-to-end latency and mobile energy consumption across different wireless settings are highlighted in this Table. Table 5 compares highlighted partitions with the mobile-only and cloud-only approaches in terms of latency and energy. For the cloud-only approach, before transmitting the input to the cloud, we apply JPEG compression on the input images which are stored in 8-bit RGB format. Note that the best partitioning for the goal of minimum end-to-end latency is the same as the best partitioning for the goal of minimum mobile energy consumption in each wireless network settings. This is mainly due to the fact that end-to-end latency and mobile energy consumption are proportional to each other since the dominant portion of both of them are associated with the wireless transfer overheads of the intermediate feature tensor. 

**Latency Improvement** - As demonstrated in Table 5, using our proposed method, the end-to-end latency achieves 63×, 21×, 8× improvements over the cloud-only approach in 3G, 4G, and Wi-Fi networks, respectively. 

**Energy Improvement** - As demonstrated in Table 5, using our proposed method, the mobile energy consumption achieves 47×, 41×, and 31× improvements over the cloud-only approach in 3G, 4G, and Wi-Fi networks, respectively. 

As observed in Table 4, the best partition across all wireless network settings is associated with placing the bottleneck unit after RB1. This is an important result since as explained before and according to Fig. 6, due to the relatively large sizes of intermediate 

5 



**Table 4: The end-to-end Latency, mobile energy consumption, and offloaded data size for different partition points in ResNet50 using the proposed method** 

|**Layer**|RB1|RB2|RB3|RB4|RB5|RB6|RB7|RB8|RB9|RB10|RB11|RB12|RB13|RB14|RB15|RB16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Offloaded Data(B)**|316|317|314|166|171|168|170|96|90|98|101|101|95|52|52|53|
|**Latency 3G(ms)**|3.1|4.1|4.9|5.2|6.3|7.5|8.2|9.6|10.7|11.6|12.8|13.4|14.8|15.1|16.0|17.1|
|**Energy 3G(mJ)**|6.6|7.6|8.1|9.7|10.8|11.9|12.6|13.9|14.1|15.8|16.1|17.6|18.5|19.8|20.7|21.9|
|**Latency 4G(ms)**|1.8|2.5|3.3|4.2|5.0|5.9|6.9|8.6|9.4|10.3|11.9|12.7|14.1|15.0|15.7|16.9|
|**Energy 4G(mJ)**|4.1|6.8|7.0|8.9|10.6|11.3|12.9|13.1|14.0|15.6|16.0|17.1|18.3|19.1|20.3|21.2|
|**Latency Wi-Fi(ms)**|1.6|2.4|3.0|4.1|4.9|5.8|6.8|8.5|9.3|10.1|11.8|12.6|14.0|14.9|15.7|16.9|
|**Energy Wi-Fi(mJ)**|3.5|5.6|6.1|7.4|9.5|10.8|12.3|12.5|13.8|14.9|15.6|16.9|18.1|19.0|20.1|21.0|



**Table 5: Comparison of the proposed method with mobile-only and cloud-only approaches** 

|Setup||Latency (ms)|Energy (mJ)|Bottleneck Unit Location|Offloaded Data(B)|Accuracy|
|---|---|---|---|---|---|---|
|Mobile-only|-|15.7|20.5|-|0|76.1|
||3G|196.2|310.1|-|26766|76.1|
|Cloud-only|4G|37.9|168.3|-|26766|76.1|
||Wi-Fi|13.1|110.7|-|26766|76.1|
||3G|3.1|6.6|After RB1|316|74.1|
|BottleNet|4G|1.8|4.1|After RB1|316|74.1|
||Wi-Fi|1.6|3.5|After RB1|316|74.1|



feature tensors in ResNet-50 compared to the input image size (especially RB1 which has the largest feature size), merely splitting the network between the mobile and cloud and transmitting the intermediate feature tensor to the cloud may not perform better than the cloud-only approach in terms of latency and mobile energy consumption. However, using our proposed method, mobile device can compute only one RB and send the reduced dense output feature tensor to the cloud, achieving minimum latency and mobile energy consumption among all possible partitions and significant improvements compared to the cloud-only approach, while the acceptable accuracy is still reached. 

## **3.3 The Efficacy of Compression-aware Training** 

Incorporating the pair of JPEG compressor and decompressor as a new computational unit in a neural network can be performed in two ways: 1) Placing the compression unit in a given trained model (Naive), 2) Training the model from scratch using the proposed compression-aware training method as explained in Section 2.2. In JPEG compressor, which is the choice of lossy compressor for our experiments, changing a parameter named _quality_ affects the amount of output bits of the compressor. Fig. 7 presents the accuracy loss obtained for ResNet-50 when the bottleneck unit is placed after RB1, versus different number of output bits of the compressor (corresponding to different values of the JPEG quality parameter, ranging from 1 to 100). As depicted in Fig. 7, the accuracy loss of the compression-aware training becomes almost zero by having a JPEG quality level of higher than 20, while the accuracy loss of the naive approach is close to 18% using the quality level of 20. In our experiments, we use the quality level of 20 for JPEG compression in order to achieve maximum bit savings while there is no accuracy loss. 

![Figure](assets/figure_0005_page_0006.svg)**Figure 7: The comparison between the accuracy loss of the proposed compression-aware training and a naively trained model for different compressed feature size values, when the bottleneck unit is placed after RB1.** 

## **3.4 Server Load Variations** 

Data centers usually experience fluctuating load patterns. High server utilization can lead to increased service times for DNN queries. Therefore, it can be desired sometimes to change the point of partition at the run time and push more work toward the mobile device to decrease the current server load level. In order to allow for flexibility in the dynamic selection of partition points, both the mobile and cloud can host all possible _M_ partitioned models obtained via the training phase of algorithm 1. For each of _M_ models, the mobile and cloud store only their assigned computing portion of the inference network. Depending on the server load, using the profiling and selection phase of algorithm 1, partition point can be changed while still providing an acceptable accuracy and offloading 

6 



less data compared to the cloud-only approach. This shows the efficacy of the learnable reductions in channel and spatial dimensions and using the compression-aware training method to avoid long latency of DNN queries caused by high user demands and server congestion. The best model (partition) can be selected at run-time by the mobile by periodically pinging the server during the mobile idle period. 

## **3.5 Comparison to Other Feature Compression Techniques** 

In comparison with other works in collaborative intelligence framework which have considered the compression of intermediate features before uploading them to the cloud, our proposed method can achieve significantly higher bit savings compared to the cloud-only approach. For instance, as reported in [12], which is one of the few works in collaborative intelligence literature which consider lossy compression of features before transmitting them to cloud, communication overhead in their work can be reduced up to 70% compared to the cloud-only approach. However, in our work, with the proposed trainable reduction unit for spatial and channel dimensions, and the proposed lossy compression-aware training method, we can achieve up to 84× bit savings compared to the cloud-only approach according to Table 5. This shows that in collaborative intelligence framework, adding a learnable reduction in channel and spatial dimension alongside with a compression-aware training method can significantly perform better than merely splitting a network with fixed weights and compressing the intermediate feature tensor before uploading to the cloud. 

## **4 CONCLUSION AND FUTURE WORK** 

Recent studies have shown that the latency and energy consumption of deep neural networks in mobile applications can be considerably reduced by splitting the network between the mobile and cloud in a collaborative intelligence framework. In this work, we develop a new partitioning scheme that creates a bottleneck in a neural network using the proposed bottleneck unit, which considerably reduces the communication costs of feature transfer between the mobile and cloud. Our proposed method can adapt to any DNN architecture, hardware platform, wireless network settings, and mobile and server load levels, and selects the best partition point for the minimum end-to-end latency and/or mobile energy consumption at run-time. The new network architecture, including the introduced bottleneck unit after a selected layer of the underlying deep model, is trained end-to-end using our proposed compression-aware training method which allows significant bit savings while providing an acceptable accuracy. Our proposed method, across different wireless network settings, achieves on average 30× improvements for end-to-end latency and 40× improvements for mobile energy consumption compared to the cloud-only approach for ResNet-50, while the accuracy loss is less than 2%. 

## **REFERENCES** 

- [1] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In _Advances in neural information processing systems_ , pages 1097–1105, 2012. 

- [2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. _CoRR_ , abs/1512.03385, 2015. 

- [3] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Region-based convolutional networks for accurate object detection and segmentation. _IEEE transactions on pattern analysis and machine intelligence_ , 38(1):142–158, 2016. 

- [4] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. _IEEE Signal processing magazine_ , 29(6):82–97, 2012. 

- [5] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In _Advances in neural information processing systems_ , pages 3111–3119, 2013. 

- [6] A. E. Eshratifar, M. S. Abrishami, and M. Pedram. Jointdnn: an efficient training and inference engine for intelligent mobile cloud computing services. _arXiv preprint arXiv:1801.08618_ , 2018. 

- [7] A. E. Eshratifar and M. Pedram. Energy and performance efficient computation offloading for deep neural networks in a mobile cloud computing environment. pages 111–116, 2018. 

- [8] Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars, and L. Tang. Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. _ACM SIGPLAN Notices_ , 52(4):615–629, 2017. 

- [9] P. M. Grulich and F. Nawab. Collaborative edge and cloud neural networks for real-time video processing. _Proceedings of the VLDB Endowment_ , 11(12):2046–2049, 2018. 

- [10] Z. Chen, W. Lin, S. Wang, L. Duan, and A. C. Kot. Intermediate deep feature compression: the next battlefield of intelligent sensing. _arXiv preprint arXiv:1809.06196_ , 2018. 

- [11] H. Choi and I. V. Bajic. Near-lossless deep feature compression for collaborative intelligence. _arXiv preprint arXiv:1804.09963_ , 2018. 

- [12] H. Choi and I. V. Bajic. Deep feature compression for collaborative object detection. _arXiv preprint arXiv:1802.03931_ , 2018. 

- [13] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. Deepface: Closing the gap to human-level performance in face verification. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1701–1708, 2014. 

- [14] A. E. Eshratifar, A. Esmaili, and M. Pedram. Towards collaborative intelligence friendly architectures for deep learning. _arXiv preprint arXiv:1902.00147_ , 2019. 

- [15] Jetson TX2 Module. https://developer.nvidia.com/embedded/buy/jetson-tx2, 2018. 

- [16] INA Current/Power Monitor. http://www.ti.com/product/INA226. 

- [17] State of Mobile Networks in USA. https://opensignal.com/reports/2017/08/usa/ state-of-the-mobile-network, 2017. 

- [18] United States Speedtest Market Report. http://www.speedtest.net/reports/ united-states/, 2017. 

- [19] J. Huang, F. Qian, A. Gerber, Z. M. Mao, S. Sen, and O. Spatscheck. A close examination of performance and power characteristics of 4g lte networks. In _Proceedings of the 10th International Conference on Mobile Systems, Applications, and Services_ , MobiSys ’12, pages 225–238, New York, NY, USA, 2012. ACM. 

- [20] NVIDIA TensorRT. https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index. html, 2018. 

- [21] S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer. cudnn: Efficient primitives for deep learning. _CoRR_ , abs/1410.0759, 2014. 

- [22] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. _CoRR_ , abs/1510.00149, 2015. 

- [23] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In _Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on_ , pages 248–255. Ieee, 2009. 

- [24] M. Slee, A. Agarwal, and M. Kwiatkowski. Thrift: Scalable cross-language services implementation. _Facebook White Paper_ , 5, 2007. 

- [25] O. Vinyals, C. Blundell, T. P. Lillicrap, K. Kavukcuoglu, and D. Wierstra. Matching networks for one shot learning. _CoRR_ , abs/1606.04080, 2016. 

For future work, the adaptability of our proposed method to various underlying deep models should be studied, as well as the usage of other compression techniques rather than JPEG in the bottleneck unit. Furthermore, the extent of reduction in the feature tensor dimension can be explored further using other architectures for the learnable reduction unit. 

7 

