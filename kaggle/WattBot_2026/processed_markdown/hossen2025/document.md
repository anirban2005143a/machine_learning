An Efficient Classification Model for Cyber Text 

Md Sakhawat Hossen<sup>1</sup> , Md. Zashid Iqbal Borshon<sup>2*</sup> , A. S. M. Badrudduza<sup>2</sup> 

> 1*Department of EEE, Chittagong University of Engineering & Technology, Chittagong, 4349, Bangladesh. 

> 2Department of ETE, Rajshahi University of Engineering & Technology, Rajshahi, 6204, Bangladesh. 

*Corresponding author(s). E-mail(s): zashidiqbal1554@gmail.com; Contributing authors: sakhawat3003@gmail.com; asmb.kanon@gmail.com; 

#### **Abstract** 

The uprising of deep learning methodology and practice in recent years has brought about a severe consequence of increasing carbon footprint due to the insatiable demands on computational resources and power. The field of text analytics also experienced a massive transformation on this trend of monopolizing methodology. In this paper, the original TF-IDF algorithm has been modified and Clement Term Frequency–Inverse Document Frequency (CTF-IDF) has been proposed for data preprocessing. This paper primarily discusses on the effectiveness of classical machine learning techniques in text analytics with CTF-IDF and faster IRLBA algorithm for dimensionality reduction. The introduction of both of these techniques in the conventional text analytics pipeline ensures a more efficient, faster, and less computationally intensive application when compared with deep learning methodology regarding carbon footprint with minor compromise in accuracy. The experimental results also exhibit a manifold of reduction in time complexity and improvement of model accuracy for the classical machine learning methods discussed further in this paper. 

**Keywords:** Tf-Idf, Ctf-Idf, IRLBA, Dimensionality Reduction, Text Analytics, BERT, SPAM 

1 



# **1 Introduction** 

Since the advent of modern technology and the internet, the ubiquitous application of electronic media in academia and research, news publications, social media, government, and non-government sites has massively contributed to the upsurge of text data stored in digital appliances. To extract the essential information from this highly unstructured data, we must first employ a variety of data mining techniques to uncover potentially valuable patterns from this enormous amount of data. Text analytics is the process of retrieving unstructured data and transforming it into structured data with the application of suitable algorithms to find patterns and trends, and classify the texts into distinct groups [1]. The standard text analytics methodology can be burdensome for any small machine when dealing with big unstructured data. Contemporary text classification task requires a copious amount of text documents in each training session. Consequentially, the feature space may explode with sparse and redundant data when transformed into a document frequency matrix. This ultimately results in a heavy toll on computational power and the time required to build any machine learning and deep learning model for the prediction and classification of text data. This is proverbially known as the curse of dimensionality. On the other hand, Deep Learning methods like Bidirectional LSTM and Transformer based models like Google’s BERT have shown significant improvements in the precision of text analysis but at a cost of huge computational time and resources, therefore aggravating the issue of carbon footprint. 

Term Frequency–Inverse Document Frequency (TF-IDF) is considered one of the stepping stones for transforming tokenized textual data. According to a 2015 survey, TF-IDF is used by 83% of recommender systems based on textual data in digital libraries [2]. This statistical metric quantifies the significance of a word in a corpus or collection of documents [3]. The classical TF-IDF severely penalizes each word/token in the documents based on the frequency of the words among the whole corpus on a logarithmic scale which in return creates a wide range of TF-IDF values and sometimes diminishes the whole effect of various keywords [4]. We have proposed a moderate and clement approach to address this issue with CTF-IDF. The experimental results showed improvement regarding model accuracy in implementing CTF-IDF over the classical TF-IDF on text classification tasks when combined with the proposed algorithm for dimensionality reduction. 

Dimensionality reduction is the method of converting high-dimensional data into a meaningful representation of lesser dimensionality [5]. This method is considered essential for transforming the features into a more compact form to increase the learning efficiency of the algorithms when the number of features exceeds significantly. Dimension reduction techniques can be utilized both with supervised and unsupervised methods. However, depending on the kind of method utilized, the properties of the dimensionality reduction technique change. For instance, dimensionality reduction techniques for unsupervised learning algorithms should work to reduce the loss of feature information. On the other hand, the goal should be to maximize class information in the case of supervised learning. There is no single strategy that works in every circumstance due to the complexity of the dimension reduction process. As a 

2 



result, numerous dimension reduction techniques have been developed and proposed over the years and put to the test in various fields of study and application domains. 

In this paper, we adopted “The augmented implicitly restarted Lanczos bidiagonalization algorithm” for dimensionality reduction [6]. This algorithm computes partial singular values decomposition and finds a few of the largest or smallest singular values along with the singular vectors of a sparse or dense matrix. This is a fast and memory-efficient method that serves to alleviate the problem of employing complex machine-learning algorithms and improving the overall per-formance of the models at the same time. 

For the initial development of the methodology, a SPAM dataset was used that consists of 5000 text data collected primarily from phones as text messages [7]. The dataset was classified into two basic categories: Spam and Ham (not Spam). After the development of the methodology on this dataset, a comparative analysis was done on another similar SMS Phishing dataset [8] in terms of model accuracy and run time. For the analysis, classical machine learning techniques like Decision Tree and Support Vector Machine have been used to evaluate the robustness and efficiency of the proposed methodology in contrast with the traditional methodology in text analytics and deep learning model like Transformer (BERT). 

The main contributions of this paper are as follows: 

- Introducing a modified data processing algorithm CTF IDF to reduce the penalty received by each term in the corpus. 

- Incorporation of a faster and memory-efficient “Augmented implicitly restarted Lanczos bidiagonalization” algorithm for dimensionality reduction. 

- Thecombined effect of both of these methods in the traditional text analytics pipeline expedited the computational time and reduced the requirement for computational resources. 

- Interpretability of the Cybertexts based on features that influence the classification of texts as either spam or non spam (ham). 

The proposed method is efficient on addressing the issue of the rising carbon footprint due to the advent of Deep Learning methodology while still increasing the robustness of the trained models. 

# **2 Background Study** 

## **2.1 Previous Works** 

Xia Hu _et al._ elaborately discussed the traditional methodology for text analytics, which consists of three key components: text preprocessing, text representation, and knowledge discovery [9]. 

Our work mainly focuses on the representation stage with a new TF-IDF and a faster method for dimensionality reduction of the vector space. TF-IDF (Term Frequency-Inverse Document Frequency) is traditionally used as a statistical method for evaluating the importance of a word in a document in relation to a corpus of documents. TF-IDF has been widely used in text classification, such as spam filtering and 

3 



![Figure](assets/figure_0001_page_0004.svg)**Fig. 1** Traditional framework for text analytics. 

sentiment analysis. Research has shown that TF-IDF often outperforms other feature selection methods, such as a bag of words and n-grams [10]. TF-IDF has also been used for keyword extraction, as it assigns high weight to important terms in a document and can identify the most relevant words for summarizing a document [11]. In high-dimensional text data, TF-IDF is useful for dimensionality reduction, as it reduces the number of features while retaining important information [12]. In Recent research, TF-IDF has been combined with word embedding methods to improve the performance of text classification tasks [13]. Hybrid Approaches like merging TF-IDF with other methods, such as word2vec, to improve its performance in text analysis [14]. This paper introduces a modified TF-IDF for data representation. 

On the other hand, dimensionality reduction is an essential technique in text analytics for reducing the high dimensionality of textual data while retaining its most informative features. Principal Component Analysis (PCA) is a commonly used dimensionality reduction technique that involves projecting data onto a lower-dimensional space while retaining as much variance as possible. In text analytics, PCA has been used for tasks such as sentiment analysis, document classification, and topic modeling [15] [16] [17]. Latent Dirichlet Allocation (LDA) is a generative probabilistic model that discovers latent topics in a corpus of text. LDA has been used for tasks such as topic modeling, document classification, and information retrieval. [18] [19] [20]. Non-negative Matrix Factorization (NMF) is a matrix decomposition technique that factorizes a matrix into two non-negative matrices, which can be interpreted as representing latent topics and word distributions. NMF is suitable for tasks such as topic modeling, document clustering, and sentiment analysis [21] [22]. Singular Value Decomposition (SVD) is a matrix factorization technique used to decompose a matrix into its constituent parts. It is applied mostly in document classification, topic modeling, and information retrieval [23]. For visualization of high-dimensional word embedding and document clustering t-distributed Stochastic Neighbor Embedding (tSNE) is preferred [24] [25] [26]. Word2Vec is frequently used for tasks such as text classification, sentiment analysis, and information retrieval [27] [28]. Another modern tool FastText has been applied in text classification, named entity recognition, and sentiment analysis [29] [30]. GloVe is a technique that learns word embedding by factorizing a matrix of word co-occurrence statistics that has been implemented to find word similarity and text classification [31]. 

Dimension reduction methods have been proven to be crucial for many text analytics tasks, and the choice of method depends on the specific task and the characteristics 

4 



of the data. PCA, LDA, NMF, SVD, t-SNE, Word2Vec, FastText, and GloVe are some of the popular dimension reduction methods used in text analytics. In this paper, we experimented with a faster singular value decomposition method for dimensionality reduction. 

# **3 Proposed Method** 

The aim of this experiment is to provide an improved and efficient methodology in text analytics with classical machine learning algorithms. The basic framework consists of data preprocessing, feature extraction with Ctf-idf, projection of a Ctf-idf document vector into the SVD semantic space with IRLBA, and classification stages. The following subsection will provide a brief explanation of the success measure as well. 

![Figure](assets/figure_0002_page_0005.svg)**Fig. 2** Proposed method of the research. 

## **3.1 Dataset Details** 

A smaller and more compact SPAM dataset has been used for the development of the method, and the SMS Phishing dataset has been used as a benchmark dataset to check the robustness of the method. 

### **3.1.1 Spam Dataset** 

The SMS Spam Collection consists of labeled SMS messages gathered for studying mobile phone spam [7]. It is publicly available for research purposes. The dataset 

5 



contains 5,574 authentic English text messages that are not encoded. These messages are categorized as either legitimate (ham) or spam. According to Collins Dictionary, spam messages are unsolicited electronic mail or messages sent simultaneously to a number of email addresses or mobile phones. 

### **3.1.2 SMS Phishing Dataset** 

The dataset consists of categorized text messages gathered for SMS phishing investigation. It comprises 5,971 text messages labeled as either ”Legitimate” (Ham), ”Spam”, or ”Smishing”. The collection encompasses 489 spam messages, 638 smishing messages, and 4,844 ham messages. For this research, the text documents labeled as ‘Smishing’ and ‘Spam’ are grouped together as ‘Spam’. This dataset contains raw message content that can be preprocessed for extracting further attributes to be followed by classification using Machine Learning techniques or can be used as labeled data in Deep Learning. Additionally, the dataset includes extracted attributes from malicious messages that aid in classifying messages as either malicious or legitimate. 

## **3.2 Preprocessing techniques** 

The text documents will undergo pre-processing which involves various tasks such as tokenization, removal of stop words, conversion to lowercase, and stemming. Tokenization refers to dividing a text into tokens such as words or phrases. Stop words are words that appear frequently in texts, such as conjunctions or prepositions, regardless of the topic. Lowercase conversion involves changing all uppercase letters to lowercase letters before the classification stage. Stemming is the process of obtaining the root or stem of derived words, and the commonly used stemming process for English is Porter Stem, which was introduced by N. Milic-Frayling [32]. 

## **3.3 Feature Extraction** 

### **3.3.1 Document Frequency matrix** 

Tokenization of the corpus is followed by the creation of a document frequency matrix to represent the connection between terms and documents. In this matrix, each row corresponds to a document, while each column corresponds to a term, and the value entered represents the frequency of the term’s occurrence within that particular document. 

### **3.3.2 N-gram modeling** 

n-gram models are now widely applied in many computational fields including text analytics. There are many variants of n-grams depending on the sequential order. In order to reduce the load of quantitative analysis and sparsely distributed data, only the unigram model has been chosen to follow through the experiment. 

### **3.3.3 Modified TF-IDF (CTF-IDF)** 

TF-IDF, short for Term Frequency-Inverse Document Frequency, combines two distinct measurements, TF and IDF, to analyze multiple documents. When dealing with 

6 



multiple documents, TF-IDF is employed, leveraging the notion that uncommon and infrequent words provide greater insights into the content of a document compared to frequently occurring words across all documents. A modified CTF-IDF algorithm is proposed for this experiment. The modified algorithm assigns greater IDF values for the rarer terms in the whole corpus by calculating through the inverse hyperbolic sine function. The most infrequent terms in the corpus convey the most significance in classifying the document whereas the common terms should have very little significance in determining the nature of the document. In that case, CTF-IDF is more prominent in leveraging the rarity of any term and assigning much higher IDF value attached to it. CTF-IDF is also less inclement on penalizing the most frequent words so the CTF-IDF value of any word never diminishes. Not losing any information from the corpus is necessary on the application of matrix decomposition in the later steps during dimensionality reduction. The mathematical details are given below, 



$$
tf(t, d) = fd(t) max w\in{}{}d fd(w) (1)
$$



$$
idf(t, D) = arcsinh � |D| |{d \in{}{}D : t \in{}{}d}| � (2)
$$



$$
tfidf(t, d, D) = tf(t, d) ∗idf(t, D) (3)
$$



$$
tfidf ′(t, d, D) = idf(t, D) |D| + tfidf(t, d, D) (4)
$$

Here, 

• _fd_ ( _t_ ) = Frequency of term _t_ in document _d_ 

- _D_ = Corpus of documents 

### **3.3.4 Dimensionality Reduction with IRLBA** 

Another fundamental aspect of this experiment is to reduce the training time of each learning algorithm as much as possible while preserving accuracy. The proposed method of augmented implicitly restarted Lanczos bidiagonalization algorithm (IRLBA) is an extension of the Lanczos bidiagonalization algorithm that finds an estimated number of largest or the smallest singular values and corresponding singular vectors of a sparse or dense matrix using the mechanism of Baglama and Reichel [6]. It is a fast and memory-efficient method for truncated singular value decomposition and principal components analysis [33]. 

In this study, the transformation through IRLBA was iterated over many numbers of right singular vectors and an optimum 300 most significant right singular vectors have been chosen based on the descending order of singular values. The number of iterations determines the number of desired singular values to compute. The mathematical formulation is provided below. Given an input matrix _A_ of size _m × n_ , where _i_ = _n_ , this algorithm iteratively constructs two matrices _B_ and _C_ , both bidiagonal, such that _B_ is similar to _A_ . 

7 



- **Initialization** 

– Choose a starting vector _v_ 1 of size _m×_ 1 with unit norm: _∥v_ 1 _∥_ = 1 – Set _β_ 0 = 0 and _v_ 0 = 0 

- **Iteration** 

For _k_ = 1 to _p_ (where _p_ is the desired number of singular values): 

– Compute _wk_ = _A · vk − βk−_ 1 _· vk−_ 1. – _αk_ = _∥wk∥_ . – Normalize _wk_ : _vk_ +1 =<sup>_wk_</sup> . _αk_ – Compute _zk_ = _A_<sup>_T_</sup> _· vk_ +1 _− αk · vk_ . – _βk_ = _∥zk∥_ . – Normalize _zk_ : _uk_ +1 =<sup>_zk_</sup> . _βk_ 

- **Implicit Restart** 

– Compute the bidiagonalization of _B_ and _C_ for the first _p_ iterations using the Lanczos bidiagonalization algorithm. 

- **Augmentation** 

– Compute the singular value decomposition of the bidiagonal matrix _C_ of size _p × p_ : _C_ = _U · S · V_<sup>_T_</sup> . 

- **Implicit Restart (continued)** 

– Set _B_ = _U_<sup>_T_</sup> _· B · V_ , which updates _B_ to be more similar to _A_ . – Repeat Steps 2–5 until convergence or desired accuracy is achieved. 

At the end of the algorithm, B and C will be similar, and the singular values of A can be computed from the singular values of C. 

## **3.4 Classification Methods** 

For simplicity, Support Vector Machine and Decision Tree Classifiers have been chosen as the classical machine learning algorithms for classification in this experiment. On the other hand, the deep learning-based Transformer model BERT has been employed for comparative purposes with the proposed methodology. 

SVM is a learning algorithm designed for solving two-group classification problems, as originally introduced by [34]. In this case, SVM is employed to categorize texts into positive or negative classes. SVM is particularly effective for text classification due to its ability to handle a large number of features with few examples when the problems can be linearly separated, as stated in [35]. 

Decision tree classifiers are widely recognized as one of the most popular and prominent approaches for representing classifiers in data classification. Decision Trees often replicate human cognitive processes when making decisions, thereby making them easily comprehensible and interpretable. A 10- fold cross-validation method has been employed during the training session. 

8 



Bidirectional Encoder Representations from Transformers (BERT) is a highly sought-after new language representation model designed to pre-train deep bidirectional representations from the unlabeled text by joint conditioning on both left and right contexts in all layers [36]. In this experiment, Multilingual BERT (mBERT) is used and it is flexible in providing sentence representations for 104 languages [37]. It is recommended that no data preprocessing is required for modeling in BERT. 

## **3.5 Performance Parameters** 

Four effective measures that have been used in this study are based on confusion matrix output, which are Sensitivity, Specificity, Balanced Accuracy, and Training time. 

- Sensitivity or Recall (True Positive Rate) = TP / (TP + FN) 

- Positive Predictive Value or Precision = TP / (TP + FP) 

- F-Measure = 2 _·_<sup>Precision</sup><sup>_·_Recall</sup> 

   - Precision + Recall 

- Training Time = Amount of time required to train the Learning Algorithm 

The usefulness of these metrics is ubiquitous in text classification for comparative analysis among numerous learning algorithms. The F-measure serves as a middle ground between recall and precision, representing a balance between the two. Its value is significant only when both recall and precision are at high levels. When _α_ (a parameter) equals 0, the F-measure is equivalent to recall, while _α_ = 1 makes it equivalent to precision. The F-measure ranges from 0 to 1, with 0 indicating that no relevant documents were retrieved and 1 indicating that all retrieved documents are relevant and all relevant documents were retrieved. 

## **3.6 Interpretability** 

For the interpretability of the models, Decision trees have been used primarily that can provide valuable insights into how the model classifies texts as spam or non-spam. The decision tree for both the SPAM data and the SMS Phishing data has been analyzed carefully to understand how it separates spam from non-spam text. Each internal node in the tree represents a decision based on a particular feature, and each leaf node represents a final decision (spam or non-spam). Attention has been paid to the split points and the features used for splitting, as they indicate the most important attributes for the classification. Features higher up in the tree that appears closer to the root node have been identified as the most important features, as they are the words or word stem that strongly influence the spam classification. 

## **3.7 Execution Environments** 

All the modeling with classical learning algorithms and analyses were carried out in R (v. 4.0.3) on an old computer equipped with a Core 2 Duo processor (Operating at 3.00 GHz base) and 8 GB of RAM. The modeling with Transformers (mBERT) was executed in a computer with an Intel Core i5-7500 processor (Base clock 3.40 GHz), 32 GB of RAM, and NVIDIA GeForce GTX 1050Ti graphics card with 4 GB DDR5 RAM and 768 CUDA cores. Text preprocessing, Decision Trees, SVMs, IRLBA, 

9 



and Transformers were respectively implemented using the `quanteda` , `caret` , `e1071` , `irlba` , `TensorFlow` , `reticulate` , and `keras` packages in R. 

# **4 Results and Discussions** 

For each trained model, a couple of comparisons were made in terms of model accuracy and training time. Each dataset was split into two parts, one for training and the other for testing: SPAM dataset with a ratio of 70:30: 70% for training and 30% testing. The SMS Phishing dataset was also partitioned with a ratio of 70:30. Table I, 2, 3, and 4 summarizes the performance metrics for the decision tree and Support Vector Machine Model for the SPAM and SMS Phishing dataset respectively. Table 5 accumulates the results after training both of the SPAM and SMS Phishing data with the Transformers model (mBERT). 

In the first phase for the SPAM dataset, all the raw text data went through the preprocessing stages followed by feature extraction through traditional tf-idf and modified tf-idf (Ctf-idf) respectively. Decision tree models were then built upon the tf-idf and Ctf-idf transformed data for classification. From the table I, it can be seen that there are no discernible changes in the performance metrics for both cases in the decision tree models. The time required to train each of the models was 13 minutes and 12 minutes respectively. 

In the second phase for the SPAM dataset, the tf-idf and Ctf-idf feature extracted models were transformed by IRLBA. 

**Table 1** Spam data 10-fold cross-validation performance metrics for Decision Tree model. 

|Models|Precision|Recall|F1-score|Training time|
|---|---|---|---|---|
|TF-IDF|0.9627|0.9687|0.9657|13 min|
|CTF-IDF|0.9627|0.9687|0.9657|12 min|
|TF-IDF (IRLBA)|0.9869|0.9420|0.9640|17 sec|
|CTF-IDF (IRLBA)|0.9889|0.9508|0.9700|16 sec|



After the transformation, it required only 17 seconds to train a decision tree model on tf-idf and 16 seconds on Ctf-idf transformed corpus (Table 1). The computational time was significantly reduced by the application of IRLBA. Table 1 also shows the combined effect of Ctf-idf and IRLBA transformation helps to increase the F1-score from 96.57% to 97% in contrast with the tf-idf and IRLBA transformed data where it decreases slightly 

All the procedures were the same for Support Vector Machine as well. It can be seen from table 2 that F1-Scores are hovering over 98% for preliminary tf-idf and Ctf-idf transformed data with 35 and 37 seconds of training time for each model respectively. After the projection of tf-idf and Ctf-idf transformed data in the semantic space through IRLBA, the overall performance of the models increased. It can be seen from Table 2 that the F-Scores improved in both cases but the Ctf-idf-IRLBA 

10 



**Table 2** Spam data performance metrics for SVM model. 

|Models|Precision|Recall|F1-score|Training time|
|---|---|---|---|---|
|TF-IDF|0.9807|0.9827|0.9817|35 sec|
|CTF-IDF|0.9793|0.9832|0.9812|37 sec|
|TF-IDF (IRLBA)|0.9750|0.9912|0.9830|5 sec|
|CTF-IDF (IRLBA)|0.9783|0.9965|0.9873|5 sec|



transformed data performed better with an F-Score of 98.73%. Not only that, the computational time was also vastly reduced to 5 seconds only for training each of the models in SVM. 

For SPAM dataset, the Transformers (mBERT) models raised the training and the validation accuracy slightly above 99% but with the expense of a huge computational power (Table 5). It required NVIDIA graphics card to run all the CUDA cores simultaneously for up to 1:30 hours on the minimum level to train the model. 

**Table 3** SMS phishing data 10-fold cross-validation performance metrics for Decision Tree model. 

|Models|Precision|Recall|F1-score|Training time|
|---|---|---|---|---|
|TF-IDF|0.9767|0.9184|0.9467|10.33 min|
|CTF-IDF|0.9767|0.9184|0.9467|9.67 min|
|TF-IDF (IRLBA)|0.9413|0.9601|0.9506|22.09 sec|
|CTF-IDF (IRLBA)|0.9520|0.9683|0.9601|19.46 sec|



The robustness of the methodology is tested on the SMS Phishing dataset. Table 3 shows, without the application of IRLBA, the Ctf-idf performs no better than the tf-idf transformed data with an F1-score of 94.67% in decision tree models. But, the training time for the Ctf-idf model was slightly better than the tf-idf model taking 9.67 minutes. After the IRLBA transformation of the tf-idf and Ctf-idf models, the F-1 scores increased above 95% for both models but again Ctf-idf-IRLBA performed better. Table 3 also shows the tremendous reduction in training time requiring approximately 20 seconds which took almost 10 minutes previously. 

**Table 4** SMS phishing data performance metrics for SVM model. 

|Models|Precision|Recall|F1-score|Training time|
|---|---|---|---|---|
|TF-IDF|0.9832|0.9642|0.9736|14.31 sec|
|CTF-IDF|0.9854|0.9738|0.9796|12.90 sec|
|TF-IDF (IRLBA)|0.9703|0.9913|0.9807|9.82 sec|
|CTF-IDF (IRLBA)|0.9737|0.9945|0.9840|9.80 sec|



11 



For Support Vector Machine, Table 4 shows the methodology exhibits the same characteristic reduction in training time but the difference is not that significant. Again, the Ctf-idf transformed model worked better on SMS Phishing data than the tf-idf model before transforming the models through IRLBA. After the application of IRLBA it can be seen from Table 4 that the F1-Scores again improved up to 98% for both of the models and still Ctf-idf-IRLBA maintained better performance. 

**Table 5** Transformer (mBERT) Performance Metrics for SPAM and SMS Phishing Data. 

|Data|No.<br>of Epoch|Training<br>Accuracy|Validation<br>Accuracy|Training<br>Time|
|---|---|---|---|---|
|SPAM Data|7|0.9913|0.9904|1:30 hrs|
|SMS Phishing<br>Data|15|0.9976|0.9888|1:43 hrs|



In the case of the Transformers (mBERT) model, table 5 shows the model performs a little better than the SVM with a validation accuracy of 98.88% for the SMS Phishing dataset. As expected the transformer model took a considerable amount of training time over 1 hour and 35 minutes with the help of a NVIDIA graphics card at the backend with all the CUDA cores running simultaneously. 

For interpretability of the spam texts in the SPAM dataset, the following stemmed words have been found influential according to the importance of making decisions on the decision tree model: call, txt, prize, guarantee, claim, won, urgent, tone, service. The important features or stemmed words for the interpretability of the spam texts in the SMS Phishing dataset are as follows: opinion, jada, spl, stylish, online, pls, want, silent, simple, character, loveable, reply, matur, come, free, can, talk, sometimes, working, overtime, sir, sent, email, log, payment, portal, send, tax, pay, citizen. The presence of these words or the word stem is suggestive of the texts being considered spam. 

It is quite evident from the experiment that Ctf-idf transformation of the dataset with the combination of IRLBA algorithm for dimensionality reduction is significantly faster in training any classical machine learning model and at the same time it improves the performance of the model after shrinking and transforming the feature space from thousands of columns to a handful of informative columns. The transformers (BERT) models are great in producing state-of-the-art accuracy but with a huge cost of computational power and producing a higher carbon footprint. 

# **5 Conclusions and Future Work** 

This study presents a comprehensive approach to tackle the computational and environmental challenges associated with text analytics, specifically in the context of Spam detection. It introduces the innovative CTF-IDF method and IRLBA for dimensionality reduction, significantly reducing the need for computational resources and improving model accuracy at the same time. 

12 



The utilization of CTF-IDF, a novel approach to transforming tokenized text data, addressed the issue of severe penalties imparted by traditional TF-IDF method, often resulting in the loss of the effects of various keywords. Our results illustrated that the deployment of CTF-IDF enhances model accuracy in text classification tasks. The incorporation of the IRLBA algorithm further contributed to the efficiency and speed of the proposed methodology. 

Through our experimentation with both the SPAM dataset and the SMS Phishing dataset, the models exhibited notable improvement in computational time and reduced requirements for computational resources. Moreover, they showed competitive accuracy rates when compared with sophisticated deep learning models like BERT, thus providing a viable alternative. Looking forward, our research will focus on enhancing the proposed methodology. For CTF-IDF, we plan to develop an adaptive model that can adjust its penalty metrics based on the specifics of the text corpus. As for the IRLBA algorithm, we aim to optimize its performance further for both sparse and dense matrix types, and potentially explore its application in other areas of data analytics. 

Moreover, we will investigate the generalizability of our proposed methodology in different domains and diverse datasets. It would be intriguing to examine the efficiency of our approach in other text analytics tasks like sentiment analysis, topic modeling, or document clustering. 

Finally, the intersection of machine learning and environmental sustainability remains a compelling field for future exploration. The current study focuses on reducing the computational resources in text analytics, but the concept can be extended to other machine learning tasks. The quest to make AI ”greener” should continue with more research on energy-efficient algorithms and models, not only for the sake of our environment but also to ensure the sustainable development of machine learning technologies. The research community must strive for a balance between algorithmic innovation and computational responsibility. In this context, our work serves as an important step towards shaping the future of responsible AI. 

# **Acknowledgment** 

The authors received no external funding for this research. 

# **References** 

- [1] McLaughlin, J.E., Lyons, K., Lupton-Smith, C., Fuller, K.: An introduction to text analytics for educators. Currents in Pharmacy Teaching and Learning **14** (10), 1319–1325 (2022) https://doi.org/10.1016/j.cptl.2022.09.005 

- [2] Beel, J., Gipp, B., Langer, S., Breitinger, C.: Research-paper recommender systems: a literature survey. International Journal on Digital Libraries **17** (4), 305–338 (2016) https://doi.org/10.1007/s00799-015-0156-0 

- [3] Rajaraman, A., Ullman, J.D.: Data mining. In: Mining of Massive Datasets. Cambridge University Press, Cambridge (2012). 

13 



https://www.cambridge.org/core/books/abs/mining-of-massive-datasets/datamining/E5BFF4C1DD5A1FB946D616D619B373C2 

- [4] Cheng, L., Yang, Y., Zhao, K., Gao, Z.: Research and improvement of tf-idf algorithm based on information theory. In: Intelligent Computing Theories and Application. Springer, Cham (2018). https://doi.org/10.1007/978-3-030-14680-1 67 

- [5] Maaten, L., Postma, E., Herik, J.: Dimensionality reduction: A comparative review. Journal of Machine Learning Research **10** , 66–71 (2009) 

- [6] Baglama, J., Reichel, L.: Augmented implicitly restarted lanczos bidiagonalization methods. SIAM Journal on Scientific Computing **27** (1), 19–42 (2005) 

- [7] Almeida, T.A., Hidalgo, J.M.G., Yamakami, A.: Contributions to the study of sms spam filtering: New collection and results. In: Proceedings of the 11th ACM Symposium on Document Engineering, pp. 259–262 (2011). https://doi.org/10. 1145/2034691.2034742 

- [8] Mishra, S., Soni, D.: SMS phishing dataset for machine learning and pattern recognition. https://doi.org/10.17632/f45bkkt8pr.1. Dataset, accessed 23 May 2023 (2022) 

- [9] Cao, J., Liu, S., Zhao, P., Zhu, H.: Rp-net: A pointnet++ 3d face recognition algorithm integrating rops local descriptor. IEEE Access **10** , 91245–91252 (2022) https://doi.org/10.1109/ACCESS.2022.3202216 

- [10] Ahuja, R., Chug, A., Kohli, S., Gupta, S., Ahuja, P.: The impact of features extraction on the sentiment analysis. Procedia Computer Science **152** , 341–348 (2019) 

- [11] Erra, U., Senatore, S., Minnella, F., Caggianese, G.: Approximate tf–idf based on topic extraction from massive message stream using the gpu. Information Sciences **292** , 143–161 (2015) https://doi.org/10.1016/j.ins.2014.08.062 

- [12] Dhar, A., Dash, N.S., Roy, K.: Application of tf-idf feature for categorizing documents of online bangla web text corpus. In: Intelligent Engineering Informatics. Advances in Intelligent Systems and Computing, vol. 695, pp. 51–59. Springer, Cham (2018). https://doi.org/10.1007/978-981-10-7566-7 ~~6~~ 

- [13] De Boom, C., Van Canneyt, S., Demeester, T., Dhoedt, B.: Representation learning for very short texts using weighted word embedding aggregation. Pattern Recognition Letters **80** , 150–156 (2016) 

- [14] Liu, C.-Z., Sheng, Y.-X., Wei, Z.-Q., Yang, Y.-Q.: Research of text classification based on improved tf-idf algorithm. In: 2018 IEEE International Conference of Intelligent Robotic and Control Engineering (IRCE), pp. 218–222 (2018). https: 

14 



//doi.org/10.1109/IRCE.2018.8492945 

- [15] Zu, G., Ohyama, W., Wakabayashi, T., Kimura, F.: Accuracy improvement of automatic text classification based on feature transformation. In: ACM Symposium on Document Engineering, pp. 118–120 (2003) 

- [16] Han, X., Zu, G., Ohyama, W., Wakabayashi, T., Kimura, F.: Accuracy improvement of automatic text classification based on feature transformation and multiclassifier combination. In: Advances in Multilingual and Multimodal Information Retrieval, pp. 463–468 (2004). https://doi.org/10.1007/978-3-540-30483-8 ~~5~~ 7 

- [17] Zareapoor, M.: Information engineering and electronic business. Information Engineering and Electronic Business **2** , 60–65 (2015) https://doi.org/10.5815/ ijieeb.2015.02.08 

- [18] Blei, D.M., Ng, A.Y., Jordan, M.I.: Latent dirichlet allocation. Journal of Machine Learning Research **3** , 993–1022 (2003) 

- [19] Blei, D.M., Ng, A.Y., Jordan, M.I.: Latent dirichlet allocation. In: Advances in Neural Information Processing Systems, vol. 14. MIT Press, Cambridge, MA (2001) 

- [20] Jelodar, H., Wang, Y., Yuan, C., _et al._ : Latent dirichlet allocation (lda) and topic modeling: Models, applications, a survey. Multimedia Tools and Applications **78** (11), 15169–15211 (2019) https://doi.org/10.1007/s11042-018-6894-4 

- [21] Unknown: Non-negative matrix factorization, a new tool for feature extraction: Theory and applications. Accessed May 23, 2023 (2023). https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi= f488014381ac79b2c4dd8921abb734b117218c7a 

- [22] Lee, D.D., Seung, H.S.: Learning the parts of objects by nonnegative matrix factorization. Nature **401** (6755), 788–791 (1999) https://doi.org/10.1038/44565 

- [23] Wang, L., Wan, Y.: Sentiment classification of documents based on latent semantic analysis. In: Advanced Data Mining and Applications, pp. 356–361 (2011). https://doi.org/10.1007/978-3-642-21802-6 ~~5~~ 7 

- [24] Liu, S., Wang, Y., Liu, M., Zhang, X., Zhu, J., Guo, B.: Visual exploration of semantic relationships in neural word embeddings. IEEE Transactions on Visualization and Computer Graphics **24** (1), 553–562 (2018) https://doi.org/10.1109/ TVCG.2017.2744478 

- [25] Bamler, R., Mandt, S.: Dynamic word embeddings. In: Proceedings of the 34th International Conference on Machine Learning, pp. 380–389 (2017) 

- [26] Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning 

15 



Research **9** , 2579–2605 (2008) 

- [27] Ma, L., Zhang, Y.: Using word2vec to process big text data. In: 2015 IEEE International Conference on Big Data, pp. 2895–2897 (2015). https://doi.org/10.1109/ BigData.2015.7364114 

- [28] Mikolov, T., et al.: Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 (2013) 

- [29] Joulin, A., et al.: Fasttext.zip: Compressing text classification models. arXiv preprint arXiv:1612.03651 (2016) 

- [30] Santos, I., Nedjah, N., Mourelle, L.: Sentiment analysis using convolutional neural network with fasttext embeddings. In: 2017 IEEE Latin American Conference on Computational Intelligence (LA-CCI), pp. 1–5 (2017). https://doi.org/10.1109/ LACCI.2017.8285683 

- [31] Pennington, J., Socher, R., Manning, C.D.: Glove: Global vectors for word representation. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543 (2014). https://aclanthology.org/D14-1162.pdf 

- [32] Porter, M.F.: An algorithm for suffix stripping. In: Readings in Information Retrieval, pp. 313–316. Morgan Kaufmann, San Francisco, CA (1997) 

- [33] Package ‘irlba’ Type Package Title Fast Truncated Singular Value Decomposition and Principal Components Analysis (2022). https://cran.r-project.org/web/ packages/irlba/irlba.pdf 

- [34] Cortes, C., Vapnik, V.: Support-vector networks. Machine Learning **20** (3), 273– 297 (1995) 

- [35] Joachims, T.: Text categorization with support vector machines: Learning with many relevant features. In: N´edellec, C., Rouveirol, C. (eds.) Machine Learning: ECML-98. Lecture Notes in Computer Science, vol. 1398, pp. 137–142. Springer, Berlin, Heidelberg (1998). https://doi.org/10.1007/BFb0026683 

- [36] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018) 

- [37] Libovick´y, J., Rosa, R., Fraser, A.: How language-neutral is multilingual bert? arXiv preprint arXiv:1911.03310 (2019) 

16 

