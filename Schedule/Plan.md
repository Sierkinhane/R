##
1. [Weakly supervised learning](#Weakly-supervised-learning)
2. [Semi-supervised learning](#Semi-supervised-learning)
3. [Self-supervised learning](#Self-supervised-learning)
4. [Unsupervised learning](#Unsupervised-learning)
5. [Transfer learning ](#Transfer-learning )
6. [Graph convolutional neural network](#Graph-convolutional-neural-network)
7. [Transformer](#Transformer)
8. [Image-Recognition](#Image-Recognition)
9. [Others](#Others)
### Weakly supervised learning
  - Weakly supervised learning <image-level>
    - Localization
    - Detection
      - [ ] [Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection](https://arxiv.org/pdf/2010.12023.pdf) (NeurIPS 2020)
    - Segmentation
      - [ ] [Revisiting Dilated Convolution: A Simple Approach for Weakly- and SemiSupervised Semantic Segmentation](https://arxiv.org/pdf/1805.04574.pdf) (CVPR 2018)
      - [X] [Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation](https://arxiv.org/pdf/2007.01947.pdf) (ECCV 2020)
      - [X] [Find it if You Can: End-to-End Adversarial Erasing for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2011.04626)
      - [ ] [UNIVERSAL WEAKLY SUPERVISED SEGMENTATION BY PIXEL-TO-SEGMENT CONTRASTIVE LEARNING](https://arxiv.org/pdf/2105.00957v2.pdf) (ICLR 2021)
    - Instance segmentation
      - [X] [Weakly Supervised Instance Segmentation using Class Peak Response](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Weakly_Supervised_Instance_CVPR_2018_paper.pdf) (CVPR 2018)
      - [X] [Leveraging Instance-, Image- and Dataset-Level Information for Weakly Supervised Instance Segmentation](https://ieeexplore.ieee.org/abstract/document/9193980) (TPAMI 2020)
      - [ ] [Towards Partial Supervision for Generic Object Counting in Natural Scenes](https://arxiv.org/pdf/1912.06448.pdf) (TPAMI 2020)

### Semi-supervised learning
  - Semi-supervised learning
    - [ ] [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/abs/1908.02983)
    - [ ] [Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2003.03773)
    - [ ] [Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks)
    - [ ] [In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning](https://arxiv.org/abs/2101.06329v1)

### Self-supervised learning
  - Self-supervised learning

### Unsupervised learning
  - Unsupervised learning
    - Contrastive representation learning
      - [X] [Introduction blog](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)
      - [ ] [Noise-contrastive estimation: A new estimation principle forunnormalized statistical models](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
      - [ ] [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670) (ICLR 2019)
      - [ ] [Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)
      - [ ] [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
      - [ ] [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf) (CVPR 2018)
      - [ ] [Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/abs/1906.00910)
      - [ ] [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/pdf/1905.09272.pdf) (ICML 2020)
      - [ ] [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) (NeurIPS 2020)
      - [ ] [Can Semantic Labels Assist Self-Supervised Visual Representation Learning?](https://arxiv.org/pdf/2011.08621.pdf)
      - [X] (SimCLR) [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (PMLR 2020)
      - [X] (MoCo) [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722v3)
      - [ ] [MoCov2] [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/pdf/2003.04297.pdf)
      - [ ] (SimCLRv2) [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://proceedings.neurips.cc/paper/2020/file/fcbc95ccdd551da181207c0c1400c655-Paper.pdf) (NeurIPS 2020)
      - [ ] (SwAV) [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/pdf/2006.09882.pdf)
      - [ ] (BYOL) [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf)
      - [ ] (SimSiam) [Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf)
      - Re-ID
        - [ ] [A Bottom-up Clustering Approach to Unsupervised Person Re-identification](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) (AAAI 2019)
        - [ ] [Unsupervised Person Re-identification: Clustering and Fine-tuning](https://arxiv.org/pdf/1705.10444.pdf)
        - [ ] [Prototypical Contrastive Learning of Unsupervised Representations](https://arxiv.org/pdf/2101.11939.pdf)
        - [ ] [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://proceedings.neurips.cc/paper/2020/file/821fa74b50ba3f7cba1e6c53e8fa6845-Paper.pdf) (NeurIPS 2020)
    - Clustering
        - [ ] [SCAN: Learning to Classify Images without Labels ](https://arxiv.org/pdf/2005.12320.pdf) (ECCV 2020)
        - [ ] [Contrastive Clustering](https://arxiv.org/pdf/2009.09687.pdf) (AAAI 2020)
        - [ ] [Improving Unsupervised Image Clustering With Robust Learning](https://arxiv.org/pdf/2012.11150v2.pdf) (CVPR 2021)
        - [ ] [SPICE: Semantic Pseudo-labeling for Image Clustering](https://arxiv.org/pdf/2103.09382v1.pdf) (Arxiv preprint)
    - Segmentation
        - [ ] [Autoregressive Unsupervised Image Segmentation](https://arxiv.org/pdf/2007.08247.pdf) (ECCV 2020)
        - [ ] [Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals](https://arxiv.org/pdf/2102.06191.pdf) (Arxiv preprint)

### Transfer learning 
  - Transfer learning 
    - Domain adaptation
      - [ ] [Hung-yi Lee](https://drive.google.com/file/d/15wlfUtTmnb4cEAHZtNJ9_jJE26nSNhAX/view)
      - [ ] [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)
      - [ ] [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/pdf/1502.02791.pdf)
      - [ ] [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf)
      - [ ] [Simultaneous Deep Transfer Across Domains and Tasks](https://arxiv.org/abs/1510.02192)
      - [ ] [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
      - [ ] [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424)
      - [ ] [Domain Separation Networks](https://arxiv.org/abs/1608.06019)

### Graph convolutional neural network
  - Graph convolutional neural network
    - Pose estimation
    - Image recognition

### Transformer
  - Transformer
    - [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762v5)
    - [ ] [DeLighT: Deep and Light-weight Transformer](https://arxiv.org/abs/2008.00623) (ICLR 2021)
    - [ ] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
    - Detection
      - [ ] [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) ([code](https://github.com/facebookresearch/detr))
      - [ ] [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) ([code](https://github.com/fundamentalvision/Deformable-DETR))
    - GAN
      - [ ] [TransGAN: Two Transformers Can Make One Strong GAN](https://arxiv.org/abs/2102.07074) ([code](https://github.com/VITA-Group/TransGAN))
      - [ ] [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf) (NeurIPS 2019)
      - [ ] [Unsupervised Image Transformation Learning via Generative Adversarial Networks](https://arxiv.org/abs/2103.07751)
      - [ ] [Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](https://arxiv.org/pdf/2002.03754.pdf)
    - Person Re-ID
      - [ ] [TransReID: Transformer-based Object Re-Identification](https://arxiv.org/pdf/2102.04378.pdf)

### Image-Recognition
  - Multi-label Classification
    - [X] [Cross-Modality Attention with Semantic Graph Embedding for Multi-Label Classification](https://arxiv.org/abs/1912.07872) (AAAI 2020)
    - [ ] [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/pdf/1904.03582.pdf) (CVPR2019)
  - Long-tailed Classification
    - [X] [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/pdf/1910.09217.pdf) (ICLR 2020)
    - [X] [BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition](https://arxiv.org/abs/1912.02413) (CVPR 2020)
    - [ ] [Dynamic Curriculum Learning for Imbalanced Data Classification](https://arxiv.org/abs/1901.06783) (ICCV 2019)
    - [ ] [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) (CVPR 2019)
    - [ ] [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413) (NeurIPS 2019)
    - [ ] [Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective](https://arxiv.org/abs/2003.10780) (CVPR 2020)
    - [ ] [Remix: Rebalanced Mixup](https://arxiv.org/abs/2007.03943) (Arxiv Preprint 2020)
    - [ ] [Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective](https://arxiv.org/abs/2002.10826) (CVPR 2020)
    - [ ] [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://arxiv.org/abs/2001.01536) (ECCV 2020)
    - [ ] [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160) (CVPR 2019)
    - [ ] [Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/abs/2010.01809) (ICLR 2021)
### Others
  - Others
  - [ ] [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
    - Person Re-ID
      - [X] [Intra-Camera Supervised Person Re-Identification](https://arxiv.org/abs/2002.05046) (IJCV)
      - [ ] [A Graph-Based Approach for Making Consensus-Based Decisions in Image Search and Person Re-Identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8852741) (TPAMI)
    - Face recognition
    - Fine-grained recognition
    - Segmantic segmentation
      - [X] [Exploring Cross-Image Pixel Contrast for Semantic Segmentation](https://arxiv.org/abs/2101.11939)
      - [ ] [Region Mutual Information Loss for Semantic Segmentation](https://arxiv.org/pdf/1910.12037.pdf) (NeurIPS 2019)
      - [ ] [Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers](https://arxiv.org/pdf/2103.12340.pdf) (CVPR 2021)
      - [ ] [Commonality-Parsing Network across Shape and Appearance for Partially Supervised Instance Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530375.pdf) (ECCV 2020)
    - Convolutions
      - [ ] [Fully Learnable Group Convolution for Acceleration of Deep Neural Networks](https://arxiv.org/pdf/1904.00346.pdf)
    - Image Recognition 
      - [ ] [Knowledge-Guided Multi-Label Few-Shot Learning for General Image Recognition](https://arxiv.org/pdf/2009.09450.pdf) (TPAMI)
      - [X] [Fixing Localization Errors to Improve Image Classification](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700273.pdf) (ECCV 2020)
      - [ ] [Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels](https://arxiv.org/pdf/2101.05022.pdf) (CVPR 2021)
      - [ ] [Multiple Instance Learning Convolutional Neural Networks for Object Recognition](https://arxiv.org/pdf/1610.03155.pdf) (Multiple Instance Learning)
    - Geoffrey Hinton
      - [ ] [How to represent part-whole hierarchies in a neural network](https://arxiv.org/pdf/2102.12627.pdf)
    - Neural architecture search
      - [ ] [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)
    - DNN
      - [ ] [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/pdf/1611.10080.pdf)
    - Image Retrival
      - [ ] [Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163) (ECCV 2020)



## 近期计划（2019-6-15 -> 2019-7~）  
- Linear Algebra  
  - ~~Hung-yi lee [Linear Algebra (2018,Fall)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_LA18.html)~~
- 概率与统计
  - 叶丙成 [几率与统计](https://www.youtube.com/watch?v=GwSEguqJj6U)
- ~~论文~~
  - ~~完成ResNet-ASPP-Net网络的训练以及论文的编写，尽量七月前投稿~~
##
## （2019-10-13 -> 2019-12-）  
- mit18.06 线性代数课程视频+笔记（补充第三部分内容）
- 开始阅读医疗影像分析综述与相关论文，并积累专业词汇
- 叶丙成 [几率与统计](https://www.youtube.com/watch?v=GwSEguqJj6U)
- C/C++复习与学习
- 统计学习方法
##
##  确定研究方向：(2020年3月19日)
  - ~~2D/3D pose estimation/face alignment, GAN, deep reinforecement learning~~
  - 多领域涉猎多领域结合于单一领域
## 
##  2020-3-19
  - C++ 侯捷
  - Digital Image Processing
  
## 2020-11-19 
  - 随机过程 1, 2, 4章节复习
  - 形式语言与自动机理论复习
  - WSOL的消融实验
  - 阅读弱监督学习的论文
    - WOSL
    - WOSD
    
