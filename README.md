# 1 basic

## 1.1 FL libraries   

### 1.1.1 FedML [link](https://fedml.ai)

![image-20211118131447721](README.assets/image-20211118131447721.png)

**contribution:**

- FedML supports three computing paradigms: on-device training for edge devices, distributed computing, and single-machine simulation.   
- FedML also promotes diverse algorithmic research with flexible and generic API design and comprehensive reference baseline implementations (optimizer, models, and datasets).   
  - Support of diverse FL computing paradigms.   
  - Support of diverse FL configurations.   
  - Standardized FL algorithm implementations.  
  - Standardized FL benchmarks.   
  - Fully open and evolving.  

**Introduction:**

- FL differs from data center-based distributed training in three major aspects: 
  - 1) statistical heterogeneity
    - Adaptive Federated Optimizer [2], FedNova [3], FedProx [4], and FedMA [5]   
  - 2) system constraints
    - sparsification and quantization techniques to reduce the communication overheads and computation costs during the training process [6, 7, 8, 9, 10, 11, 12].  
  - 3) trustworthiness.   
    - focuses on developing new defense techniques for adversarial attacks to make FL robust [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 22], and proposing methods such as differential privacy (DP) and secure multiparty computation (SMPC) to protect privacy [25, 26, 27, 28, 29, 30, 31, 32, 33].  
- Existing efforts are confronted with a number of limitations that we argue are critical to FL research:
  - Lack of support of diverse FL computing paradigms.  
    - simulation-oriented FL libraries: Tensorflow-Federated(TFF),PySyft,LEAF
      - with simulation in a single machine, making them unsuitable for FL algorithms which require the exchange of complex auxiliary information and customized training procedure.  
    - Production-oriented  FL libraries : FATE,PaddleFL
      - they are not designed as flexible frameworks that aim to support algorithmic innovation for open FL problems  
  - Lack of support of diverse FL configurations  
    - In terms of network topology  ,vertical FL [44, 45, 46, 47, 48, 49, 50], split learning [51, 52], decentralized FL [53, 54, 55, 56],  hierarchical FL [57, 58, 59, 60, 61, 62], and meta FL [63, 64, 65]   
    - In terms of exchanged information, besides exchanging gradients and models, recent FL algorithms  propose to exchange information such as pseudo labels in semi-supervised FL [66] and architecture parameters in neural architecture search-based FL  
    - In terms of training procedures, the training procedures in federated GAN [70, 71] and transfer learning-based FL [72, 73, 74, 75, 76]  
  - Lack of standardized FL algorithm implementations and benchmarks.   

