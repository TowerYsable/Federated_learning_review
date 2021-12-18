### 1 FedGen

问题：User heterogeneity has imposed significant challenges to FL, which can incur drifted global models that are slow to converge.   

解决方法：

- Knowledge Distillation: by refining the server model using aggregated knowledge from heterogeneous users, other than directly aggregating their model parameters. 
  - 聚合知识而不是聚合模型参数
  - 缺点：depends on a proxy dataset
    - making it impractical unless such prerequisite is satisfied
    - Moreover, the ensemble knowledge is not fully utilized to guide local model learning, which may in turn affect the quality of the aggregated model  
- data-free knowledge distillation
  - server: the server learns a lightweight generator to ensemble user information in a data-free manner, which is then broadcasted to users, regulating local training using the learned knowledge as an inductive bias. 

相关工作：

- FedAvg

挑战：

- data heterogeneity：Along with its promising prospect, FL faces practical challenges from data heterogeneity (Li et al., 2020b), in that user data from real-world is usually non-iid distributed, which inherently induces deflected local optimum (Karimireddy et al., 2020). 
- model heterogeneity: Moreover, the permutation-invariant property of deep neural networks has further increased the heterogeneity among user models (Yurochkin et al., 2019; Wang et al., 2020b).  

> 参数化变量