# SGDD: Does Graph Distillation See Like Vision Dataset Counterpart?

Official implementation of "[Does Graph Distillation See Like Vision Dataset Counterpart](http://arxiv.org/abs/2310.09192)", published as a conference paper at NeurIPS 2023.

The authors of this paper are: Beining Yang*, Kai Wang*, Qingyun Sun, Cheng Ji, Xingcheng Fu, Hao Tang, Yang You, Jianxin Li
![Does Graph Distillation See Like Vision Dataset Counterpart?](./images/yang2023does.png)

# Abstract
Training on large-scale graphs has achieved remarkable results in graph representation learning, but its cost and storage have attracted increasing concerns. Existing graph condensation methods primarily focus on optimizing the feature matrices of condensed graphs while overlooking the impact of the structure information from the original graphs. To investigate the impact of the structure information, we conduct analysis from the spectral domain and empirically identify substantial Laplacian Energy Distribution (LED) shifts in previous works. Such shifts lead to poor performance in cross-architecture generalization and specific tasks, including anomaly detection and link prediction. In this paper, we propose a novel Structure-broadcasting Graph Dataset Distillation (\textbf{SGDD}) scheme for broadcasting the original structure information to the generation of the synthetic one, which explicitly prevents overlooking the original structure information. 
Theoretically, the synthetic graphs by SGDD are expected to have smaller LED shifts than previous works, leading to superior performance in both cross-architecture settings and specific tasks.
We validate the proposed SGDD~across 9 datasets and achieve state-of-the-art results on all of them: for example, on YelpChi dataset, our approach maintains 98.6\% test accuracy of training on the original graph dataset with 1,000 times saving on the scale of the graph. Moreover, we empirically evaluate there exist 17.6\% $\sim$ 31.4\% reductions in LED shift crossing 9 datasets. Extensive experiments and analysis verify the effectiveness and necessity of the proposed designs.

# OS Requirements
* Linux OS
* Python 3.7

# Requirements
```code
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
torchvision==0.8.0
configs==3.0.3
deeprobust==0.2.4
scikit_learn==1.0.2
```

# Download Datasets
Cora, Citeseer: [Pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
Reddit, Ogbn-arxiv, Flick: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) [GCond](https://github.com/ChandlerBang/GCond)
YelpChi: [DGL](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudYelpDataset.html#dgl.data.FraudYelpDataset)
Amazon: [DGL](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudAmazonDataset.html#dgl.data.FraudAmazonDataset)
DBLP, Citeseer: [Pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html#torch_geometric.datasets.DBLP)


# Getting started
* Clone this repo
```
git clone ...
cd SGDD/
```
* Install the required packages
```
pip install -r ./requirements.txt
```
* Dwonload the datasets from the above links and put them in the `./data` folder

* Train the model (setting dataset to your dataset name)
```
python train_SGDD.py --dataset ${dataset}  --nlayers=2 -beta 0.1 --r=0.5 --gpu_id=0
```



