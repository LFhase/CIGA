
<h1 align="center">Causality Inspired Invariant Graph LeArning (CIGA)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2202.05441"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href="https://github.com/LFhase/CIGA"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <!-- <a href="https://colab.research.google.com/drive/1t0_4BxEJ0XncyYvn_VyEQhxwNMvtSUNx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <a href="https://openreview.net/forum?id=A6AFK_JwrIW"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2722&color=blue"> </a>
    <!-- <a href="https://github.com/Graph-COM/GSAT/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/Graph-Com/GSAT?color=blue"> </a>
    <a href="https://icml.cc/virtual/2022/spotlight/17430"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a>
    <a href="https://icml.cc/media/icml-2022/Slides/17430.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
    <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>


This repo contains the sample code for reproducing the results of NeurIPS 2022 paper: *[Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs](https://arxiv.org/abs/2202.05441)* 😆😆😆.

TODO items:
- [ ] Full code and instructions will be released soon!
- [ ] Benchmarking CIGA on [GOOD](https://github.com/divelab/GOOD) benchamrk, which is recently accepted by NeurIPS 2022 Datasets and Benchmarks Track!

## Introduction
Despite recent success in using the invariance principle for out-of-distribution (OOD) generalization on Euclidean data (e.g., images), studies on graph data are still limited. Different from images, the complex nature of graphs poses unique challenges to adopting the invariance principle:

1. Distribution shiftson graphs can appear in **<ins>a variety of forms</ins>**:
    - Node attributes;
    - Graph structure;
    - Both;

2. Each distribution shift can spuriously correlate with the label in **<ins>different modes</ins>**. We divide the modes into FIIF and PIIF, according to whether the latent causal feature $C$ fully determines the label $Y$, i.e., or $(S,E)\perp\!\!\!\!\perp Y|C$:
    - Fully Informative Invariant Features (FIIF): $Y\leftarrow C\rightarrow S\leftarrow E$;
    - Partially Informative Invariant Features (PIIF): $C\rightarrow Y\leftarrow S \leftarrow E$;
    - Mixed Informative Invariant Features (MIIF): mixed with both FIIF and PIIF;

3. **<ins>Domain or environment partitions</ins>**, which are often required by OOD methods on Euclidean data, can be highly expensive to obtain for graphs.


<p align="center"><img src="./data/arch.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em> The architecture of CIGA.</p>

This work addresses the above challenges by generalizing the causal invariance principle to graphs, and instantiating it as CIGA. Shown as in Figure 1, CIGA is powered by an information-theoretic objective that extracts the subgraphs which maximally preserve the invariant intra-class information.
With certain assumptions, CIGA provably identifies the underlying invariant subgraphs (shown as the orange subgraphs).
Learning with these subgraphs is immune to distribution shifts. 

We implement CIGA using the interpretable GNN architecture, where the featurizer $g$ is designed to extract the invariant subgraph, and a classifier $f_c$ is designed to classify the extracted subgraph.
The objective is imposed as an additional contrastive penalty to enforce the invariance of the extracted subgraphs at a latent sphere space (CIGAv1).

1. When the size of underlying $G_c$ is known and fixed across different graphs and environments, CIGAv1 is able to identify $G_c$. 
2. While it is often the case that the underlying $G_c$ varies, we further incorporate an additional penalty that maximizes $I(G_s;Y)$ to absorb potential spurious parts in the estimated $G_c$ (CIGAv2).

Extensive experiments on $16$ synthetic or real-world datasets, including a challenging setting -- DrugOOD, from AI-aided drug discovery, validate the superior OOD generalization ability of CIGA.


## Instructions

### Installation and data preparation
Our code is based on the following libraries:

```
torch==1.9.0
torch-geometric==1.7.2
scikit-image==0.19.1 
```

plus the [DrugOOD](https://github.com/tencent-ailab/DrugOOD) benchmark repo.

The data used in the paper can be obtained following the [instructions](./dataset_gen/README.md).

### Reproduce results
We provide the hyperparamter tuning and evaluation protocal details in the paper and appendix.

To obtain results of ERM, simply run 
```
python main.py --erm
```
with corresponding datasets and models specifications.

We use `--ginv_opt` to specify the interpretable GNN architectures, 
which can be `asap` or `gib` to test with ASAP or GIB respectively.
Note that `--r` is also needed for interpretable GNN architectures.
To test with DIR, simply specify `--ginv_opt` as default and `--dir` a value larger than `0`.

To test with CIGAv1, simply specify `--ginv_opt` as default, and `--contrast` a value larger than `0`.
While for CIGAv2, additionally specify `--spu_coe` to include the other objective.

To test with invariant learning baselines, specify `--num_envs=2` and
use `--irm_opt` to be `irm`, `vrex`, `eiil` or `ib-irm` to specify the methods,
and `--irm_p` to specify the penalty weights.

Due to the additional dependence of a ERM reference model in CNC, we need to train a ERM model and save it first,
and then load the model to generate ERM predictions for postive/negative pairs sampling in CNC. 
Here is an simplistic example:
```
python main.py --erm --contrast 0 --save_model
python main.py --erm --contrast 1  -c_sam 'cnc'
```

## Reference

If you find our paper and repo useful, please cite our paper:

```bibtex
@InProceedings{chen2022ciga,
  title       = {Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs},
  author      = {Yongqiang Chen and Yonggang Zhang and Yatao Bian and Han Yang and Kaili Ma and Binghui Xie and Tongliang Liu and Bo Han and James Cheng},
  booktitle   = {Advances in Neural Information Processing Systems},
  year        = {2022}
}
```
Ack: The readme is inspired from [GSAT](https://github.com/Graph-COM/GSAT) 😄.
