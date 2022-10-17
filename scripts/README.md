
# Reproducing Results

Our datasets are generated as the following procedures.

## SPMotif Datasets

We follow DIR setup and report the mean and standard deviation of test accuracy from `5` runs. The running scripts are `spmotif-struc.sh` and `spmotif-mixed.sh`. 
The parameter search spaces are `[0.5,1,2,4,8,16,32]` and `[0.5,1,2]` for $\alpha$ and $\beta$, respectively.  
Experiments on these datasets run on NVIDIA TITAN Xp and NVIDIA RTX 2080Ti graphics cards with CUDA 10.2.

## DrugOOD Datasets

We follow DrugOOD setup and report the mean and standard deviation of test auc-roc from `5` runs. The running script is `drugood.sh`. 
The parameter search spaces are `[0.5,1,2,4,8,16,32]` and `[0.5,1,2]` for $\alpha$ and $\beta$, respectively. 
All experiments on these datasets run on NVIDIA RTX 3090Ti graphics cards with CUDA 11.3.


## CMNIST-sp, Graph-SST5 and Twitter

The running script is `others.sh`. We report the mean and standard deviation of test accuracy from `5` runs.
For CMNIST-sp, the parameter search space is `[1,2,4,8,16,32]` for both $\alpha$ and $\beta$, respectively, as we empirically find a larger  $\beta$ is necessary for the dataset.
For Graph-SST and Twitter, the parameter search spaces are `[1,2,4,6,8]` and `[0.5,1,2]` for $\alpha$ and $\beta$, respectively. 
All experiments on these datasets run on NVIDIA TITAN Xp graphics cards with CUDA 10.2.

## NCI1, NCI109, PROTEINS and DD

We follow [size-invariant-GNNs](https://github.com/PurdueMINDS/size-invariant-GNNs) and report the mean and standard deviation of Matthews correlation coefficient from `10` runs. The running script is `tu_datasets.sh`. The parameter search space is in `[0.5,1,2]` for both $\alpha$ and $\beta$. All experiments on these datasets run on NVIDIA RTX 2080Ti graphics cards with CUDA 10.2.

## Reproducing Tips
- We suggest re-searching the hyper-parameters from the same space if there is a shift of the software or hardware setups. 
- In fact, using a finegrained hyperparemter search space for $\alpha$ and $\beta$ can obtain better results (cf. the ablation studies in experiments and Appendix G.4).
- Using more powerful GNNs as backbones is also likely to improve the performances (cf. the failure case studies in Appendix D.2).
- We also empirically observe high variances for datasets involved with graph size shifts, which aligns with the results of the benchmarking results from [GOOD](https://github.com/divelab/GOOD) benchamrk. Therefore we believe a lower variance could also be a good indicator for the OOD performance. 
