# STAEformer: Spatio-Temporal Adaptive Embedding Network

#### H. Liu*, Z. Dong*, R. Jiang#, J. Deng, J. Deng, Q. Chen, X. Song#, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", Proc. of 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023. (*Equal Contribution, #Corresponding Author)

## Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

## Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- METRLA
- PEMSBAY
- PEMS04
- PEMS07
- PEMS08
