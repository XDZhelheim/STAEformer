## STAEformer: Spatio-Temporal Adaptive Embedding Transformer

#### H. Liu*, Z. Dong*, R. Jiang#, J. Deng, J. Deng, Q. Chen, X. Song#, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", Proc. of 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023. (*Equal Contribution, #Corresponding Author)

![model_arch](https://github.com/XDZhelheim/STAEformer/assets/57553691/f0620d5b-2b7f-47bc-bf76-5fccf48fae35)

#### Citation
```
@inproceedings{liu2023spatio,
  title={Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting},
  author={Liu, Hangchen and Dong, Zheng and Jiang, Renhe and Deng, Jiewen and Deng, Jinliang and Chen, Quanjun and Song, Xuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4125--4129},
  year={2023}
}
```

#### CIKM23 Proceedings (including METRLA, PEMSBAY, PEMS04, PEMS07, PEMS08 results)
[https://dl.acm.org/doi/abs/10.1145/3583780.3615160](https://dl.acm.org/doi/10.1145/3583780.3615160)

#### Preprints (including METRLA, PEMSBAY, PEMS03, PEMS04, PEMS07, PEMS08 results)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=STAEformer&color=red&logo=arxiv)](https://arxiv.org/abs/2308.10425)

#### Performance on Traffic Forecasting Benchmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems04)](https://paperswithcode.com/sota/traffic-prediction-on-pems04?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems07)](https://paperswithcode.com/sota/traffic-prediction-on-pems07?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems08)](https://paperswithcode.com/sota/traffic-prediction-on-pems08?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-metr-la)](https://paperswithcode.com/sota/traffic-prediction-on-metr-la?p=spatio-temporal-adaptive-embedding-makes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-adaptive-embedding-makes/traffic-prediction-on-pems-bay)](https://paperswithcode.com/sota/traffic-prediction-on-pems-bay?p=spatio-temporal-adaptive-embedding-makes)

![perf1](https://github.com/XDZhelheim/STAEformer/assets/57553691/8049bce2-9bc2-4248-a911-25468e9bbab4)

<img width="600" alt="image" src="https://github.com/XDZhelheim/STAEformer/assets/57553691/abf009aa-b145-451c-aff6-27031d60a612">

#### Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

#### Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- METRLA
- PEMSBAY
- PEMS03
- PEMS04
- PEMS07
- PEMS08
