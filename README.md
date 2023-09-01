## STAEformer: Spatio-Temporal Adaptive Embedding Transformer

#### H. Liu*, Z. Dong*, R. Jiang#, J. Deng, J. Deng, Q. Chen, X. Song#, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", Proc. of 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023. (*Equal Contribution, #Corresponding Author)

![model_arch](https://github.com/XDZhelheim/STAEformer/assets/57553691/f0620d5b-2b7f-47bc-bf76-5fccf48fae35)

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

## Performance

![perf1](https://github.com/XDZhelheim/STAEformer/assets/57553691/8049bce2-9bc2-4248-a911-25468e9bbab4)

<img src="https://github.com/XDZhelheim/STAEformer/assets/57553691/5702ecaf-6315-4350-be22-b53cef1b1205" width="500">

## Citation

```
@article{liu2023spatio,
  title={Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting},
  author={Liu, Hangchen and Dong, Zheng and Jiang, Renhe and Deng, Jiewen and Deng, Jinliang and Chen, Quanjun and Song, Xuan},
  journal={arXiv preprint arXiv:2308.10425},
  year={2023}
}
```


