Boosting 3D Point Cloud Registration by Transferring Multi-modality Knowledge
===
This repository represents the official implementation of the paper:
[Boosting 3D Point Cloud Registration by Transferring Multi-modality Knowledge](https://ieeexplore.ieee.org/document/10161411)


#### Requirements and data
PLease refer to https://github.com/XiaoshuiHuang/IMFNet. 


### Training
```shell
python train_3dmatch.py
```

### Evaluating
```shell
python ./scripts/generate_desc_kp.py
python ./scripts/evaluation_3dmatch.py
```


### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{yuan2023boosting,
  title={Boosting 3D Point Cloud Registration by Transferring Multi-modality Knowledge},
  author={Yuan, Mingzhi and Huang, Xiaoshui and Fu, Kexue and Li, Zhihao and Wang, Manning},
  journal={arXiv preprint arXiv:2302.05210},
  year={2023}
}
```
