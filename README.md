# Jt-NeRF

## Introduction
Implement **NeRF-base-model** in jittor, mainly to provide an API that is easy to debug and compare, and to prepare for the implementation of dynamic networks.

## Quick Start
### Enviroment
You need following package:
- `jittor`: You can get the jittor in their website: [[Jittor-Install]](https://cg.cs.tsinghua.edu.cn/jittor/download/)
- `numpy`: You can use `pip install numpy` to get it.
- `opencv-python`: You can use `pip install opencv-python` to get this package.

### Run the codes
Make sure that you have `tiny_nerf_data.npz` in your `data` folder.
Then, run:
```powershell
python main.py
```

Some Hyperparameters:
- `--test`: 还没写
- `--iter`: iters. Default `1000`
- `--encpara`: Parameters of positional encoding. Default `6`


## To Do List

- [x] Component of NeRF
- [x] Complete the Baseline's implementation
- [x] Output the image
- [x] Compute PSNR
- [ ] `test` mode
- [ ] Load the dataset with the submodule of `Dataset` Class
- [ ] Save and Get the model's parameters
- [ ] Implement the `ray_marching` part's `CUDA` codes in Jittor.