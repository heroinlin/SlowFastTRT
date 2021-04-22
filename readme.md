# SlowFast接口使用说明

## 开源代码

[https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)

使用commit分支:  c22c7c91d0dfa56dec1f733a02d331dd34492d67

## 安装环境

> 设置pip安装源为阿里源

```shell
mkdir -p ~/thirdpartys/slowfast/lib/python3.7/site-packages

pip install cython torch torchvision opencv-python fvcore tensorboard matplotlib cloudpickle pydot mock -t ~/thirdpartys/slowfast/lib/python3.7/site-packages
```

### 安装detectron2

打开git clone 拉取的官方代码

```shell
git clone https://github.com/facebookresearch/detectron2.git
```

```shell
cd detectron2
env PYTHONPATH=/home/heroin/thirdpartys/slowfast/lib/python3.7/site-packages python setup.py install --prefix  ~/thirdpartys/slowfast/ 
```

> tensorboard matplotlib cloudpickle pydot mock是detectron2依赖的包
>
> 需要手动把pycocotools和detectron2从*.egg文件夹中拷贝出来

