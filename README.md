

<img src="./pic/pytorch-cpp.jpg" width=30% align="right" /> 

This demo will demonstrate how to use LibTorch to build your C++ application.

**[UPDATE 2020/02/22]** : Thanks for [Ageliss](https://github.com/Ageliss) and his [PR](https://github.com/BIGBALLON/PyTorch-CPP/pull/4), which update this demo to fit LibTorch1.4.0 and OpenCV4.0.  
**[UPDATE 2020/04/15]** : Retest this tutorial with **OpenCV4.3**/**PyTorch1.4**/**LibTorch1.4**, update readme for beginner.  
**[UPDATE 2020/04/25]** : Update ``CMakeLists.txt`` to fit **C++14,** retest with **CUDA10.2**/**PyTorch1.5**/**LibTorch1.5**.  


## Contents

- [Contents](#contents)
- [Requirements](#requirements)
- [Preparation](#preparation)
  - [Step 0x00](#step-0x00)
  - [Step 0x01](#step-0x01)
  - [Step 0x02](#step-0x02)
  - [Step 0x03](#step-0x03)
- [Build](#build)
- [Usage](#usage)


## Requirements 

- PyTorch (>= 1.5.0)
- LibTorch (>= 1.5.0)
- OpenCV (>= 4.0)

## Preparation


### Step 0x00

**Make sure** LibTorch and OpenCV have been installed correctly.

- **Install OpenCV**: for [Linux](https://docs.opencv.org/4.3.0/d7/d9f/tutorial_linux_install.html), for [Mac OS](https://docs.opencv.org/4.3.0/d0/db2/tutorial_macos_install.html)
- **Get LibTorch**: download LibTorch package from the official [website](https://pytorch.org/get-started/locally/), then unpack it, for example:

```bash
cd path_to_your_workspace
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.0.zip
unzip libtorch-shared-with-deps-1.5.0.zip
```

### Step 0x01

Export PyTorch model to torch script file, we use ``resnet50`` in this demo. (see [model_trace.py](./model_trace.py))

### Step 0x02

Write C++ application program. (see [prediction.cpp](./prediction.cpp))

**PS**: ``module->to(at::kCUDA)`` and ``input_tensor.to(at::kCUDA)`` will switch your model & tensors to GPU mode, comment out them if you just want to use CPU. 


### Step 0x03

Write a [CMakeLists.txt](./CMakeLists.txt). (check [cppdocs](https://pytorch.org/cppdocs/) for more details)

## Build

- run ``model_trace.py``, you will get a converted model ``resnet50.pt``.
- compile your cpp program, you need to use ``-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch``, for example:

```bash
mkdir build
cd build
# change "/home/bigballon/libtorch" to your libtorch path
cmake -DCMAKE_PREFIX_PATH=/home/bigballon/libtorch ..
make
```


**PS**: If you get the compile error: ``error: undefined reference to `cv::imread(std::string const&, int)'``, check [issues 14684](https://github.com/pytorch/pytorch/issues/14684) and [issues 14620](https://github.com/pytorch/pytorch/issues/14620) for more details.

## Usage


```bash
classifier <path-to-exported-script-module> <path-to-lable-file>
# example:
# ./classifier ../resnet50.pt ../label.txt
```

![video](./pic/video.gif)

```
> ./classifier ../resnet50.pt ../label.txt
== Switch to GPU mode
== ResNet50 loaded!
== Label loaded! Let's try it
== Input image path: [enter Q to exit]
../pic/dog.jpg
== image size: [976 x 549] ==
== simply resize: [224 x 224] ==
    ============= Top-1 =============
    Label:  beagle
    With Probability:  99.1228%
    ============= Top-2 =============
    Label:  Walker hound, Walker foxhound
    With Probability:  0.469356%
    ============= Top-3 =============
    Label:  English foxhound
    With Probability:  0.110916%
== Input image path: [enter Q to exit]
```
![dog](./pic/dog.jpg)

```
../pic/shark.jpg
== image size: [800 x 500] ==
== simply resize: [224 x 224] ==
    ============= Top-1 =============
    Label:  tiger shark, Galeocerdo cuvieri
    With Probability:  92.2599%
    ============= Top-2 =============
    Label:  great white shark, white shark, man-eater, man-eating shark
    With Probability:  5.94252%
    ============= Top-3 =============
    Label:  hammerhead, hammerhead shark
    With Probability:  1.77418%
== Input image path: [enter Q to exit]
Q
```
![shark](./pic/shark.jpg)


Take it easy!! :love_letter:
