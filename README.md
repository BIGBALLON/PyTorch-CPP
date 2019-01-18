# pytorch_cpp

This demo will show you how to use libtorch to build your C++ application.

## Contents

1. [Requirements](#requirements)
2. [Build](#build)
3. [Usage](#usage)


## Requirements

- Pytorch (tag: pytorch v1.0)
- Libtorch
- OpenCV

## Build

### Step 1

Export your pytorch model to torch script file, We will simply use resnet50 in this demo

### Step 2

Write your C++ program, check the file ``prediction.cpp`` for more detial.  

PS: ``module->to(at::kCUDA)`` and ``input_tensor.to(at::kCUDA)`` will switch your model & tensor to GPU mode,  
comment out them if you just want to use CPU mode. 


### Step 3

Write a ``CMakeLists.txt``, the version of OpenCV must the same as your libtorch.
Otherwise, you may get the compile error:

```
error: undefined reference to `cv::imread(std::string const&, int)'
```

check [issues 14684](https://github.com/pytorch/pytorch/issues/14684) and [issues 14620](https://github.com/pytorch/pytorch/issues/14620) for more details.

## Usage

- run ``model_trace.py``,   then you will get a file ``resnet50.pt``
- compile your cpp program, you need to use ``-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch``, for example:

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/cgilab/pytorch/torch/lib/tmp_install ..
make
```

- test your program

``classifier <path-to-exported-script-module> <path-to-lable-file>``

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
![](./pic/dog.jpg)

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
![](./pic/shark.jpg)


Take it easy!!
