# Action4Pose_FG2017
Source code for our FG'17 paper: 

**Umar Iqbal, Martin Garbade, Juergen Gall**  
 Action for Pose - Pose for Action
IEEE Conference on Automatic Face and Gesture Recognition (FG) 2017. 

For more information visit http://pages.iai.uni-bonn.de/iqbal_umar/action4pose/ 

## Installation

##### Dependencies
- [OpenCV](http://opencv.org/downloads.html)
- [Boost-1.55](http://www.boost.org/)
- [Caffe](http://caffe.berkeleyvision.org/)
- [CUDA >= 7](https://developer.nvidia.com/cuda-zone)
- [LibConfig++](http://www.hyperrealm.com/libconfig/)
- [GFlags](https://github.com/gflags/gflags)
- [GLOG](https://github.com/google/glog)

##### Installation Instructions
1. Clone repository	
   ```
   $ git clone https://github.com/iqbalu/Action4Pose_FG2017.git --recursive
   ```
2. Download and build Caffe. Follow [Caffe building instructions](http://caffe.berkeleyvision.org/installation.html). 

3. Build the code
  ```
  $ cd demo_PennAction && mkdir build && cd build
  $ cmake ..
  $ make
  ```
  if cmake is not available, a simple Makefile is also provide:
  ```
  $ make
  ```
  

#### Download models

## Citing
```
@inproceedings{iqbal2017FG,
	author = {Umar Iqbal and Martin Garbade, Juergen Gall},
	title = {Action for Pose - Pose for Action},
	booktitle = {IEEE Conference on Automatic Face and Gesture Recognition},
	year = {2017},
	url = {https://arxiv.org/pdf/1603.04037.pdf}
}
```
