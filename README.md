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
2. Download models
   ```
   $ cd data
   $ ./download_models.sh
   ```
3. Download and build Caffe. Follow [Caffe building instructions](http://caffe.berkeleyvision.org/installation.html).  
   You may also need to run the following commands to connect Caffe with the code using CMake. 
   ```
   $ cd [caffe_dir]
   $ mkdir cmake_build && cd cmake_build
   $ cmake .. -DBUILD_SHARED_LIB=ON
   $ cmake . -DCMAKE_BUILD_TYPE=Release   
   $ make -j 12 && make install          
   ```
4. Build the code
  ```
  $ cd demo_PennAction && mkdir build && cd build
  $ cmake ..
  $ make
  ```
5. Running 
  ```
  $ ./bin/demo_PennAction config_file.txt
  ```
  Parameters in Config file:
  - save: set to True if you want to save the poses in a text file. The text file will be stored in cache folder. Set it to False if you only want to visualize the results.
  - Other parameters are self explanatory.
  
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
