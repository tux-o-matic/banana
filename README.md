## Setup
For training this CNN with over 200 steps and more than 10 epochs, a GPU with CUDA support is a must. CPU based training will take several hours.

At this time, all tests were done with CUDA 9.0.
### Nvidia drivers
You'll need the closed source Nvidia drivers, one supporting the correct CUDA version.
Blacklist the Nouveau open source driver it's loaded at boot.
### CUDA
Some environments, such as Python 3 from SCL require to export the CUDA path:
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
```
### CUDAnn
Requires a developer account on Nvidia's website to download. Extract the content of the tarball.
```shell
sudo cp cuda/include/cudnn.h /usr/include
sudo cp cuda/lib64/* /usr/lib64/
```
### TensorFlow
Rename the entry requirements.txt from
```ini
tensorflow==2.1.0
```
to
```ini
tensorflow-cpu==2.1.0
```
If you do not have an Nvidia GPU and which to use the lighter module which will run only on CPU.
If your CPU doesn't support the AVX1 instruction set (Intel chip shipped before 2012), you'll need to use an older version of TensorFlow such as 1.5.0 (see keras-tf1 branch).


## Results
So far training gave an accuracy of 78%.
