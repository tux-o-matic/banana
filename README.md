## Setup

At this time, all tests were done with CUDA 9.0.
### Nvidia drivers
You'll need the closed source Nvidia drivers, one supporting the correct CUDA version.
Blacklist the Nouveau open source driver it's loaded at boot.
### CUDA
Some environmens, such as Python 3 from SCL require to export the CUDA path:
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
tensorflow==1.8.0
```
to
```ini
tensorflow-gpu==1.8.0
```
If your CPU doesn't support AVX2 (Intel chip shipped before 2012), you'll need to use an older version of TensorFlow such as 1.5.0.
