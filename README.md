# Setup
For training this CNN with over 200 steps and more than 10 epochs, a GPU is a must. CPU based training will take several hours but for portability reasons, the `requirements.txt` file installs `tensorflow-cpu` by default on systems other than macOS.

All tests were done with Metal on Apple Silicon and CUDA 9.0 on Linux x86_64.

## Metal
pip will auto detect macOS and install the improved TensorFlow implementations from Apple to use Metal on Apple Silicon and x86_64 Mac with discrete AMD GPU. Integrated Intel GPU support Metal shaders but those TF backend didn't boost TF performance (Apple therefor gave up supporting Intel iGPU for the Metal backend) so those Mac should fallback to `tensorflow-cpu` or try the Intel distribution with further SSE/AVX optimization `intel-tensorflow`.

### vecLib
NumPy performance on Apple Silicon can be further improved by using [Apple's vecLib](https://developer.apple.com/documentation/accelerate/veclib) instead of the default OpenBLAS otherwise used by the arm64 NumPy wheel. A `site.cfg` file is present in the repository to pass hints when building NumPy.
Within your virtualenv:
```shell
pip install Cython
pip install -U --no-binary :all: --no-use-pep517 numpy
```

To verify, within an interactive Python interpreter in your virtualenv:
```shell
>>> import numpy
>>> numpy.show_config()
```
You should see the vecLib headers listed.

Based on [this post](https://developer.apple.com/forums/thread/695963).

## Nvidia drivers
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
tensorflow-cpu=XXX
```
to
```ini
tensorflow==XXX
```


## Results
So far training gave an accuracy of 78%.
The more RAM and bandwith your GPU has, the larger batch it can handle. So the `batch_size` value should be tuned accordingly.

## Known Issues
Apple's Metal backend for Tensorflow has been broken for a few release after TensorFlow 2.9 and `tensorflow-metal` 0.5. So even though it might lack some MPS optimizations, those older versions can actually run.

Apple's Metal optimized TensorFlow can have "likely leaking IOGPUResource" [errors](https://developer.apple.com/forums/thread/706920?login=true&page=1#741165022) after running for a few epochs.

If your CPU doesn't support the AVX1 instruction set (Intel chip shipped before 2012), you'll need to use an older version of TensorFlow such as 1.5.0 (see keras-tf1 branch).
