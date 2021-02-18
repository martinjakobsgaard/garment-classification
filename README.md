# garment-classification

## Prerequisites
This project is used with a Nvidia RTX 3060Ti, and the following installation 
is based on compatability with it.

### CUDA 11.2.1
Install CUDA:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
sudo sh cuda_11.2.1_460.32.03_linux.run
```
Add the following to your .bashrc.
```bash
export PATH="/usr/local/cuda-11.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH"
```
Add the following symlink hack to ensure compatibility with TensorFlow:
```bash
sudo ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.10
```

### cuDNN 8.0.5
From [cuDNN archives](https://developer.nvidia.com/rdp/cudnn-archive) download the following:
```bash
cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1
```

Unzip the cuDNN package.
```bash
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
```
Copy the following files into the CUDA Toolkit directory.
```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### TensorFlow GPU 2.4.0
```bash
wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.4.0-cp36-cp36m-manylinux2010_x86_64.whl
pip3 install tensorflow_gpu-2.4.0-cp36-cp36m-manylinux2010_x86_64.whl
```


