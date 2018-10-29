__Requires__:
* A CUDA enabled device (most modern NVIDIA GPUs)
* CUDA Toolkit
* conan (conan.io)

__Run tests__:
```
$ cd test
$ mkdir build && cd build
$ conan install ..
$ conan build ..
```

__Notes__:
* Not tested on Windows

__Credit__:
* Some code from: https://devblogs.nvidia.com/even-easier-introduction-cuda
* Conan template from: https://github.com/lasote/conan-gtest-example
* *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu