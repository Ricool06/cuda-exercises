__Requires__:
* A CUDA enabled device (most modern NVIDIA GPUs)
* CUDA Toolkit
* conan (conan.io)

__How to build & run tests__:
```
$ mkdir build && cd build
$ conan install ..
$ conan build ..
```

__Notes__:
* Not tested on Windows

__Credit__:
* Some code from: https://devblogs.nvidia.com/even-easier-introduction-cuda
* Conan examples from:
    * https://github.com/lasote/conan-gtest-example
    * https://github.com/maitesin/tries
* *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu