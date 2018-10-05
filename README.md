__Requires__:
* A CUDA enabled device (most modern NVIDIA GPUs)
* CUDA Toolkit
* Bazel

__How to use__:
1. Run `bazel run //<package_name>`. e.g. To build and run the code in the vector-addition package, run `bazel run //vector-addition`.
1. That's it. :sweat_smile:

__Notes__:
* Some folders will also contain a main.cpp file showing how the same program can be written in a serial, single-threaded manner. This is not included in the bazel builds, so build and run it however you like.
* Not tested on Windows

__Credit__:
* Some code from: https://devblogs.nvidia.com/even-easier-introduction-cuda/
* *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu