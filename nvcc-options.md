```
Options for specifying the compilation phase
============================================
More exactly, this option specifies up to which stage the input files must be compiled,
according to the following compilation trajectories for different input file types:
        .c/.cc/.cpp/.cxx : preprocess, compile, link
        .o               : link
        .i/.ii           : compile, link
        .cu              : preprocess, cuda frontend, PTX assemble,
                           merge with host C code, compile, link
        .gpu             : cicc compile into cubin
        .ptx             : PTX assemble into cubin.

--cuda  (-cuda)
        Compile all .cu input files to .cu.cpp.ii output.

--cubin (-cubin)
        Compile all .cu/.gpu/.ptx input files to device-only .cubin files.  This
        step discards the host code for each .cu input file.

--fatbin(-fatbin)
        Compile all .cu/.gpu/.ptx/.cubin input files to device-only .fatbin files.
        This step discards the host code for each .cu input file.

--ptx   (-ptx)
        Compile all .cu/.gpu input files to device-only .ptx files.  This step discards
        the host code for each of these input file.

--preprocess                               (-E)
        Preprocess all .c/.cc/.cpp/.cxx/.cu input files.

--generate-dependencies                    (-M)
        Generate a dependency file that can be included in a make file for the .c/.cc/.cpp/.cxx/.cu
        input file (more than one are not allowed in this mode).

--compile                                  (-c)
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file.

--device-c                                 (-dc)
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
        relocatable device code.  It is equivalent to '--relocatable-device-code=true
        --compile'.

--device-w                                 (-dw)
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
        executable device code.  It is equivalent to '--relocatable-device-code=false
        --compile'.

--device-link                              (-dlink)
        Link object files with relocatable device code and .ptx/.cubin/.fatbin files
        into an object file with executable device code, which can be passed to the
        host linker.

--link  (-link)
        This option specifies the default behavior: compile and link all inputs.

--lib   (-lib)
        Compile all inputs into object files (if necessary) and add the results to
        the specified output library file.

--run   (-run)
        This option compiles and links all inputs into an executable, and executes
        it.  Or, when the input is a single executable, it is executed without any
        compilation or linking. This step is intended for developers who do not want
        to be bothered with setting the necessary environment variables; these are
        set temporarily by nvcc).
```

```
toolchain/nvcc-gcc.sh -MD -MF bazel-out/nvidia_gpu-fastbuild/bin/vector-addition/_objs/vector-addition/maymay.d '-frandom-seed=bazel-out/nvidia_gpu-fastbuild/bin/vector-addition/_objs/vector-addition/maymay.o' -iquote . -iquote bazel-out/nvidia_gpu-fastbuild/genfiles -iquote bazel-out/nvidia_gpu-fastbuild/bin -iquote external/bazel_tools -iquote bazel-out/nvidia_gpu-fastbuild/genfiles/external/bazel_tools -iquote bazel-out/nvidia_gpu-fastbuild/bin/external/bazel_tools -c vector-addition/src/maymay.cc -o bazel-out/nvidia_gpu-fastbuild/bin/vector-addition/_objs/vector-addition/maymay.o)
```