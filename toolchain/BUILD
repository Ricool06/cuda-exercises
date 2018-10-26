package(default_visibility = ['//visibility:public'])

cc_toolchain_suite(
    name = "nvcc",
    toolchains = {
         "nvidia_gpu|nvcc": ":nvidia-gpu-toolchain",
    },
)

filegroup(name = "empty")
filegroup(
    name = "all",
    srcs = [ 
        "nvcc-gcc.py",
        "nvcc-ld.py",
    ],
 )

cc_toolchain(
    name = "nvidia-gpu-toolchain",
    all_files = ":all",
    compiler_files = ":all",
    cpu = "nvidia_gpu",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":all",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
)