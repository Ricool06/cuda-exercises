load("//nvcc_tools:nvcc_binary.bzl", "nvcc_binary")

platform(
    name = "linux",
    constraint_values = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
)

nvcc_binary(
    name = "vector-addition",
    srcs = glob(["src/**/*.cu", "src/**/*.h"]),
)