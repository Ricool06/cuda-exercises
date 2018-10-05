load(":nvcc_toolchain.bzl", "nvcc_toolchain")
package(default_visibility = ["//visibility:public"])

toolchain(
    name = "nvcc_linux_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":nvcc_linux",
    toolchain_type = ":toolchain_type",
)

nvcc_toolchain(
    name = "nvcc_linux",
    compiler_path = "/usr/local/cuda/bin/nvcc",
)

toolchain_type(
    name = "toolchain_type",
    visibility = ["//visibility:public"],
)