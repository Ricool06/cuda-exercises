platform(
    name = "linux",
    constraint_values = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
)

cc_binary(
    name = "vector-addition",
    srcs = glob(["src/**/*.cc", "src/**/*.h"]),
)

# cc_test(
#     name = "vector-addition-tests",
#     srcs = glob(["test/**/*.cc", "test/**/*.h"]),
#     deps = ["@gtest//:gtest"],
# )