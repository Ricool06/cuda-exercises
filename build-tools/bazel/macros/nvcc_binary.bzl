def nvcc_binary(name, srcs, visibility=None):
    native.genrule(
        name = "nvcc_binary_" + name,
        srcs = srcs,
        outs = [name],
        cmd = "nvcc $(SRCS) -o $@",
        visibility = visibility,
        output_to_bindir = 1,
    )