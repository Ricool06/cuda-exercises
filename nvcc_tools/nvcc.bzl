def _impl(ctx):

    info = ctx.toolchains["//nvcc_tools:toolchain_type"].nvccinfo
    args = [src.path for src in ctx.files.srcs] + ["--output-file", ctx.outputs.out.path]

    ctx.actions.run(
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        arguments = args,
        progress_message = "NVCC is compiling srcs to create %s" % ctx.outputs.out.path,
        executable = info.compiler_path,
        use_default_shell_env = True,
    )

    return DefaultInfo(executable = ctx.outputs.out)

nvcc_binary = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
    toolchains = ["//nvcc_tools:toolchain_type"],
    outputs = {"out": "%{name}.binary"},
    executable = True,
)