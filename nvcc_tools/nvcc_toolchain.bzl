NvccInfo = provider(
    doc = "Information about how to invoke the nvcc compiler.",
    fields = ["compiler_path"],
)

def _impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        nvccinfo = NvccInfo(
            compiler_path = ctx.attr.compiler_path,
        ),
    )
    return [toolchain_info]

nvcc_toolchain = rule(
    _impl,
    attrs = {
        "compiler_path": attr.string(),
    },
)