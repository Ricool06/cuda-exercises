from conans import ConanFile
from conans import CMake


class CudaExercises(ConanFile):
    name = "cuda-exercises"
    version = "0.1.0"
    url = "https://github.com/Ricool06/cuda-exercises"
    author = "Ricool06"
    license = "MIT"
    settings = "os", "arch", "compiler", "build_type"
    generators = "cmake"
    exports = "*"
    description = "Some CUDA C++ exercises"
    requires = ""
    options = {"shared": [True, False]}
    default_options = "shared=False"

    def build(self):
        shared = {"BUILD_SHARED_LIBS": self.options.shared}
        cmake = CMake(self)
        cmake.configure(defs=shared)
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include")
        self.copy("*.lib", dst="lib", src="lib", keep_path=False)
        self.copy("*.dll", dst="bin", src="bin", keep_path=False)
        self.copy("*.dylib", dst="bin", src="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["cuda-exercises"]