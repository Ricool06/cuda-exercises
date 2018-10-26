from conans.model.conan_file import ConanFile
from conans import CMake


class CudaExercisesTest(ConanFile):
    name = "cuda-exercises-test"
    version = "0.1.0"
    author = "Ricool06"
    url = "https://github.com/Ricool06/cuda-exercises"
    license = "MIT"
    settings = "os", "compiler", "arch", "build_type"
    generators = "cmake"
    cmake = None
    requires = "gtest/1.8.1@bincrafters/stable"
    default_options = "gtest:shared=True"

    def build(self):
        self.cmake = CMake(self)
        self.cmake.configure()
        self.cmake.build()

    def imports(self):
        self.copy("*.so", "bin", "lib")
        self.copy("*.dll", "bin", "bin")
        self.copy("*.dylib", "bin", "lib")

    def test(self):
        target_test = "RUN_TESTS" if self.settings.os == "Windows" else "test"
        self.cmake.build(target=target_test)
