import os
from distutils.core import setup, Extension
import libceed
CEED_DIR = os.path.dirname(libceed.__file__)

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
qf_module = Extension("libceed_qfunctions",
                      include_dirs=[os.path.join(CEED_DIR, 'include')],
                      sources=["libceed-qfunctions.c"],
                      extra_compile_args=["-O3", "-std=c99",
                                          "-Wno-unused-variable",
                                          "-Wno-unused-function"])

setup(name="libceed_qfunctions",
      description="libceed qfunction pointers",
      ext_modules=[qf_module])

# ------------------------------------------------------------------------------
