from setuptools import Extension, setup

# ----------- MATRIXS -----------
module = Extension("myMatrixs", sources=['symnmfmodule.c', 'symnmf.c'])
setup(name='myMatrixs',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])
# ----------- MATRIXS -----------
