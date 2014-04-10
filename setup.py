from setuptools import setup


# def readme():
#     with open('README.rst') as f:
#         return f.read()

setup(name='cudaconvnet',
      version='0.1',
      description='Cuda-convnet integrated with scikit-learn',
      url='https://github.com/alemagnani/cuda-convnet',
      author='',
      author_email='',
      license='MIT',
      packages=['cudaconvnet'],
      install_requires=[
          'scikit-learn'
      ],
      zip_safe=False)