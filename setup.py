from setuptools import setup

setup(name='inversefuns',
      version='0.1',
      description='Functions for Lei Yang dissertation INFINITE DIMENSIONAL STOCHASTIC INVERSE PROBLEMS',
      url='https://github.com/yangleicq/inversefuns',
      author='Lei Yang',
      author_email='yangleicq@gmail.com',
      license='MIT',
      packages=['inversefuns'],
      install_requires=['BET', 'scipy',
                        'numpy', 'matplotlib'])
