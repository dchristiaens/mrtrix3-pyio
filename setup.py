from setuptools import setup

setup(name='mrtrix3',
      version='1.0.0',
      description='Python package to interface with MRtrix3 file formats.',
      url='http://github.com/dchristiaens/mrtrix3-pyio.git',
      author='Daan Christiaens',
      author_email='daan.christiaens@kcl.ac.uk',
      license='MPL',
      packages=['mrtrix3'],
      install_requires=['numpy'],
      zip_safe=False)

