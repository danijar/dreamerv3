import pathlib
import setuptools


def parse_reqs(filename):
  requirements = pathlib.Path(filename).read_text(encoding='utf-8').splitlines()
  requirements = [x.split('#', 1)[0].strip() for x in requirements]
  requirements = [x for x in requirements if x]
  return requirements


setuptools.setup(
    name='dreamer',
    version='3.4.0',
    author='Danijar Hafner',
    author_email='mail@danijar.com',
    description='Mastering Diverse Domains through World Models',
    url='http://github.com/danijar/dreamerv3',
    long_description=pathlib.Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    python_requires='>=3.11',
    packages=setuptools.find_namespace_packages(
        include=['dreamerv3*', 'embodied*']),
    include_package_data=True,
    package_data={
        'dreamerv3': ['*.yaml'],
        'embodied.envs': ['*.xml'],
    },
    install_requires=parse_reqs('requirements.txt'),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
