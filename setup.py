import setuptools


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='openood',
    version='1.5',
    author='openood dev team',
    author_email='jingkang001@e.ntu.edu.sg',
    description='This package provides a unified test platform for '
    'Out-of-Distribution detection.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Jingkang50/OpenOOD',
    packages=setuptools.find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)
