import setuptools

with open('./README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('./requirements.txt', 'r') as rfh:
    lib_flag = False
    requirements = []

    for line in rfh.readlines():
        requirements.append(line)


setuptools.setup(
    name='seqfishtools',
    version='0.0.1',
    author='Lincoln Ombelets',
    author_email='lombelets@caltech.edu',
    description='Miscellaneous useful functions for seqFISH data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CaiGroup/useful-tools',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    package_dir={'seqfishtools': './seqfishtools'},
    packages=['seqfishtools'],
    python_requires='>=3.9',
    install_requires=requirements
)