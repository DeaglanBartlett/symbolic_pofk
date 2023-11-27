from setuptools import setup

setup(
    name='symbolic_pofk',
    version='0.1.0',
    description='A short python module to compute symnbolic approximations to P(k)',
    url='https://github.com/DeaglanBartlett/symbolic_pofk',
    author='Deaglan Bartlett',
    author_email='deaglan.bartlett@physics.ox.ac.uk',
    license='MIT licence',
    packages=['symbolic_pofk'],
    install_requires=[
        'numpy',
        'colossus',
        'matplotlib',
        ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
