from setuptools import setup, find_packages

setup(
    name='simple3D',
    version='0.1.0',
    author='Francois Serra',
    author_email='serra.francois@gmail.com',
    description='Simple optimizer of spatial restraints using SciPy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fransua/simple3D',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPLv3 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
