import setuptools

setuptools.setup(
    name='quantum-simba',
    version='0.9.6dev',
    packages=['simba'],
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Joe Bentley",
    author_email="joebentley10@gmail.com",
    description="Systematic realisation of quantum systems directly from transfer function.",
    url="https://github.com/joebentley/simba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'sympy>=1.5.1', 'pytest'],
    python_requires='>=3.6'
)
