import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fremu",
    version="0.0.6",
    author="Jiachen Bai",
    author_email="astrobaijc@gmail.com",  
    description="Emulator for f(R) gravity",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="https://github.com/astrobai/fremu",
    py_modules=['fremu'],
    packages=['fremu'],
    package_data={'fremu': ['cache/*']},
    install_requires=[
        'numpy',
        'scipy',
        'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
