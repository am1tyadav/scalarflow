import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()


setuptools.setup(
    name="scalarflow",
    version="0.1.2",
    author="Amit Yadav",
    author_email="amit.yadav.iitr@gmail.com",
    description="A Machine Learning library written in pure Python for educational purpose.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/am1tyadav/scalarflow.git",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.0",
)
