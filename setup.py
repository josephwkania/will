"""
Install will.
"""
# import glob

from setuptools import find_packages, setup

with open("will/_version.py", encoding="UTF-8") as f:
    for line in f:
        if "__version__" in line:
            line = line.replace(" ", "").strip()
            version = line.split("__version__=")[1].strip('"')

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="will",
    version=version,
    description="Weighted Injector of Luminous Lighthouses",
    url="https://github.com/josephwkania/will",
    author="Joseph W Kania",
    packages=find_packages(),
    # scripts=glob.glob("bin/*"),
    python_requires=">=3.6, <4",
    install_requires=required,
    extras_require={"tests": ["pytest", "pytest-cov"], "cupy": ["cupy>=9.2"]},
)
