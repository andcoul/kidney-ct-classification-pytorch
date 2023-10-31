import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "kidney-ct-classification-pytorch"
AUTHOR_USER_NAME = "andcoul"
SRC_REPO = "kidneyCtClassifier"
AUTHOR_EMAIL = "andycoul@yahoo.com"

setuptools.setup(
    version=__version__,
    name=SRC_REPO,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Kidney ct image classification",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")

)
