from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="generalist-rl",
    version="0.0.1",
    author="anonymous",
    author_email="anonymous@anon.ymous",
    description="Generalist RL, unify api for different environments and algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        "torch",
        "numpy",
    ],
    entry_points={
        # 'console_scripts': [
        #     'srl-ray = rayrl.apps.main:main',
        # ]
    }
)