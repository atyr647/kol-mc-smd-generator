from setuptools import setup, find_packages

setup(
    name="ko2mc",
    version="1.0.0",
    description="Convert Knight Online .gtd/.opd map files to Minecraft worlds",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=9.0",
    ],
    package_data={"ko2mc": ["viewer.html"]},
    entry_points={
        "console_scripts": [
            "ko2mc=ko2mc.__main__:main",
            "ko2mc-preview=ko2mc.preview:main",
        ],
    },
)
