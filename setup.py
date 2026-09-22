from setuptools import setup, find_packages

setup(
    name="ko2mc",
    version="1.1.0",
    description="Convert Knight Online .gtd/.opd map files to Minecraft worlds",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # NBT/Anvil output is written by ko2mc.mc_world itself (gzip/zlib/struct),
    # so numpy is the only runtime dependency.
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "test": ["pytest>=7"],
    },
    entry_points={
        "console_scripts": [
            # Legacy heightmap/pattern-matching converter (python -m ko2mc)
            "ko2mc=ko2mc.__main__:main",
            # OPD-driven mesh voxelization pipeline (python -m ko2mc.zone_converter)
            "ko2mc-zone=ko2mc.zone_converter:main",
        ],
    },
)
