from setuptools import setup, find_packages

setup(
    name="rl2024",
    version="0.1",
    description="Reinforcement Learning in UoE (CW)",
    author="Filippos Christianos, Mhairi Dunion, Samuel Garcin, Shangmin Guo, Trevor McInroe, Lukas Schaefer, ",
    url="https://github.com/uoe-agents/uoe-rl2024-coursework",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "numpy>=1.18",
        "torch>=1.3",
        "gymnasium>=0.26",
        "gymnasium[box2d]",
        "tqdm>=4.41",
        "pyglet>=1.3",
        "matplotlib>=3.1",
        "pytest>=5.3",
        "pytest-csv>=3.0",
        "pytest-json>=0.4",
        "pytest-json-report>=1.5",
        "pytest-timeout>=2.1",
        "highway-env"
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
