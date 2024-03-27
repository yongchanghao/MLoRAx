from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mlorax",
    version="0.9.5",
    description="MLoRAx is a minimalist library for low-rank adaptation designd to effortlessly enable parameter-efficient training for Transformer-based models.",  # noqa: E501
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yongchanghao/MLoRAx",
    author="Yongchang Hao",
    packages=["mlorax"],
    install_requires=["jax", "flax", "chex"],
    license="Apache 2.0",
)
