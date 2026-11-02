from setuptools import setup, find_packages

setup(
    name="pint-unlearning",
    version="0.1.0",
    author="Andrii Shkabrii, Claudia Plant, Martin Pawelczyk",
    description="PINT: LLM unlearning via statistical independence, built on OpenUnlearning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    extras_require={
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install ".[dev]"`
    },
    python_requires=">=3.11",
)
