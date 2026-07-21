from setuptools import find_packages, setup

# Compatibility metadata for older pip/setuptools editable installs.
# Newer tooling uses pyproject.toml as the source of truth.
setup(
    name="unirl",
    version="0.1.0",
    description="Unified multimodal RL training framework",
    python_requires=">=3.12",
    include_package_data=True,
    packages=find_packages(
        where=".",
        include=(
            "unirl",
            "unirl.*",
        ),
    ),
    install_requires=[
        "numpy>=1.24,<3",
        "torch>=2.1",
        "ray[default]>=2.9,<3",
        "sglang[diffusion]==0.5.12.post1",
        "diffusers>=0.37.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "transformers>=5.6,<5.7",
        "peft>=0.14.0",
        "safetensors>=0.4",
        "Pillow>=10",
        "requests>=2.31",
        "psutil>=5.9",
        "tensordict>=0.5",
    ],
    extras_require={
        "train": [
            "wandb>=0.16,<0.20",
            "aiohttp>=3.9",
        ],
        "infer": [
            "accelerate>=0.30",
        ],
        "eval": [
            "torchvision>=0.16",
            "easyocr>=1.7",
        ],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff>=0.6",
            "pre-commit>=3.6",
        ],
    },
)
