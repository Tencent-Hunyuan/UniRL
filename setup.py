from setuptools import find_packages, setup

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
    package_data={
        "unirl.models.janus_pro.vendor": ["LICENSE-CODE", "VENDOR_COMMIT.txt"],
    },
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
        "cosmos3": [
            "diffusers>=0.39",
        ],
        "infer": [
            "accelerate>=0.30",
            "einops>=0.7",
            "timm>=0.9.16",
        ],
        "eval": [
            "torchvision>=0.16",
            "paddlepaddle==3.2.2",
            "paddleocr==3.5.0",
            "python-Levenshtein>=0.27",
        ],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff>=0.6",
            "pre-commit>=3.6",
        ],
    },
)
