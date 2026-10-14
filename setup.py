from setuptools import find_packages, setup


setup(
    name="pv-dinov3-adapter",
    version="0.1.0",
    description="InternVL-guided DINOv3 adapter for photovoltaic segmentation",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
)
