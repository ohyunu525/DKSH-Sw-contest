"""Install the DKSH Isaac Lab extension."""

from setuptools import find_packages, setup


setup(
    name="dksh_isaaclab",
    version="0.1.0",
    description="Isaac Lab environments for the DKSH spiderbot",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["gymnasium"],
    zip_safe=False,
)
