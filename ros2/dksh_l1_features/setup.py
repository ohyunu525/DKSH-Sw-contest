from setuptools import find_packages, setup


PACKAGE_NAME = "dksh_l1_features"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DKSH",
    maintainer_email="dksh@example.invalid",
    description="Unitree L1 RM and L2 point-cloud encoders for DKSH policy observations.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "l1_policy_features = dksh_l1_features.node:main",
            "l2_policy_features = dksh_l1_features.node:main_l2",
        ],
    },
)
