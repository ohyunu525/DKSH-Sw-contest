from setuptools import find_packages, setup

setup(
    name="dksh_remote_navigation", version="0.1.0", packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/dksh_remote_navigation"]),
                ("share/dksh_remote_navigation", ["package.xml"])],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="DKSH", maintainer_email="dksh@example.invalid",
    description="Sensor-free Unity waypoint gateway with optional Nav2 simulation adapter.",
    license="MIT", entry_points={"console_scripts": ["waypoint_gateway = dksh_remote_navigation.server:main"]},
)
