import os
from glob import glob
from setuptools import setup, find_packages


package_name = "ls2n_drone_direct_motor_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name),
            glob("launch/onboard_app/*.launch.py")
        )
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julien Stauder",
    maintainer_email="julien.stauder@eleves.ec-nantes.fr",
    description="Allow direct motor control",
    license="Apache 2.0",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            'direct_motor_control = ls2n_drone_direct_motor_control.direct_motor_control:main'
        ],
    },
)
