from setuptools import setup, find_packages

setup(
    name="dsc180a_q1_project",        # Package name
    version="0.1.0",                  # Version number
    packages=find_packages(where="src"),  # Automatically find all modules in src/
    package_dir={"": "src"},          # Tell setuptools that src is the root
    install_requires=[                # Dependencies
        "numpy",
        "pandas",
        "plotly",
        "matplotlib",
        "scipy",
        "ot",                          # Python Optimal Transport
    ],
    entry_points={                     # Optional: command line scripts
        "console_scripts": [
            "run-script=scripts.script:main",  # if your script has a main() function
        ],
    },
)
