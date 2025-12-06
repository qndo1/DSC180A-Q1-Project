from setuptools import setup, find_packages

setup(
    name="dsc180a_q1_project",        # Package name
    version="0.1.0",                  # Version number
    packages=find_packages(where="src"),  # Automatically find all modules in src/
    package_dir={"": "src"},          # Tell setuptools that src is the root
    install_requires=[                # Dependencies
        "requests==2.30.0",
        "beautifulsoup4==4.12.2",
        "numpy==1.25.2",
        "pandas==2.1.1",
        "matplotlib==3.8.1",
        "plotly==6.1.2",
        "scipy==1.11.3",
        "scikit-learn==1.3.1",
        "POT==0.9.6",  # Python Optimal Transport library
        "ipykernel==6.25.2",
    ],
    entry_points={                     #command line scripts
        "console_scripts": [
        "download_csv=scripts.download_and_convert:main",
        "graphics=scripts.graphics:main",
        "multi_frame_ot=scripts.multi_frame_ot:main",
        ],
    },
)
