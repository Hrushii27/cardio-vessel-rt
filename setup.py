from setuptools import setup, find_packages

setup(
    name="cardio_rt",
    version="1.0.0",
    description="Real-Time Cardiology X-ray & Cine Coronary Angiography Image Processing System",
    author="Cardiology Real-Time Imaging Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "tifffile>=2023.1.1",
        "pydicom>=2.4.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "psutil>=5.9.0",
    ],
)
