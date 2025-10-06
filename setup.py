import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

ROOT = Path(__file__).resolve().parent
HANABI_DIR = ROOT / "third_party" / "hanabi"


def _build_hanabi() -> None:
    """Build the Hanabi Learning Environment C++ extension."""
    if not HANABI_DIR.exists():
        raise RuntimeError(f"Hanabi directory not found at {HANABI_DIR}")

    build_dir = HANABI_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["cmake", ".."], cwd=str(build_dir))
    subprocess.check_call(["make", "-j"], cwd=str(build_dir))


class PostInstallCommand(install):
    def run(self) -> None:
        super().run()
        _build_hanabi()


class PostDevelopCommand(develop):
    def run(self) -> None:
        super().run()
        _build_hanabi()


setup(
    name="hanabi-learning-env",
    version="0.1.0",
    description="Helpers for fetching and building the Hanabi Learning Environment.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
    install_requires=[
        "cffi>=1.16.0",
        "cython>=3.0.0",
        "protobuf>=4.25.0",
    ],
)
