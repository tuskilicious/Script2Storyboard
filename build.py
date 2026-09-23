import shutil
import subprocess
import sys


def build_executable():
    # Clean up previous builds
    shutil.rmtree('dist', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)

    # Run PyInstaller through this interpreter, so it doesn't need to be on PATH,
    # and stop on failure instead of reporting success.
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', 'Script2Storyboard.spec'], check=True)

    print(r"Build completed! Run dist\Script2Storyboard\Script2Storyboard.exe --script path\to\script.txt")


if __name__ == "__main__":
    build_executable()
