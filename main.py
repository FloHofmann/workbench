import subprocess
import os


def main():
    print("Hello from workbench!")


if __name__ == "__main__":
    main()
    # this is just a small test to call the rustsorter from python
    rust_binary = os.path.join(
        "rustsorter", "target", "debug", "rustsorter.exe")
    subprocess.run(rust_binary)
