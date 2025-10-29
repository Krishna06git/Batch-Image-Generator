import argparse
import os
import sys
import subprocess


def sh(cmd: str) -> int:
    print(">", cmd)
    return subprocess.check_call(cmd, shell=True)


def ensure_deps() -> None:
    sh(f"{sys.executable} -m pip install --upgrade pip")
    sh(f"{sys.executable} -m pip install -r requirements.txt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="Expose public Gradio link")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--fast", action="store_true", help="Prefer fast defaults")
    args = p.parse_args()

    ensure_deps()

    env = os.environ.copy()
    if args.fast:
        env["DEFAULT_QUALITY_MODE"] = "Fast"

    flags = f"--server_port {args.port} --server_name 0.0.0.0"
    if args.share:
        flags += " --share"

    os.execle(sys.executable, sys.executable, "app.py", *flags.split(), env)


if __name__ == "__main__":
    main()