import os
import subprocess
from datetime import datetime

def run_cmd(cmd: str):
    """Run a shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main():
    # 1) Add all changes
    run_cmd("git add .")

    # 2) Generate commit message with timestamp
    msg = f"update: sync latest changes ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    run_cmd(f'git commit -m "feat: add UNet+VAE with train/val/test split, logging, and organized outputs"')

    # 3) Push to remote
    run_cmd("git push origin main")

if __name__ == "__main__":
    main()
