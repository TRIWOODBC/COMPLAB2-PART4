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
    run_cmd(f'git commit -m "feat: upgrade VAE to 6-layer architecture (256x256, latent_dim=64, GroupNorm, KL warm-up, grad clipping)"')

    # 3) Push to remote
    run_cmd("git push origin main")

if __name__ == "__main__":
    main()
