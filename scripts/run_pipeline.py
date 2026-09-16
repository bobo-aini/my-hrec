"""Run the four explicit pipeline commands in dependency order."""
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config",required=True)
args = parser.parse_args()
for command in ("audit","pretrain","build-memory","train","evaluate"):
    subprocess.run([sys.executable,"-m","hrec",command,"--config",args.config],check=True)
