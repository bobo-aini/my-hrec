"""Five independent runs and mean/std aggregation; no test-set model selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config",required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
base = Path(cfg.get("run_dir","runs/hrec"))
configs = base/"seed_configs"
configs.mkdir(parents=True,exist_ok=True)
all_metrics = []
for seed in (1111,2222,3333,4444,5555):
    run = {**cfg,"seed":seed,"run_dir":str(base/f"seed_{seed}")}
    path = configs/f"{seed}.json"
    path.write_text(json.dumps(run,indent=2)+"\n",encoding="utf-8")
    subprocess.run([sys.executable,"scripts/run_pipeline.py","--config",str(path)],check=True)
    all_metrics.append(json.loads((Path(run["run_dir"])/"test_all_metrics.json").read_text()))
summary = {}
for key in all_metrics[0]:
    values = [r[key] for r in all_metrics]
    if all(isinstance(v,(int,float)) for v in values):
        summary[key] = {"mean":float(np.mean(values)),"sample_std":float(np.std(values,ddof=1))}
(base/"five_seed_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
