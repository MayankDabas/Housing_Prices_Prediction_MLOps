"""
This script generates Kubernetes job YAML files for hyperparameter tuning 
of a RandomForest model. It creates separate job YAML files for each combination 
of hyperparameters and saves them in the specified output directory.

Hyperparameters tuned:
1. n_estimators
2. max_depth
3. min_samples_split
4. min_samples_leaf

The generated YAML files can be used to run parallel jobs on a Kubernetes cluster.
"""

import itertools
import os
import yaml

n_estimators = [50, 100]
max_depth = [5, 10]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]

output_dir = "generated_jobs"
os.makedirs(output_dir, exist_ok=True)

def create_job_yaml(name, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "house-price-container",
                        "image": "house-price-model:v1",
                        "command": ["python", "house_price_prediction.py"],
                        "args": [
                            f"--n_estimators={n_estimators}",
                            f"--max_depth={max_depth}",
                            f"--min_samples_split={min_samples_split}",
                            f"--min_samples_leaf={min_samples_leaf}"
                        ],
                    }],
                    "restartPolicy": "Never"
                }
            },
            "backoffLimit": 4
        }
    }
    return job_manifest

combinations = list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf))

for idx, (n_est, depth, split, leaf) in enumerate(combinations):
    job_name = f"hp-tuning-jobs-{idx}"
    job_yaml = create_job_yaml(job_name, n_est, depth, split, leaf)

    yaml_file = os.path.join(output_dir, f"{job_name}.yaml")
    with open(yaml_file, 'w') as f:
        yaml.dump(job_yaml, f, default_flow_style=False)
    
    print(f"Generated {yaml_file}")

print(f"All {len(combinations)} job YAML files generated in '{output_dir}' directory.")
