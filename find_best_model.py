"""
This script processes results from multiple model training runs, extracts 
the hyperparameters and evaluation metrics, and identifies the best-performing 
model based on the highest R² score.

The script assumes that each result is stored as a text file in the specified 
directory (`results_dir`). Each file should contain the hyperparameters and 
R² score obtained on the validation set.

Key Features:
- Iterates through result files to extract hyperparameters and R² scores.
- Sorts models based on their R² scores in descending order.
- Saves the best hyperparameters to a JSON file.
"""

import os
import json

results_dir = './results'

results_list = []

for file_name in os.listdir(results_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(results_dir, file_name)

        with open(file_path, 'r') as file:
            lines = file.readlines()

            hyperparams = None
            r2_score = None

            for line in lines:
                line = line.strip()
                if line.startswith('Hyperparameters:'):
                    hyperparams = json.loads(line.split('Hyperparameters: ')[-1])
                elif 'R2 score on Validation Set:' in line or 'R2 score on Vaildation Set:' in line:
                    r2_score = float(line.split(': ')[-1])

            if hyperparams is not None and r2_score is not None:
                results_list.append({
                    'hyperparameters': hyperparams,
                    'r2_score': r2_score
                })
sorted_results = sorted(results_list, key=lambda x: x['r2_score'], reverse=True)

if not sorted_results:
    print("No valid results found.")
else:

    best_result = sorted_results[0]
    best_hyperparams = best_result['hyperparameters']

    with open('best_hyperparams.json', 'w') as file:
        json.dump(best_hyperparams, file)

    print(f"Best Hyperparameters: {json.dumps(best_hyperparams)}")
    print(f"Best R2 score on Validation Set: {best_result['r2_score']}")
    print("Best hyperparameters saved to 'best_hyperparams.json'")
