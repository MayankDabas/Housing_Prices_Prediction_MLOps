# import os
# import json

# # Directory where results files are stored
# results_dir = './results'  # Adjust to the actual path

# # List to store the results
# results_list = []

# # Iterate over the results files
# for file_name in os.listdir(results_dir):
#     if file_name.endswith('.txt'):
#         file_path = os.path.join(results_dir, file_name)

#         try:
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#                 # Initialize variables to store extracted data
#                 hyperparams = None
#                 r2_score = None

#                 # Parse each line in the file
#                 for line in lines:
#                     line = line.strip()
#                     if line.startswith('Hyperparameters:'):
#                         # Extract hyperparameters
#                         hyperparams = json.loads(line.split('Hyperparameters: ')[-1])
#                     elif 'R² score on Validation Set:' in line or 'R2 score on Vaildation Set:' in line:
#                         # Extract R² score
#                         r2_score = float(line.split(': ')[-1])

#                 # If both hyperparameters and R² score are found, store them
#                 if hyperparams is not None and r2_score is not None:
#                     results_list.append({
#                         'hyperparameters': hyperparams,
#                         'r2_score': r2_score
#                     })

#         except Exception as e:
#             print(f"Error processing file {file_name}: {e}")

# # Sort results by R² score in descending order
# sorted_results = sorted(results_list, key=lambda x: x['r2_score'], reverse=True)

# # Check if results are available
# if not sorted_results:
#     print("No valid results found.")
# else:
#     # Get the best hyperparameters
#     best_result = sorted_results[0]

#     # Print the best hyperparameters and R² score
#     print(f"Best Hyperparameters: {json.dumps(best_result['hyperparameters'])}")
#     print(f"Best R² score on Validation Set: {best_result['r2_score']}")


import os
import json

# Directory where results.txt files are stored
results_dir = './results'  # Adjust to the actual path

# List to store the results
results_list = []

# Iterate over the results files
for file_name in os.listdir(results_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(results_dir, file_name)

        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Initialize variables to store extracted data
            hyperparams = None
            r2_score = None

            # Parse each line in the file
            for line in lines:
                line = line.strip()
                if line.startswith('Hyperparameters:'):
                    # Extract hyperparameters
                    hyperparams = json.loads(line.split('Hyperparameters: ')[-1])
                elif 'R² score on Validation Set:' in line or 'R2 score on Vaildation Set:' in line:
                    # Extract R² score
                    r2_score = float(line.split(': ')[-1])

            # If both hyperparameters and R² score are found, store them
            if hyperparams is not None and r2_score is not None:
                results_list.append({
                    'hyperparameters': hyperparams,
                    'r2_score': r2_score
                })

# Sort results by R² score in descending order
sorted_results = sorted(results_list, key=lambda x: x['r2_score'], reverse=True)

# Check if results are available
if not sorted_results:
    print("No valid results found.")
else:
    # Get the best hyperparameters
    best_result = sorted_results[0]
    best_hyperparams = best_result['hyperparameters']

    # Save the best hyperparameters to a JSON file
    with open('best_hyperparams.json', 'w') as file:
        json.dump(best_hyperparams, file)

    # Print the best hyperparameters and R² score
    print(f"Best Hyperparameters: {json.dumps(best_hyperparams)}")
    print(f"Best R² score on Validation Set: {best_result['r2_score']}")
    print("Best hyperparameters saved to 'best_hyperparams.json'")
