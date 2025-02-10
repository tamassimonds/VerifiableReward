import os
import json
import datasets

if __name__ == '__main__':
    # Path to the directory containing your JSON files
    data_dir = 'variant_results'  # Adjust this path as needed

    samples = []
    # Define the integration-specific instruction.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before solving, don't include any constants of integration."
    )
    
    total_questions = 0  # Initialize a counter for the total number of questions

    # Loop over every JSON file in the specified directory.
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Process each list of variants in the JSON file.
            for variant_list in data:
                for idx, variant in enumerate(variant_list):
                    # The integration expression is stored in the "variant" field.
                    integration_expr = variant.get("variant", "").strip()
                    if not integration_expr:
                        continue  # Skip if there is no variant text.
                    
                    # Check if verification_passed is True
                    if not variant.get("verification_passed", False):
                        continue  # Skip if verification_passed is not True
                    
                    # Build the prompt by combining the integration expression with the instruction.
                    prompt_content = f"{integration_expr}\n{instruction_following}"
                    
                    # Use the 'solution' field as the ground truth.
                    ground_truth = variant.get("solution", "")
                    
                    # Build a sample dictionary for this variant.
                    sample = {
                        "data_source": "integration_variants",
                        "prompt": [{
                            "role": "user",
                            "content": prompt_content
                        }],
                        "ability": "integration",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": ground_truth
                        },
                        "extra_info": {
                            "original": variant.get("original", ""),
                            "requested_difficulty": variant.get("requested_difficulty", ""),
                            "verification_passed": variant.get("verification_passed", False),
                            "reasoning": variant.get("reasoning", ""),
                            "timestamp": variant.get("timestamp", "")
                        }
                    }
                    samples.append(sample)
                    total_questions += 1  # Increment the counter for each valid question

    # Create a Hugging Face dataset from the list of samples.
    dataset = datasets.Dataset.from_list(samples)
    
    # Split the dataset into train and test sets (using a small test fraction).
    split_datasets = dataset.train_test_split(test_size=0.1)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']
    
    # Define a local directory to save the output files.
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Optionally limit the test dataset to at most 1024 samples.
    test_dataset = test_dataset.select(range(min(len(test_dataset), 1024)))
    
    # Save the train and test datasets.
    train_dataset.to_parquet(os.path.join(output_dir, 'integration_train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir, 'integration_test.parquet'))
    
    print(f"Integration dataset created successfully with {total_questions} questions.")
