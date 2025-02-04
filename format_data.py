import os
import json
import datasets

if __name__ == '__main__':
    # Path to the directory containing your JSON files
    data_dir = 'dataset'  # Adjust this path as needed

    samples = []
    # Define the integration-specific instruction.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before answering."
    )
    
    # Loop over every JSON file in the specified directory.
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract the question ID (if available)
            question_id = data.get("question_id", "")
            
            # Process each variant in the JSON file.
            for idx, variant in enumerate(data.get("variants", [])):
                # The integration expression is stored in the "variant" field.
                integration_expr = variant.get("variant", "").strip()
                if not integration_expr:
                    continue  # Skip if there is no variant text.
                
                # Build the prompt by combining the integration expression with the instruction.
                prompt_content = f"{integration_expr}\n{instruction_following}"
                
                # Use the 'evaluations' field as the ground truth.
                # (It should be a dictionary mapping evaluation point strings to numerical values.)
                ground_truth = variant.get("solution", "")
                print(ground_truth)
                
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
                        "question_id": question_id,
                        "variant_index": idx,
                        "requested_difficulty": variant.get("requested_difficulty", ""),
                        "verification_passed": variant.get("verification_passed", False),
                        "reasoning": variant.get("reasoning", ""),
                        "timestamp": variant.get("timestamp", "")
                    }
                }
                samples.append(sample)
    
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
    
    print("Integration dataset created successfully.")
