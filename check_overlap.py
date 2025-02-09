import json


from base_datasets.test_questions import TEST_QUESTIONS

# Load the TEST_QUESTIONS list

# Read the contents of the file
with open('output/integration_train.json', 'r') as f:
    file_content = f.read()

# Check for contamination
contaminated_questions = []
for question in TEST_QUESTIONS:
    if question in file_content:
        contaminated_questions.append(question)

# Print the contaminated questions
if contaminated_questions:
    print("The following questions are present in output/integration_train.json:")
    for question in contaminated_questions:
        print(f"- {question}")
else:
    print("No contamination found.")

print(len(contaminated_questions))