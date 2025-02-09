import json
import os

# Open the JSON file
with open('output/integration_train.json', 'r') as f:
    data = json.load(f)

# Create a set to store unique original integrals
unique_originals = set()

# Loop over the data
for item in data:
    original = item['extra_info']['original']
    unique_originals.add(original)

# Print the unique original integrals
for original in unique_originals:
    print(original)
print(len(unique_originals))