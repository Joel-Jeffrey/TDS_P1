import os
import json

# Base directory containing the markdown files
docs_dir = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/docs/'
base_path = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/'  # Base path for relative paths

if not os.path.exists(docs_dir):
    print(f"Directory not found: {docs_dir}")
    exit(1)

# Initialize a dictionary to store relative paths and their titles
index = {}

for root, _, files in os.walk(docs_dir):
    for filename in files:
        if filename.endswith('.md'):
            filepath = os.path.join(root, filename)  # Full path
            rel_path = os.path.relpath(filepath, base_path)  # Convert to relative path
            
            print(f"Processing: {rel_path}")  # Debug: Print relative path

            # Open the markdown file and find the first header line
            try:
                with open(filepath, 'r', encoding="utf-8") as file:
                    for line in file:
                        if line.startswith('#'):  # Check for header (line starts with '#')
                            title = line.lstrip('#').strip()  # Remove '#' and extra spaces
                            index[rel_path] = title  # Store relative path and title
                            print(f"Extracted title: {title}")  # Debug: Print extracted title
                            break  # Stop after finding the first header
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Write the index dictionary to the index.json file
output_file = os.path.join(docs_dir, "index.json")
with open(output_file, 'w', encoding="utf-8") as json_file:
    json.dump(index, json_file, indent=4)

print(f"Index file has been created at {output_file} with {len(index)} entries.")
