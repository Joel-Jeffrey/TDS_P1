import requests
import numpy as np
import json
import os
from scipy.spatial.distance import cosine

# Your API key
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
API_KEY = os.getenv("API_KEY")

# Load comments from the file
comments_file = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/comments.txt'
comments = []

with open(comments_file, 'r') as file:
    comments = [line.strip() for line in file.readlines()]

# Function to get embeddings from the API
def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "text-embedding-3-small",  # Use the small text embedding model
        "input": text
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code} from the API.")
        print("Response content:", response.text)
        return None

    # Parse the response
    try:
        response_data = response.json()
        # Check if 'data' is in the response
        if 'data' not in response_data:
            print("Error: 'data' key not found in the API response.")
            print("Response content:", response_data)
            return None
        return response_data['data'][0]['embedding']
    except json.JSONDecodeError:
        print("Error: Failed to parse the API response as JSON.")
        return None

# Get embeddings for all comments
embeddings = []
for comment in comments:
    embedding = get_embedding(comment)
    embeddings.append(embedding)

# Calculate cosine similarity between all pairs
most_similar_pair = None
highest_similarity = -1  # Cosine similarity ranges from -1 to 1

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = 1 - cosine(embeddings[i], embeddings[j])  # Cosine similarity is 1 - distance
        if sim > highest_similarity:
            highest_similarity = sim
            most_similar_pair = (comments[i], comments[j])

# Write the most similar comments to the output file
output_file = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/comments-similar.txt'
if most_similar_pair:
    with open(output_file, 'w') as file:
        file.write(f"{most_similar_pair[0]}\n")
        file.write(f"{most_similar_pair[1]}\n")
        # file.write(f"Similarity Score: {highest_similarity}\n")

print("Most similar comments have been written")
