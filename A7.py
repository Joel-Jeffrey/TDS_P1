import os
import requests

# API details
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    print("Error: API_KEY environment variable is not set.")
    exit(1)
    
# File paths
email_file = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/email.txt"
output_file = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/email-sender.txt"

# Check if email file exists
if not os.path.exists(email_file):
    print(f"File not found: {email_file}")
    exit(1)

# Read email content
with open(email_file, "r", encoding="utf-8") as file:
    email_content = file.read()

# Prepare API request
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Extract the sender's email address from the following email and return only the email address."},
        {"role": "user", "content": email_content}
    ]
}

# Call GPT-4o-mini API
response = requests.post(API_URL, json=payload, headers=headers)

# Handle API response
if response.status_code == 200:
    sender_email = response.json()["choices"][0]["message"]["content"].strip()

    # Write the extracted email to the output file
    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write(sender_email + "\n")
    print(f"Sender's email is {sender_email} which has been extracted and saved to {output_file}")

else:
    print(f"API Error: {response.status_code} - {response.text}")
    exit(1)
