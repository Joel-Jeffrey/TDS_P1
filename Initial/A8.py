import os
import requests
import pytesseract
from PIL import Image

# API details
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

# File paths
image_file = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/credit_card.png"
output_file = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/credit-card.txt"

# Check if image file exists
if not os.path.exists(image_file):
    print(f"File not found: {image_file}")
    exit(1)

# Extract text using OCR
image = Image.open(image_file)
extracted_text = pytesseract.image_to_string(image)
extracted_text = extracted_text.replace(" ", "").replace(".", "")
print("Extracted text:", extracted_text)


# Prepare API request
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Extract the credit card number from the given text. Return only the number, without spaces or additional characters."},
        {"role": "user", "content": extracted_text}
    ]
}

# Call GPT-4o-mini API
response = requests.post(API_URL, json=payload, headers=headers)

# Handle API response
if response.status_code == 200:
    card_number = response.json()["choices"][0]["message"]["content"].strip()

    # Write the extracted credit card number to the output file
    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write(card_number + "\n")
    print(f"Credit card number {card_number} extracted and saved to {output_file}")

else:
    print(f"API Error: {response.status_code} - {response.text}")
    exit(1)
