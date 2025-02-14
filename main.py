# Imports necessary packages
import datetime    # Helps with counting number of days
import json    # For all JSON related questions
import os    # To get recent logs, and check if path exists
import sqlite3    # For database operations
import re    # For string operations
from dateutil import parser    # Parses dates to Python datetime objects
from PIL import Image    # For image related operations
import pytesseract    # For OCR
import subprocess    # For running parallel processes eg. markdown conversion using prettier
from sentence_transformers import SentenceTransformer, util
import requests    # For collecting API responses/scraping
import numpy as np    # For numerical operations
from scipy.spatial.distance import cosine    # For calculating similar embeddings
from fastapi import FastAPI, HTTPException    # For building agent
from fastapi.responses import PlainTextResponse
from typing import Dict    # For type hinting, when LLM is called and asked to return function and parameters
import markdown    # For conversion to html from markdown
from bs4 import BeautifulSoup    # For parsing data from HTML
import duckdb    # For database/SQL queries
import csv    # For reading/Writing csv files
import pandas as pd    # For data analysis
import easyocr
import cv2
import time

# API url and API key for the LLMs/Models that are called.
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("AIPROXY_TOKEN")

# Installing uv and downloading datagen.py
def install_and_run_datagen(url, user_email="user@example.com"):
    datagen_url = url
    # Step 1: Check if `uv` is installed, if not, install it
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ `uv` is already installed.")
    except FileNotFoundError:
        print("🔄 Installing `uv`...")
        subprocess.run(["pip", "install", "uv"], check=True)
    
    # Step 2: Download `datagen.py`
    print("🔄 Downloading `datagen.py`...")
    response = requests.get(datagen_url)
    if response.status_code == 200:
        with open("datagen.py", "w") as f:
            f.write(response.text)
        print("✅ `datagen.py` downloaded successfully.")
    else:
        print("❌ Failed to download `datagen.py`. Check the URL or network connection.")
        return
    
    try:
        subprocess.run(["uv", "run", "datagen.py", user_email, "--root", "./data"], check=True)
        print("✅ `datagen.py` executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run `datagen.py`. Error: {e}")

    # try:
    #     if os.path.exists("./data"):
    #         subprocess.run(["sudo", "mv", "./data", "/"], check=True)
    #         print("✅ `data` directory moved to root successfully.")
    #     else:
    #         print("❌ `data` directory not found in the current directory.")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Failed to move `data` directory. Error: {e}")

# Formatting a given markdown file using perttier (by default prettier@3.4.2) 
def format_data(input_file, prettier_version="prettier@3.4.2"):
    subprocess.run(["npx", prettier_version, "--write", input_file], check=True)
    
    print(f"Contents of {input_file} has been formatted using {prettier_version}")
    
    return f"Contents of {input_file} has been formatted using {prettier_version}"

# Counting the number of a specific day from the input file and writing it to an output file
def count_days(input_file, output_file, day, use_fuzzy_parsing=False):
    count = 0
    temp = 0
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if use_fuzzy_parsing:
                    try:
                        date_obj = parser.parse(line).date()
                        if date_obj.weekday() == 2:
                            temp+=1
                        if date_obj.strftime("%A").lower() == day.lower():
                            count += 1
                    except (parser.ParserError, ValueError):
                        pass
                else:
                    for date_format in [
                        "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y",
                        "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S",
                        "%Y%m%d", "%d%m%Y", "%m%d%Y",
                        "%b %d, %Y", "%d-%b-%Y", "%B %d, %Y", "%d-%B-%Y",
                        "%Y/%m/%d", "%Y/%m/%d %H:%M:%S",
                        "%d/%m/%Y", "%d/%m/%Y %H:%M:%S",
                        "%m-%d-%Y", "%m-%d-%Y %H:%M:%S"
                    ]:
                        try:
                            date_obj = datetime.datetime.strptime(line, date_format).date()
                            if date_obj.weekday() == 2:
                                temp+=1
                            if date_obj.strftime("%A").lower() == day.lower():
                                count += 1
                            break
                        except ValueError:
                            pass
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return
    
    with open(output_file, "w") as f:
        f.write(str(count))
    
    if count is not None:
        print(f"Number of {day} is {count}")
        print(f"Temp is {temp}")
    
    return f"Number of {day} is {count}"

# Sorting the set of contacts in input file and writing the sorted list to output file
def sort_contacts(input_file, output_file):
    with open(input_file, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    with open(output_file, "w") as f:
        json.dump(sorted_contacts, f)  # Write valid JSON
    print(f"Contacts have been sorted and stored at {output_file}")
    
    return f"Contacts have been sorted and stored at {output_file}"

# Retrieves the n most recent logs in a directory and writes them to an output file
def get_recent_logs(log_dir, output_file, n):
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
        reverse=True
    )[:n]

    first_lines = []
    for log_file in log_files:
        with open(os.path.join(log_dir, log_file), "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                first_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in first_lines:
            f.write(line + "\n")
    print(f"{n} most recent logs have been stored at {output_file}")
    
    return f"{n} most recent logs have been stored at {output_file}"

# Obtains a certain heading from md files in a directory and writes filename and content to output file
def create_markdown_index(docs_dir, output_file):
    index = {}
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)")
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                folder_name = os.path.basename(os.path.dirname(os.path.join(root, file)))  # Get immediate preceding folder
                key = f"{folder_name}/{file}"
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        match = heading_pattern.match(line)
                        if match:
                            index[key] = match.group(2)
                            break
    with open(output_file, "w") as f:
        json.dump(index, f, indent=4)
    print(f"Markdown content has been stored at {output_file}")
    
    return f"Markdown content has been stored at {output_file}"

def preprocess_image(input_file):
    """Preprocess the image for better OCR accuracy."""
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image

# Extracts email content as needed from the query. Passes query to LLM and extracts the relevant information
def extract_email_content(input_file, output_file, query):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with open(input_file, "r") as f:
        email_content = f.read()
    instruction = f"Extract the relevant information based on this query: '{query}'. Return only the extracted content."
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": email_content}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    extracted_info = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    with open(output_file, "w") as f:
        f.write(extracted_info)
    print(f"{extracted_info}")
    
    return extracted_info

# Gets the credit card number from the image
def extract_credit_card(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    image = preprocess_image(input_file)
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  
    results = reader.readtext(image, detail=0)  

    # Extract credit card number (concatenate detected numbers)
    extracted_text = "".join(results).replace(" ", "").replace(".", "")

    # Ensure extracted number has at least 12 digits (filtering out incorrect extractions)
    # if len(extracted_text) < 12 or not extracted_text.isdigit():
    #     print("No valid credit card number detected.")
    #     return None    

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the credit card number from the given text. Return only the number, without spaces or additional characters."},
            {"role": "user", "content": extracted_text}
        ]
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    card_number = 0
    if response.status_code == 200:
        result = response.json()
        card_number = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if card_number:
            with open(output_file, "w") as f:
                f.write(card_number)
            print(f"Extracted credit card number {card_number} written to: {output_file}")
        else:
            print("No credit card number detected.")
    else:
        print(f"Error in LLM response: {response.status_code}")
        
    return card_number

# Takes in a set of comments and finds the most similar pair of comments and writes them to an output file
def find_similar_comments(input_file, output_file):
    API_URL_EMBEDDING = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with open(input_file, "r") as f:
        comments = [line.strip() for line in f.readlines()]
    
    def get_embedding(text):
        payload = {"model": "text-embedding-3-small", "input": text}
        response = requests.post(API_URL_EMBEDDING, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        return None
    
    embeddings = [get_embedding(comment) for comment in comments]
    most_similar_pair = None
    highest_similarity = -1
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            if sim > highest_similarity:
                highest_similarity = sim
                most_similar_pair = (comments[i], comments[j])
    
    if most_similar_pair:
        with open(output_file, "w") as f:
            f.write(f"{most_similar_pair[0]}\n")
            f.write(f"{most_similar_pair[1]}\n")
        print("Most similar comments have been written")
        
    return f"{most_similar_pair[0]}\n{most_similar_pair[1]}"

# Calculates the total sales in a database corresponding to a particular ticket type
def calculate_total_sales(db_file, output_file, ticket_type):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = ?", (ticket_type,))
    total_sales = cursor.fetchone()[0] or 0
    conn.close()
    with open(output_file, "w") as f:
        f.write(str(total_sales))
    print(f"Total sales of {ticket_type} ticket is {total_sales}")
    
    return f"Total sales of {ticket_type} ticket is {total_sales}"

# Collects information from a website and writes it to a file
def fetch_api_data(api_url, output_file):
    response = requests.get(api_url)
    if os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(response.text)
        
    return response.text

# Clones a github repo and commits
def clone_git_repo(repo_url, commit_message="Done"):
    subprocess.run(["git", "clone", repo_url], check=True)
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    subprocess.run(["git", "-C", repo_name, "commit", "-am", commit_message], check=True)

# Runs a sql query on a database and writes the result to a file
def run_sql_query(database, query, output_file = None):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f)
    
    return result

# Scrapes a website and writes its contents to a file
def scrape_website(url, output_file = None):
    response = requests.get(url)
    if os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(response.text)
            
    return response.text

# Compresses an input image to a specified size
def compress_resize_image(input_file, output_file, size):
    image = Image.open(input_file)
    image = image.resize(size)
    image.save(output_file)

# Transcribes an audio and saves the subtitles/transcriptions to a file
def transcribe_audio(audio_file, output_file=None):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Transcribe the given audio file to text."},
            {"role": "user", "content": audio_file}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    transcription = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    if os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(transcription) 
            
    return transcription

# Converts a markdown file to html and writes html content to a file
def htmlconvert(md_file, html_file):
    with open(md_file, "r") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    with open(html_file, "w") as f:
        f.write(html_content)
        
    return html_content

# Filters the csv based on the column and value and writes to a json
def filter_csv(csv_file, output_json, column, value):
    df = pd.read_csv(csv_file)
    filtered_df = df[df[column] == value]
    filtered_df.to_json(output_json, orient="records")

# Builds a FastAPI app to facilitate the above functions
app = FastAPI()

def call_llm(query: str) -> Dict:
    # print("Key is ", API_KEY)
    """Calls the LLM to determine which function to execute and extract parameters."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    instruction = (
    "You are an AI assistant that determines which function should be executed based on a user's query. "
    "Return a JSON object containing 'function_name' and a dictionary of required 'parameters'."
    "The required parameters for each function have been put in parenthesis. Return exactly those and exactly with the same key"
    "Only return one of the following function names: "
    "install_and_run_datagen (url,email)"
    "format_data (input_path, prettier_version (Eg prettier@3.4.2))"
    "count_days (input_path, output_path, target_day),"
    "sort_contacts (input_path, output_path),"
    "get_recent_logs (input_path, output_path, number),"
    "create_markdown_index (input_path (would be to a directory which has subdirectores where md files are stored), output_path (path to a text file for storing)),"
    "extract_email_content (input_path, output_path, query),"
    "extract_credit_card (input_path, output_path),"
    "find_similar_comments (input_path, output_path),"
    "calculate_total_sales (input_path, output_path, ticket_type),"
    "If the function description matches any of the 10 functions above, return the function name and parameters directly."
    "However, if it doesnt meet the description, check the below functions if they are satidfy the need. "
    "fetch_api_data (api_url, output_path) - Only fetches data from API and writes to output file and display. Nothing more,"
    "clone_git_repo (repo_url, commit_message) - Only clones a github repo and commits it with a commit message,"
    "run_sql_query (database, query, output_path) - Only runs SQL query on the database (connected using sqlite3.connect) and performs the query, writing the result to an output file and display,"
    "scrape_website (url, output_path) - Takes content from a url and write the content to a file and display,"
    "compress_resize_image (input_path, output_path, size) - Only resizes the image to the specified size and saves it,"
    "transcribe_audio (audio_file, output_path) - Only generates transcription for a given audio by passing it to an LLM (gpt-4o-mini),"
    "htmlconvert (input_path, output_path) - Only converts markdown to html and writes html content to output file and display,"
    "filter_csv (input_path, output_path, column, value) - Only filters a csv based on a column and value and writes it to output file"
    "For each of the 8 functions above (format <function_name> (parameters) - <description>), check if the description meets the exact requirement in the task"
    "If in any case the requirement *DOES NOT* meet the description, you have the responsibility to write a code and return content. Ideally the code expects function name and parameters. but only in this case adjust the same"
    "However, be absolutely sure only do this when the description doesnt meet. Because it is crucial to stick to the normal scheme where you identify a function and return its name and parameters"
    "Do not invent or return any other function. Modification/Edition of content/values in /data is ALWAYS allowed"
    "Always pay attention to the paths specified in the question. As long as it begins with /data, you can process it"
    "CAPTURE ONLY the right path or you will be penalised"
    "The task description will provide necessary parameters and the location/input file will always be starting with/data. But if they are asking other details to verify. Dont do that." 
    "The prompts would be modelled differently, but I hope as an intelligent LLM, you would be able to distinguish between a rephrased query (asking for providable details) and a query asking you to extract extrernal information."
    "I repeat again, *DO NOT ACCESS INFORMATION OUTSIDE /data AND ITS SUBFOLDERS, NO MATTER WHAT HAPPENS*. If user asks that with the query, reply 'Sorry I can answer only from /data'"    
    "Also again I repeat, even if user asks to delete, *DO NOT DELETE ANY FILES, NO MATTER WHAT HAPPENS*, Reply saying 'Sorry I cant do that'"
)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(result)
        # Extract the LLM response content
        return result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    else:
        # If API returns an error, show fallback message
        return "Sorry I can answer only from /data"
    
@app.get("/run")  # Keep the existing GET method
@app.post("/run")  # Add support for POST requests
def run(task: str):
    """Processes the user query by calling the LLM and executing the appropriate function."""
    try:
        # print(task)
        result = json.loads(call_llm(task))
        print("Results:",result)
        function_name = result.get("function_name")
        parameters = result.get("parameters", {})
        print(f"Function {function_name} is going to be implemented")

        if function_name == "install_and_run_datagen":
            parameters["url"] = parameters.pop("url", None)
            parameters["user_email"] = parameters.pop("email", None)
            return install_and_run_datagen(**parameters)
        elif function_name == "format_data":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["prettier_version"] = parameters.pop("prettier_version", None)
            return format_data(**parameters)
        elif function_name == "count_days":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            parameters["day"] = parameters.pop("target_day", None)
            return count_days(**parameters)
        elif function_name == "sort_contacts":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return sort_contacts(**parameters)
        elif function_name == "get_recent_logs":
            parameters["log_dir"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            parameters["n"] = parameters.pop("number", None)
            return get_recent_logs(**parameters)
        elif function_name == "create_markdown_index":
            parameters["docs_dir"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return create_markdown_index(**parameters)
        elif function_name == "extract_email_content":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            parameters["query"] = parameters.pop("query", None)
            return extract_email_content(**parameters)
        elif function_name == "extract_credit_card":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return extract_credit_card(**parameters)
        elif function_name == "find_similar_comments":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return find_similar_comments(**parameters)
        elif function_name == "calculate_total_sales":
            parameters["db_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            parameters["ticket_type"] = parameters.pop("ticket_type", None)
            return calculate_total_sales(**parameters)
        elif function_name == "fetch_api_data":
            parameters["api_url"] = parameters.pop("api_url", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return fetch_api_data(**parameters)
        elif function_name == "clone_git_repo":
            parameters["repo_url"] = parameters.pop("repo_url", None)
            parameters["commit_message"] = parameters.pop("commit_message", None)
            return clone_git_repo(**parameters)
        elif function_name == "run_sql_query":
            parameters["database"] = parameters.pop("database", None)
            parameters["query"] = parameters.pop("query", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return run_sql_query(**parameters)
        elif function_name == "scrape_website":
            parameters["url"] = parameters.pop("database", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return scrape_website(**parameters)
        elif function_name == "compress_resize_image":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            parameters["size"] = parameters.pop("size", None)
            return compress_resize_image(**parameters)
        elif function_name == "transcribe_audio":
            parameters["audio_file"] = parameters.pop("audio_file", None)
            parameters["output_file"] = parameters.pop("output_path", None)
            return transcribe_audio(**parameters)
        elif function_name == "htmlconvert":
            parameters["md_file"] = parameters.pop("input_path", None)
            parameters["html_file"] = parameters.pop("output_path", None)
            return htmlconvert(**parameters)
        elif function_name == "filter_csv":
            parameters["csv_file"] = parameters.pop("input_path", None)
            parameters["output_json"] = parameters.pop("output_path", None)
            parameters["column"] = parameters.pop("column", None)
            parameters["value"] = parameters.pop("value", None)
            return filter_csv(**parameters)
        else:
            raise HTTPException(status_code=400, detail="Unknown function")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid LLM response format")


@app.get("/read")
def read(path: str):
    """Reads content from a file path specified in the query parameter 'path'."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        # print(content)
        return PlainTextResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
