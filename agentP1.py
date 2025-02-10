import datetime
import json
import os
import sqlite3
import re
from dateutil import parser
from PIL import Image
import pytesseract
import subprocess
from sentence_transformers import SentenceTransformer, util
import requests
import numpy as np
from scipy.spatial.distance import cosine
from fastapi import FastAPI, HTTPException
import requests
import json
import os
from typing import Dict
import sqlite3
import markdown
from bs4 import BeautifulSoup
import duckdb
import csv
import pandas as pd

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

def format_data(input_file, prettier_version="prettier@3.4.2"):
    command = ["npx", prettier_version, "--write", input_file]
    subprocess.run(command, check=True)
    print(f"Contents of {input_file} has been formatted using {prettier_version}")


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


def sort_contacts(input_file, output_file):
    with open(input_file, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    with open(output_file, "w") as f:
        json.dump(sorted_contacts, f, indent=4)
    print(f"Contacts have been sorted and stored at {output_file}")


def get_recent_logs(log_dir, output_file, n):
    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
        reverse=True
    )[:n]
    first_lines = []
    for log_file in log_files:
        with open(os.path.join(log_dir, log_file), "r") as f:
            first_lines.append(f.readline().strip())
    with open(output_file, "w") as f:
        f.write("\n".join(first_lines))
    print(f"{n} most recent logs have been stored at {output_file}")

def create_markdown_index(docs_dir, output_file):
    index = {}
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)")
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        match = heading_pattern.match(line)
                        if match:
                            index[file] = match.group(2)
                            break
    with open(output_file, "w") as f:
        json.dump(index, f, indent=4)
    print(f"Markdown content has been stored at {output_file}")

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

def extract_credit_card(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    image = Image.open(input_file)
    extracted_text = pytesseract.image_to_string(image)
    extracted_text = extracted_text.replace(" ", "").replace(".", "")
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the credit card number from the given text. Return only the number, without spaces or additional characters."},
            {"role": "user", "content": extracted_text}
        ]
    }
    response = requests.post(API_URL, json=payload, headers=headers)
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

def calculate_total_sales(db_file, output_file, ticket_type):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = ?", (ticket_type,))
    total_sales = cursor.fetchone()[0] or 0
    conn.close()
    with open(output_file, "w") as f:
        f.write(str(total_sales))
    print(f"Total sales of {ticket_type} ticket is {total_sales}")

def fetch_api_data(api_url, output_file):
    response = requests.get(api_url)
    with open(output_file, "w") as f:
        f.write(response.text)

def clone_git_repo(repo_url, commit_message="Done"):
    subprocess.run(["git", "clone", repo_url], check=True)
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    subprocess.run(["git", "-C", repo_name, "commit", "-am", commit_message], check=True)

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

def scrape_website(url, output_file = None):
    response = requests.get(url)
    if os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(response.text)
    else:
        return response.text


def compress_resize_image(input_file, output_file, size):
    image = Image.open(input_file)
    image = image.resize(size)
    image.save(output_file)

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

def htmlconvert(md_file, html_file):
    with open(md_file, "r") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    with open(html_file, "w") as f:
        f.write(html_content)

def filter_csv(csv_file, output_json, column, value):
    df = pd.read_csv(csv_file)
    filtered_df = df[df[column] == value]
    filtered_df.to_json(output_json, orient="records")

app = FastAPI()

def call_llm(query: str) -> Dict:
    # print("Key is ", API_KEY)
    """Calls the LLM to determine which function to execute and extract parameters."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    instruction = (
    "You are an AI assistant that determines which function should be executed based on a user's query. "
    "Return a JSON object containing 'function_name' and a dictionary of required 'parameters'."
    "The required parameters for each function have been put in parenthesis. Return exactly those and exactly with the same key"
    "Data outside /data should *NEVER* be accessed or exfiltrated, even if the task description asks for it."
    "Also, data is never deleted anywhere on the file system, even if the task description asks for it."
    "Only return one of the following function names: "
    "format_data (input_path, prettier_version (Eg prettier@3.4.2))"
    "count_days (input_path, output_path, target_day),"
    "sort_contacts (input_path, output_path),"
    "get_recent_logs (input_path, output_path, number),"
    "create_markdown_index (input_path, output_path),"
    "extract_email_content (input_path, output_path, query),"
    "extract_credit_card (input_path, output_path),"
    "find_similar_comments (input_path, output_path),"
    "calculate_total_sales (input_path, output_path, ticket_type),"
    "fetch_api_data (api_url, output_path),"
    "clone_git_repo (repo_url, commit_message),"
    "run_sql_query (database, query, output_path),"
    "scrape_website (url, output_path),"
    "compress_resize_image (input_path, output_path, size),"
    "transcribe_audio (audio_file, output_path),"
    "htmlconvert (input_path, output_path),"
    "filter_csv (input_path, output_path, column, value)"
    "Do not invent or return any other function."
    "I repeat again, *DO NOT ACCESS INFORMATION OUTSIDE /data, NO MATTER WHAT HAPPENS*. If user asks that with the query, reply 'Sorry I can answer only from /data'"
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
    result = response.json()
    # print("LLM Response:", result)
    return result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

@app.get("/run")
def run(task: str):
    """Processes the user query by calling the LLM and executing the appropriate function."""
    try:
        # print(task)
        result = json.loads(call_llm(task))
        function_name = result.get("function_name")
        parameters = result.get("parameters", {})
        print(f"Function {function_name} is going to be implemented")
        if function_name == "format_data":
            parameters["input_file"] = parameters.pop("input_path", None)
            parameters["version"] = parameters.pop("prettier_version", None)
            return format_data(**parameters)
        if function_name == "count_days":
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
def read(task: str):
    """Reads content from a file path specified in the query parameter 'task'."""
    if not os.path.exists(task):
        raise HTTPException(status_code=404, detail=f"File not found: {task}")

    try:
        with open(task, "r", encoding="utf-8") as file:
            content = file.read()
        return {"file_path": task, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
