import os
import subprocess
import sys

# Function to check and install uv if not already installed
def install_uv():
    try:
        import uv
    except ImportError:
        print("uv not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])

# os.environ["CONFIG_ROOT"] = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/data"  # Use any directory where you have write access

# Run the datagen.py script with the user's email as an argument
def run_datagen_script(user_email):
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    script_name = "datagen.py"

    # Download the script
    subprocess.run(["curl", "-O", url])

    # Run the script with the user's email as an argument
    subprocess.run(["python", script_name, user_email])

if __name__ == "__main__":
    # Replace with the user's email
    user_email = "21f3001689@ds.study.iitm.ac.in"  # Ensure to replace with actual email
    
    # Install uv if needed
    install_uv()

    # Run the datagen.py script
    run_datagen_script(user_email)
