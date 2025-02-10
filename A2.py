import subprocess

def format_md_file(file_path):
    # Run the prettier command with the --write flag to format the file in-place
    command = ["npx", "prettier@3.4.2", "--write", file_path]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Path to the file to format
    file_path = "/data/format.md"
    
    # Format the file
    format_md_file(file_path)
