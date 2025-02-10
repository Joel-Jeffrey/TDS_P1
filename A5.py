import os

# Get a sorted list of log files by modification time
log_dir = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/logs'
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')],
                   key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
                   reverse=True)[:10]

# Read the first line from each of the 10 most recent log files
first_lines = []
for log_file in log_files:
    with open(os.path.join(log_dir, log_file), 'r') as f:
        first_lines.append(f.readline().strip())

# Write the first lines to a new file
with open('/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/logs-recent.txt', 'w') as f:
    for line in first_lines:
        f.write(line + '\n')

print("The first lines of the 10 most recent log files have been saved")
