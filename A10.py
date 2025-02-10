import sqlite3

# Connect to the SQLite database
db_path = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/ticket-sales.db'
conn = sqlite3.connect(db_path)

# Create a cursor object to execute queries
cursor = conn.cursor()

# Query to calculate total sales for the "Gold" ticket type
cursor.execute("""
    SELECT SUM(units * price) 
    FROM tickets 
    WHERE type = 'Gold'
""")

# Fetch the result
total_sales = cursor.fetchone()[0]

# If the result is None (no "Gold" ticket sales), set it to 0
if total_sales is None:
    total_sales = 0

# Write the total sales to the output file
output_path = '/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/ticket-sales-gold.txt'
with open(output_path, 'w') as f:
    f.write(str(total_sales))

# Close the database connection
conn.close()

print(f"Total sales for 'Gold' ticket type: {total_sales} written to {output_path}")
