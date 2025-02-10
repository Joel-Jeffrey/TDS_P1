import datetime
from dateutil import parser  # For fuzzy parsing (if needed)

def count_wednesdays(filepath, use_fuzzy_parsing=False):
    """Counts Wednesdays in a file with various date formats.

    Args:
        filepath: Path to the text file.
        use_fuzzy_parsing: If True, uses dateutil for fuzzy parsing (for messy data).

    Returns:
        The number of Wednesdays, or None if the file is not found.
    """

    wednesday_count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace

                if use_fuzzy_parsing:
                    try:
                        date_obj = parser.parse(line).date()
                        if date_obj.weekday() == 2:
                            wednesday_count += 1
                    except (parser.ParserError, ValueError):  # Handle parsing errors from dateutil
                        pass  # Skip lines that can't be parsed
                else:  # Use strptime for well-formatted dates
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
                                wednesday_count += 1
                            break  # Stop checking formats after a match
                        except ValueError:
                            pass  # Move to the next format

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    return wednesday_count


# Example usage:
filepath = "/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/dates.txt"  # Replace with your file path

# Choose one of the following:

# 1. For mostly well-formatted dates (faster):
# count = count_wednesdays(filepath)

# 2. For messy/inconsistent dates (slower but more flexible):
count = count_wednesdays(filepath, use_fuzzy_parsing=True)  # Uncomment to use dateutil

if count is not None:
    print(f"Number of Wednesdays in the file: {count}")

# Write the result to the output file
with open('/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/dates-wednesdays.txt', 'w') as f:
    f.write(str(count))
