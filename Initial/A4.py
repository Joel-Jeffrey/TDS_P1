import json

# Load the contacts from the JSON file
with open('/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/contacts.json', 'r') as f:
    contacts = json.load(f)

# Sort the contacts by last_name, then first_name
sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

# Write the sorted contacts to a new JSON file
with open('/mnt/f24a4001-dd56-4460-a9b9-60aed01a8e61/IITM/TDS_P1/data/contacts-sorted.json', 'w') as f:
    json.dump(sorted_contacts, f, indent=4)

print("Contacts have been sorted and saved")
