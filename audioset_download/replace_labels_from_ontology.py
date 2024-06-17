'''
The following script replaced the label ID in csv file with their corresponding label names that are provided in ontology.json
'''

import csv
import json

# Step 1: Load the JSON file to create a mapping from label IDs to names
label_mapping = {}
with open('ontology.json') as json_file:
    labels = json.load(json_file)
    for label in labels:
        label_mapping[label['id']] = label['name']

# Step 2: Load the CSV file
entries = []
with open('unbalanced_train_segments.csv', mode='r') as infile:
    reader = csv.reader(infile, skipinitialspace=True)
    for row in reader:
        entries.append(row)

# Step 3: Replace label IDs with names, handling multiple label IDs
new_entries = []
for entry in entries:
    youtube_id, start_time, end_time, label_ids = entry
    # Splitting the label_ids string by comma, stripping whitespace, and replacing them with label names
    label_names = [label_mapping[label_id.strip()] for label_id in label_ids.split(',') if label_id.strip() in label_mapping]
    # Joining the label names with commas for entries with multiple label IDs
    new_entry = [youtube_id, start_time, end_time, ", ".join(label_names)]
    new_entries.append(new_entry)

# Step 4: Save the modified entries to a new CSV file
with open('unbalanced_train_segments_modified.csv', mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    # If your CSV includes a header, you can write it here before writing the rows
    # writer.writerow(['YouTube ID', 'Start Time', 'End Time', 'Labels'])
    writer.writerows(new_entries)