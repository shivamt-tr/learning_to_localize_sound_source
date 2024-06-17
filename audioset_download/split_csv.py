import csv
from itertools import islice

def split_csv(source_filepath, output_template, n_splits=10):
    # Open the source CSV file
    with open(source_filepath, 'r', newline='') as source_file:
        # Detect the delimiter
        delimiter = csv.Sniffer().sniff(source_file.readline()).delimiter
        source_file.seek(0)  # Go back to the start of the file
        
        # Initialize CSV reader
        reader = csv.reader(source_file, delimiter=delimiter)
        
        # Get total number of rows, excluding the header
        total_rows = sum(1 for row in reader) - 1
        source_file.seek(0)  # Go back to the start of the file
        next(reader)  # Skip header row
        
        # Calculate the number of records per file
        records_per_file = total_rows // n_splits
        
        # Split the CSV
        for split_num in range(n_splits):
            # Compute start and end lines for slicing
            start = split_num * records_per_file
            end = start + records_per_file if split_num < n_splits - 1 else total_rows
            
            # Reset the file pointer and skip header for new splits
            source_file.seek(0)
            rows = islice(reader, start, end)
            
            # Write the current split into a new file
            with open(f'{output_template}_{split_num + 1}.csv', 'w', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=delimiter)
                if split_num == 0:  # Write header only for the first split
                    source_file.seek(0)  # Go to the start to capture the header
                    writer.writerow(next(csv.reader(source_file, delimiter=delimiter)))
                for row in rows:
                    writer.writerow(row)

# Example usage
split_csv('/backup/data3/shivam/audio-visual-dataset/unbalanced_train_segments_modified.csv', 'unbalanced_train_split')

