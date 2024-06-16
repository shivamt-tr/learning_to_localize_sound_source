import os
import shutil
import numpy as np

def split_files(source_folder, train_folder, test_folder, test_ratio=0.1):
    # Create train and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all .pt files in the source folder
    files = [f for f in os.listdir(source_folder) if f.endswith('.pt')]

    # Shuffle the files
    np.random.shuffle(files)

    # Calculate the number of test files
    num_files = len(files)
    num_test = int(num_files * test_ratio)

    # Split the files into test and train
    test_files = files[:num_test]
    train_files = files[num_test:]

    # Move the test files
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))

    # Move the train files
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))

    print(f'Moved {len(test_files)} files to {test_folder}')
    print(f'Moved {len(train_files)} files to {train_folder}')

# Example usage
source_folder = '/mnt/nvme-4tb/shivam/avdataset/precomputed_mel'
train_folder = '/mnt/nvme-4tb/shivam/dataset/av_train'
test_folder = '/mnt/nvme-4tb/shivam/dataset/av_test'

split_files(source_folder, train_folder, test_folder)
