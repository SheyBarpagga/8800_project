import os
import csv
import natsort
import re

# Split CSV file and perform natural sort on filenames
# (single use script for moving data over)
def move_csv():

    csv_path = "./archive(3)/test.csv"
    folder_path = "./archive(3)/audio-wav/audio-wav-16khz/"

    file_list = os.listdir(folder_path)
    sorted_files = natsort.natsorted(file_list)

    with open(csv_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader) 
        csv_data = list(reader)


    reordered_data = [header]  
    matched_rows = []

    for file_name in sorted_files:
        for row in csv_data:
            if row[0].replace("audio-wav-16khz/", "") == file_name: 
                # print(row)
                matched_rows.append(row)
                break

    reordered_data.extend(matched_rows)

    half_index = len(matched_rows) // 2
    first_half = [header] + matched_rows[:half_index]
    second_half = [header] + matched_rows[half_index:]

    with open("first_half.csv", mode="w", newline="") as first_csv:
        writer = csv.writer(first_csv)
        writer.writerows(first_half)

    with open("second_half.csv", mode="w", newline="") as second_csv:
        writer = csv.writer(second_csv)
        writer.writerows(second_half)


# Remove all non-ASCII characters from the csv
def clean_csv():

    input_csv = "./archive(3)/metadata.csv"
    output_csv = "test.csv"

    def remove_non_ascii(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    with open(input_csv, mode="r", newline="", encoding="utf-8") as infile, \
        open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            cleaned_row = [remove_non_ascii(cell) if isinstance(cell, str) else cell for cell in row]
            writer.writerow(cleaned_row)

    print(f"Non-ASCII characters removed and saved as '{output_csv}'")


def write_to_csv(spectogram, mfcc, transcript, output_csv, flag):

    spectogram_files = []
    mfcc_files = []
    txt_files = []
    phishing = 1

    if flag == 0:
        phishing = 0


    for filename in os.listdir(spectogram):
        spectogram_files.append(f"{spectogram}/{filename}")
        

    for filename in os.listdir(mfcc):
        mfcc_files.append(f"{mfcc}/{filename}")

    for filename in os.listdir(transcript):
        txt_files.append(f"{transcript}/{filename}")



    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["spectogram", "mfcc", "transcript", "phishing"])

        for spec, mfcc_file, transcript_file in zip(spectogram_files, mfcc_files, txt_files):

            with open(transcript_file, 'r') as tfile:
                text_content = tfile.read().strip() 

            writer.writerow([spec, mfcc_file, text_content, f"{phishing}"])

spectogram = "./spectogram"
mfcc = "./mfcc"
transcript = "./transcript"

# spectogram = "./phishing_spectogram"
# mfcc = "./phishing_mfcc"
# transcript = "./transcript_phishing"

# spectogram = "./new_specto"
# mfcc = "./new_mfcc"
# transcript = "./transcript_new"

output = "test.csv"

write_to_csv(spectogram, mfcc, transcript, output, 0)
# move_csv()