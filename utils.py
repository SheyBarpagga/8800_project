import os
import csv

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


output = "data.csv"

write_to_csv(spectogram, mfcc, transcript, output, 0)