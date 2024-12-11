import whisper
import os
import re


model = whisper.load_model("base") 


def transcribe_audio(dir, flag):

    transcript_folder = "transcript_new"
    if flag == 1:
        transcript_folder = "transcript_phishing"
    os.makedirs(transcript_folder, exist_ok=True)
    def remove_non_ascii(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    for filename in os.listdir(dir):
        
        result = model.transcribe(f"{dir}/{filename}")

        transcript_save_path = os.path.join(transcript_folder, f"{os.path.splitext(filename)[0]}_transcript.txt")
        transcript_text = remove_non_ascii(result['text'])

        with open(transcript_save_path, "w") as f:
            f.write(transcript_text)

        print(f"{filename} saved at: {transcript_save_path}")

#transcribe_audio("./non_phishing", 0)
# transcribe_audio("./phishing", 1)
# transcribe_audio("./new", 0)