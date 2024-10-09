import whisper
import os

model = whisper.load_model("medium") 


def transcribe_audio(dir, flag):

    transcript_folder = "transcript"
    if flag == 1:
        transcript_folder = "transcript_phishing"
    os.makedirs(transcript_folder, exist_ok=True)

    for filename in os.listdir(dir):
        
        result = model.transcribe(f"{dir}/{filename}")

        transcript_save_path = os.path.join(transcript_folder, f"{os.path.splitext(filename)[0]}_transcript.txt")
        transcript_text = result['text']

        with open(transcript_save_path, "w") as f:
            f.write(transcript_text)

        print(f"{filename} saved at: {transcript_save_path}")

#transcribe_audio("./non_phishing", 0)
transcribe_audio("./phishing", 1)