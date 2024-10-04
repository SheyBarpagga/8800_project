import whisper

model = whisper.load_model("medium") 

result = model.transcribe("T")

print(result['text'])
