import jwt
import time

API_KEY = 'B2cmAr_XT1iPpXO7wvcsNA'
API_SECRET = 'N6hJEkfluVJROFRyv2ijJT9dtUSBKs6L'

def generate_zoom_signature(meeting_number, role=0):
    iat = int(time.time())
    exp = iat + 60 * 5
    payload = {
        'sdkKey': API_KEY,
        'mn': meeting_number,
        'role': role,
        'iat': iat,
        'exp': exp,
        'appKey': API_KEY,
        'tokenExp': exp
    }
    return jwt.encode(payload, API_SECRET, algorithm='HS256')

@app.get("/zoom-auth")
async def zoom_auth():
    meeting_number = 'your_meeting_number'
    signature = generate_zoom_signature(meeting_number)
    return {"signature": signature}
