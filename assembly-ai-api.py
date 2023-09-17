import os

import requests
import json
import time

import dotenv

dotenv.load_dotenv()

# URL of the file to transcribe
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

# AssemblyAI transcript endpoint (where we submit the file)
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

# request parameters
data = {
    "audio_url": FILE_URL # You can also use a URL to an audio or video file on the web
}

# HTTP request headers
headers={
  "Authorization": os.environ['ASSEMBLY_AI_API_KEY'],
  "Content-Type": "application/json"
}

# submit for transcription via HTTP request
response = requests.post(transcript_endpoint,
                         json=data,
                         headers=headers)

# polling for transcription completion
polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{response.json()['id']}"

while True:
    transcription_result = requests.get(polling_endpoint, headers=headers).json()

    if transcription_result['status'] == 'completed':
        # print the results
        print(json.dumps(transcription_result, indent=2))
        break
    elif transcription_result['status'] == 'error':
        raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
    else:
        time.sleep(3)
