import os

import assemblyai as aai
import dotenv

dotenv.load_dotenv()

# replace with your API token
aai.settings.api_key = os.environ['ASSEMBLY_AI_API_KEY']

transcriber = aai.Transcriber()
transcript_group = transcriber.transcribe_group(
    [
        "https://example.org/customer1.mp3",
        "https://example.org/customer2.mp3",
    ],
)

params = {
  "context": "The caller is interested in buying a car.",
  "answer_format": "Short sentence"
}

result = transcript_group.lemur.summarize(**params)
print(result.response)

result = transcript_group.lemur.task(**params)
print(result.response)

result = transcript_group.lemur.action_items(**params)
print(result.response)