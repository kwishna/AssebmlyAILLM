import os

import assemblyai as aai
import dotenv

dotenv.load_dotenv()

# replace with your API token
aai.settings.api_key = os.environ['ASSEMBLY_AI_API_KEY']

# URL of the file to transcribe
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

# print(transcript.export_subtitles_srt())
# print(transcript.export_subtitles_vtt())

print(transcript.text)
print(transcript.summary)

# Get the parts of the transcript that were tagged with topics
for result in transcript.iab_categories.results:
    print(result.text)
    print(f"Timestamp: {result.timestamp.start} - {result.timestamp.end}")
    for label in result.labels:
        print(label.label)  # topic
        print(label.relevance)  # how relevant the label is for the portion of text

# Get a summary of all topics in the transcript
for label, relevance in transcript.iab_categories.summary.items():
    print(f"Audio is {relevance * 100}% relevant to {label}")

for result in transcript.auto_highlights.results:
    print(f"Highlight: {result.text}, Count: {result.count}, Rank: {result.rank}, Timestamps: {result.timestamps}")

    # extract all utterances from the response
    utterances = transcript.utterances

    # For each utterance, print its speaker and what was said
    for utterance in utterances:
        speaker = utterance.speaker
        text = utterance.text
        print(f"Speaker {speaker}: {text}")

# Podcast summarize podcast
for chapter in transcript.chapters:
    print("Chapter Start Time:", chapter.start)
    print("Chapter End Time:", chapter.end)
    print("Chapter Gist:", chapter.gist)
    print("Chapter Headline:", chapter.headline)
    print("Chapter Summary:", chapter.summary)
