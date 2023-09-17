import os

from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import assemblyai as aai
import dotenv

dotenv.load_dotenv()

FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

config = aai.TranscriptionConfig(speaker_labels=True, auto_chapters=True, entity_detection=True)

loader = AssemblyAIAudioTranscriptLoader(FILE_URL, config=config, api_key=os.environ["ASSEMBLY_AI_API_KEY"])
docs = loader.load()

llm = OpenAI()
qa_chain = load_qa_chain(llm, chain_type="stuff")

answer = qa_chain.run(input_documents=docs, question="Where did the wildfire start?")
print(answer)