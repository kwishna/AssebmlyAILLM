import 'dotenv/config';
import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from 'langchain/chains';
import { AudioTranscriptLoader } from 'langchain/document_loaders/web/assemblyai';
import {config} from 'dotenv'
config()

(async () => {
  const llm = new OpenAI({});
  const chain = loadQAStuffChain(llm);

  const loader = new AudioTranscriptLoader({
    // You can also use a local path to an audio file, like ./sports_injuries.mp3
    audio_url: "https://storage.googleapis.com/aai-docs-samples/sports_injuries.mp3",
    language_code: "en_us"
  });
  const docs = await loader.load();

  const response = await chain.call({
    input_documents: docs,
    question: "What is a runner's knee?",
  });
  console.log(response.text);
})();
