from langchain.llms.google_palm import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

import os
load_dotenv()
llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

summarize_chain = load_summarize_chain(llm)

document_loader = PyPDFLoader(file_path="/.pdf")
document = document_loader.load()

summary = summarize_chain(document)
print(summary['output_text'])
