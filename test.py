from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()
embeddings = OpenAIEmbeddings()

loader = DirectoryLoader('data', glob="**/*.txt")

documents = loader.load()
print(len(documents))