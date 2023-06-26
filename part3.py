from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader('data').load_data()

index = GPTSimpleVectorIndex.from_documents(documents)

# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')

response = index.query("What are the different ways to load data into TensorFlow?  What is the role of Pandas in this?")
print(response)
