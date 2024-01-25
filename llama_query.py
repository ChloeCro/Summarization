from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
import os

if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'api-key-here'

    documents = SimpleDirectoryReader('data').load_data()

    try:
        storage_context = StorageContext.from_defaults('..')

        index = load_index_from_storage(storage_context)
    except:
        index = GPTVectorStoreIndex.from_documents(documents)

        index.storage_context.persist()

query_engine = index.as_query_engine()
response = query_engine.query("query here")
print(response)