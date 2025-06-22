from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

loader = TextLoader("new-Policies.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(data)
watsonx_embedding = watsonx_embedding()
ids = [str(i) for i in range(0, len(chunks))]

vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

query = "Smoking policy"
top_k_results = vectordb.similarity_search(query, k = 5)
print(top_k_results)

