from langchain_community.document_loaders import PyPDFLoader

## Document loader
def document_loader(pdf_url):
    loader = PyPDFLoader(pdf_url)
    loaded_document = loader.load()
    return loaded_document

pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WgM1DaUn2SYPcCg_It57tA/A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf"

loaded_pdf = document_loader(pdf_url)
print(loaded_pdf[0].page_content[:1000])