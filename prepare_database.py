import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    #Check if database should be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action = "store_true", help = "Reset the database")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    #Create or update data store
    else:
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    #Load existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function= get_embeddings()
    )
    
    #Calculate page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    #Add or update documents
    existing_items = db.get(include=[]) #IDs are included by default
    existing_ids = set(existing_items['ids'])
    print(f'Number of existing documents in DB: {len(existing_ids)}')

    #Only add documents that don't exist in DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Add new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # Create IDs in the format of Page Source : Page Number : Chunk Index
    # Ex) data/AWS_Clouds_Practitioner_Notes:2:4

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f'{source}:{page}'

        if current_page_id == last_page_id:
            current_chunk_index+=1
        else:
            current_chunk_index = 0
        
        #Calculate the chunk ID
        chunk_id = f'{current_page_id}:{current_chunk_index}'
        last_page_id = current_page_id

        #Add to the page meta-data
        chunk.metadata['id'] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == '__main__':
    main()