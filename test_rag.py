from flask import Flask
import os
os.environ.setdefault('FLASK_APP', 'app.py')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md', 'pptx', 'csv'}

from app import RAGSystem

def test_rag_system():
    print("Testing RAG System...")
    
    os.makedirs('uploads', exist_ok=True)
    
    with open('uploads/test.txt', 'w') as f:
        f.write("The capital of France is Paris. It is known for the Eiffel Tower.")
    
    rag = RAGSystem(llm_model="qwen2.5:3b", embed_model="bge-m3:latest")
    
    print(f"Vector store initialized: {rag.vector_store is not None}")
    print(f"QA chain initialized: {rag.qa_chain is not None}")
    
    if rag.qa_chain:
        result = rag.query("What is the capital of France?")
        print(f"Query result: {result}")
    
    os.remove('uploads/test.txt')
    print("Test complete!")

if __name__ == '__main__':
    test_rag_system()
