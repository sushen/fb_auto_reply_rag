from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from rag import RAGSystem, init_memory_db
import fb_bot

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md', 'pptx', 'csv'}

# Initialize memory database
init_memory_db()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cms')
def cms():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
            if os.path.isfile(filepath):
                files.append({
                    'name': f,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
    files.sort(key=lambda x: x['modified'], reverse=True)
    return render_template('cms.html', documents=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files and 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if files and files[0].filename:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename or "")
                if filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
        flash(f'Successfully uploaded {len([f for f in files if f.filename])} files')
        return redirect(url_for('cms'))
    
    file = request.files.get('file')
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename or "")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        flash('File successfully uploaded')
        return redirect(url_for('cms'))
    else:
        flash('Allowed file types are txt, pdf, docx, csv')
        return redirect(url_for('cms'))

@app.route('/upload-folder', methods=['POST'])
def upload_folder():
    files = request.files.getlist('files')
    
    if not files or not files[0].filename:
        flash('No files selected')
        return redirect(url_for('cms'))
    
    uploaded_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename or "")
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1
    
    flash(f'Successfully uploaded {uploaded_count} files')
    return redirect(url_for('cms'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<name>')
def delete_file(name):
    filename = secure_filename(name)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        flash('File deleted successfully')
    
    return redirect(url_for('cms'))

# Initialize RAG system
rag_system = RAGSystem(upload_folder=app.config['UPLOAD_FOLDER'])

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', "")
    web_user_id = session.get("web_user_id")
    if not web_user_id:
        web_user_id = f"web-{uuid4().hex}"
        session["web_user_id"] = web_user_id

    return jsonify(rag_system.query(message, user_id=web_user_id))

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify([])

@app.route('/api/reload', methods=['POST'])
def reload_knowledge_base():
    try:
        rag_system.reload()
        return jsonify({"status": "success", "message": "Knowledge base reloaded"})
    except Exception as e:
        logging.error(f"Reload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Setup Facebook webhook routes
fb_bot.setup_facebook_routes(app, rag_system)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
