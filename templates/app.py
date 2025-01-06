from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
import nltk
import os

# Download NLTK tokenizer models
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Load a robust Question Generation Pipeline
question_generator = pipeline("text2text-generation", model="google/flan-t5-large")  # Use a more powerful model

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Preprocess sentences before generating questions
def preprocess_sentence(sentence):
    # Remove unnecessary whitespace and ensure the sentence is meaningful
    sentence = sentence.strip()
    if len(sentence.split()) < 5:  # Skip very short sentences
        return None
    return sentence

# Generate questions from a paragraph
def generate_questions_from_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    questions = []

    for sentence in sentences:
        preprocessed_sentence = preprocess_sentence(sentence)
        if not preprocessed_sentence:
            continue

        try:
            input_text = f"Create a well-formed question based on this sentence: {preprocessed_sentence}"
            result = question_generator(input_text, max_length=128, do_sample=False)
            if result and result[0]['generated_text']:
                questions.append(result[0]['generated_text'])
        except Exception as e:
            print(f"Error generating question for sentence: {sentence}\n{e}")
    
    return questions

# Post-process generated questions
def post_process_questions(questions):
    processed_questions = []
    seen = set()
    for question in questions:
        question = question.strip()
        if question and question not in seen and question.endswith('?'):  # Ensure it's a valid question
            seen.add(question)
            processed_questions.append(question)
    return processed_questions

# Home route to render the page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the PDF upload and generate questions
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(file_path)

        if pdf_text:
            # Generate questions from the extracted text
            questions = generate_questions_from_paragraph(pdf_text)
            questions = post_process_questions(questions)
            return jsonify({"questions": questions})
        else:
            return jsonify({"error": "No text extracted from the PDF."}), 400
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
