# TradeGuide RAG Chatbot

TradeGuide is an early-stage Retrieval-Augmented Generation (RAG) chatbot prototype for a group project. The project includes a topic-based educational knowledge base, an evaluation question set, design notes, and a simple Gradio app for local testing.

## Project Structure

```text
TradeGuide-RAG-Chatbot/
├── docs/              # Topic-based educational knowledge base files
├── test_questions/    # Evaluation dataset with 50 sample user questions
├── design_notes/      # Knowledge base design notes
└── src/               # Gradio prototype and retrieval code
```

## Features

* Local Gradio chatbot interface
* Knowledge base stored as topic-based documents
* Sample evaluation questions for testing chatbot responses
* Retrieval pipeline that can be extended with embeddings, FAISS, and Hugging Face models
* Early-stage structure suitable for experimentation and future RAG improvements

## Requirements

* Python 3.10 or newer recommended
* `pip`
* Optional: Hugging Face token for higher Hub rate limits

## Setup

```bash
git clone <repository-url>
cd TradeGuide-RAG-Chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Build the Index

```bash
python src/build_index.py
```

## Run the Prototype

```bash
python src/app.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Hugging Face Token

The app can run without a Hugging Face token, but setting one may improve download reliability and rate limits.

```bash
export HF_TOKEN="your_hugging_face_token"
```

## Current Notes

The prototype is currently local and experimental. It can be improved by adding:

* Sentence Transformers for semantic embeddings
* FAISS for vector search
* A Hugging Face language model for answer generation
* Better chunking and metadata handling
* Automated evaluation using the sample questions

## Known Warning

The current retriever may show this warning:

```text
FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
```
This does not stop the app from running, but the retriever code can be updated later to use the newer method name.

