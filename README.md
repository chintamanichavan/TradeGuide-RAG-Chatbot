# TradeGuide Knowledge Base Package

This package contains early-stage project materials for the TradeGuide group project.

## Contents
- docs/: Topic-based educational knowledge base files.
- test_questions/: Evaluation dataset with 50 sample user questions.
- design_notes/: Knowledge base design notes.
- src/: Simple Gradio prototype using keyword-based retrieval.

## Run Prototype
```bash
pip install -r src/requirements.txt
python src/app.py
```

This prototype can be extended with Sentence Transformers, FAISS, and a Hugging Face language model.
