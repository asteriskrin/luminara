# Luminara
Luminara is an chatbot designed to answer questions about stories contained in PDF files. When an answer is not found within the PDFs, it searches the internet for relevant information.

## Requirements
- Python 3.10 or higher

## Installation
1. Install dependencies by running:
   ```bash
   poetry install
   ```
2. Create the following directories in the root of the project:
   - `data/chroma_db`
   - `data/document`
3. Place your PDF files into the `data/document` directory.
4. Build the Chroma database by executing the script `rag_chroma_db.ipynb`.
5. Launch the chatbot with:
   ```bash
   chainlit run main.py
   ```
