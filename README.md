# PDF Categorizer

A Streamlit-based application for uploading ZIP archives containing PDF documents, extracting their text, transforming content into numerical features, and organizing files into categories using PCA + SVM-based document classification.

## Features

- Upload a ZIP file containing multiple PDFs
- Extract text from each PDF
- Inspect document previews and summary stats
- Convert document text to TF-IDF features
- Reduce dimensionality with PCA
- Train and apply an SVM classifier for document categorization
- Download an organized ZIP file with documents grouped by predicted category

## Tech Stack

- Python
- Streamlit
- PyPDF2
- scikit-learn
- pandas
- NumPy
- reportlab

## Project Structure

- `app.py` – main Streamlit application
- `pdf_gen.py` – PDF generation utilities
- `pdf_utils/` – helper functions for extraction, text processing, and classification
- `requirements.txt` – Python dependencies
- `DockerFile` – container configuration
- `sample-case-files/` and `sample-case-files-synthetic-ai/` – example document sets

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ADITYAPIMPALE11127/pdf-categorizer.git
   cd pdf-categorizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown by Streamlit in your browser.

## How It Works

1. Upload a ZIP archive with one or more PDFs.
2. The app extracts text from each PDF.
3. TF-IDF features are generated for the extracted text.
4. PCA reduces high-dimensional text features.
5. A Support Vector Machine (SVM) is trained on sample data and used to classify uploaded documents.
6. Documents are grouped by predicted category and can be downloaded as a reorganized ZIP.

## Notes

- This project is intended for document categorization experiments and demonstrations.
- For production use, you would typically replace the sample training data with a properly labeled real-world dataset.
- Some PDFs may not be suitable for text extraction if they are image-only or scanned documents without selectable text.

## License

This project does not currently include a license file. If you plan to share or distribute it publicly, consider adding an appropriate open-source license.

## Example Usage

- Create a ZIP of several PDFs in different categories, such as contracts, invoices, reports, or service agreements.
- Upload the ZIP in the app.
- Review the extracted summaries, PCA insights, and SVM predictions.
- Download the categorized output ZIP.
