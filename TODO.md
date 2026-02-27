# PDF Categorizer Project TODO

## Current Status
- ✅ Project setup with Streamlit app
- ✅ PDF text extraction functionality
- ✅ TF-IDF vectorization
- ✅ PCA dimensionality reduction
- ✅ SVM classification model
- ❌ **BUG**: PCA_Reducer missing `transform` method causing AttributeError

## Immediate Fixes
- [ ] Add `transform` method to PCA_Reducer class in `pdf_utils/tfidf_converter.py`
- [ ] Test the SVM classification pipeline
- [ ] Run the Streamlit app to verify functionality

## Workflow Summary
1. User uploads ZIP file containing PDFs
2. Extract text from all PDFs
3. Convert text to TF-IDF features
4. Apply PCA for dimensionality reduction
5. Train SVM classifier on sample data
6. Classify uploaded documents
7. Return categorized PDFs with content info

## Testing
- [ ] Test with sample PDF files
- [ ] Verify PCA transform works correctly
- [ ] Check SVM predictions are accurate
- [ ] Ensure app runs without errors
