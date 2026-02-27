import streamlit as st

def display_pdf_list(pdf_files, column):
    """
    Display list of PDF files in a column
    """
    with column:
        st.write("### 📄 PDF Files Found")
        for i, pdf_name in enumerate(pdf_files):
            st.write(f"{i+1}. {pdf_name}")

def display_text_preview(pdf_name, text, column):
    """
    Display text preview in a column
    """
    with column:
        st.write(f"### 📝 Preview: {pdf_name}")
        
        preview_text = text[:500] + "..." if len(text) > 500 else text
        st.text_area("First 500 characters:", preview_text, height=200, key="preview")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chars", len(text))
        with col2:
            st.metric("Words", len(text.split()))
        with col3:
            st.metric("Avg Word Length", f"{sum(len(w) for w in text.split())//len(text.split()) if text.split() else 0}")

def display_tfidf_results(tfidf_processor, doc_index, pdf_name, column):
    """
    Display TF-IDF results for a document
    """
    with column:
        st.write(f"### 🔢 TF-IDF Analysis: {pdf_name}")
        
        # Get top words
        top_words = tfidf_processor.get_top_words(doc_index, 10)
        
        if top_words:
            st.write("**Top 10 Important Words:**")
            for i, (word, score) in enumerate(top_words):
                if score > 0:
                    st.write(f"{i+1}. **{word}**: `{score:.4f}`")
                else:
                    st.write(f"{i+1}. {word}: {score:.4f}")
        else:
            st.info("No significant words found (all scores are 0)")

def create_progress_bar():
    """
    Create and return a progress bar
    """
    return st.progress(0)

def update_progress(progress_bar, value, message=""):
    """
    Update progress bar with optional message
    """
    progress_bar.progress(value)
    if message:
        st.caption(message)