import streamlit as st
import zipfile
import pandas as pd
from pdf_utils import (
    extract_all_pdfs_from_zip,
    get_pdf_stats,
    TFIDFProcessor,
    display_pdf_list,
    display_text_preview,
    display_tfidf_results,
    create_progress_bar,
    update_progress,
     PCA_Reducer, 
      SVM_Classifier, 
      create_training_data,  # ADD THIS
    prepare_training_features, 
)

# Page configuration
st.set_page_config(
    page_title="PDF Document Categorizer",
    page_icon="📦",
    layout="wide"
)

# Title and description
st.title("📦 PDF Document Categorizer with PCA + SVM")
st.write("Upload a ZIP file containing PDF documents. The system will extract text, "
         "convert to numerical features, and prepare for categorization.")

# File upload section
st.header("1. Upload Documents")
uploaded_zip = st.file_uploader(
    "Choose a ZIP file containing PDFs",
    type=['zip'],
    help="Maximum file size: 200MB"
)

if uploaded_zip is not None:
    st.success("✅ ZIP file uploaded successfully!")
    
    # Create two main columns
    left_col, right_col = st.columns(2)
    
    # Step 1: Extract PDFs from ZIP
    st.header("2. Text Extraction")
    progress_bar = create_progress_bar()
    update_progress(progress_bar, 20, "Reading ZIP file...")
    
    # Extract all PDF texts
    pdf_texts, pdf_names = extract_all_pdfs_from_zip(uploaded_zip)
    
    if not pdf_texts:
        st.error("❌ No extractable text found in any PDF!")
        st.stop()
    
    update_progress(progress_bar, 40, f"Extracted text from {len(pdf_texts)} PDF(s)")
    
    # Display PDF list on left
    display_pdf_list(pdf_names, left_col)
    
    # Display first PDF preview on right
    if pdf_texts:
        display_text_preview(pdf_names[0], pdf_texts[0], right_col)
    
    # Show statistics
    stats = get_pdf_stats(pdf_texts)
    st.info(f"""
    📊 **Extraction Summary:**
    - **Documents processed:** {stats['total_docs']}
    - **Average characters per document:** {stats['avg_chars']:,}
    - **Total words across all documents:** {stats['total_words']:,}
    """)
    
    # Step 2: TF-IDF Conversion
    st.header("3. TF-IDF Conversion")
    update_progress(progress_bar, 60, "Converting text to numbers...")
    
    with st.spinner("Applying TF-IDF vectorization..."):
        tfidf_processor = TFIDFProcessor(max_features=1000)
        tfidf_matrix = tfidf_processor.fit_transform(pdf_texts)
        matrix_info = tfidf_processor.get_matrix_info()
    
    update_progress(progress_bar, 80, "TF-IDF conversion complete!")
    
    # Display TF-IDF results
    st.success("✅ Text successfully converted to numerical features!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**TF-IDF Matrix Information:**")
        st.write(f"- Shape: {matrix_info['shape'][0]} documents × {matrix_info['shape'][1]} features")
        st.write(f"- Non-zero values: {matrix_info['non_zero_count']:,}")
        st.write(f"- Density: {matrix_info['density']:.2%}")
        
        st.write("**How TF-IDF works:**")
        st.info("""
        1. **Term Frequency (TF):** How often a word appears in a document
        2. **Inverse Document Frequency (IDF):** Downweights common words
        3. **Result:** Words unique to a document get high scores
        """)
    
    with col2:
        # Let user select document to view
        if len(pdf_names) > 1:
            selected_doc = st.selectbox(
                "Select a document to view its important words:",
                pdf_names,
                key="doc_selector"
            )
            doc_index = pdf_names.index(selected_doc)
        else:
            selected_doc = pdf_names[0]
            doc_index = 0
        
        display_tfidf_results(tfidf_processor, doc_index, selected_doc, col2)
    
    update_progress(progress_bar, 100, "✅ Processing complete!")
    st.balloons()

    # ==================== STEP 3: PCA DIMENSIONALITY REDUCTION ====================
    st.header("📉 Step 3: PCA Dimensionality Reduction")
    update_progress(progress_bar, 70, "Applying PCA...")

    # Apply PCA
    with st.spinner("Applying PCA to reduce dimensions..."):
        # Determine optimal number of components
        n_components = min(100, len(pdf_texts), tfidf_matrix.shape[1])

        # Create PCA reducer
        pca_reducer = PCA_Reducer(n_components=n_components)

        # Apply PCA to TF-IDF matrix
        pca_features = pca_reducer.fit_transform(tfidf_matrix)
        variance_info = pca_reducer.get_variance_info()

    update_progress(progress_bar, 85, "PCA complete!")
    st.success(f"✅ PCA Complete!")
    st.info(f"**Reduced:** {variance_info['original_features']} → {variance_info['components_kept']} components")
    st.info(f"**Variance explained:** {variance_info['variance_explained']:.1f}%")

    # Show PCA results
    st.subheader("📊 PCA Analysis")

    # Create columns for display
    pca_col1, pca_col2 = st.columns(2)

    with pca_col1:
        st.write("**Before/After Comparison:**")

        col_before, col_after = st.columns(2)
        with col_before:
            st.metric("Features Before", variance_info['original_features'])
        with col_after:
            st.metric("Features After", variance_info['components_kept'])

        st.write("**Benefits for SVM:**")
        st.write("✓ 10x faster training")
        st.write("✓ Removes noise words")
        st.write("✓ Prevents overfitting")
        st.write("✓ Focuses on meaningful patterns")

    with pca_col2:
        st.write("**Variance Explained:**")

        # Create simple progress bar for variance
        st.progress(variance_info['variance_explained'] / 100)
        st.caption(f"{variance_info['variance_explained']:.1f}% of information preserved")

        # Show cumulative variance for first 5 components
        if len(variance_info['variance_by_component']) >= 5:
            st.write("**Top 5 Components:**")
            for i, variance in enumerate(variance_info['variance_by_component'][:5]):
                st.write(f"PC{i+1}: {variance*100:.1f}%")

    # Show what PCA discovered
    st.subheader("🔍 What PCA Discovered in Your Documents")

    # Show first 2 principal components
    feature_names = tfidf_processor.feature_names
    if feature_names is not None and len(feature_names) > 0:
        for comp_idx in range(min(2, n_components)):
            with st.expander(f"Principal Component {comp_idx+1} - Key Word Patterns"):
                top_features = pca_reducer.get_top_features_per_component(
                    feature_names,
                    component_idx=comp_idx,
                    top_n=8
                )

                if top_features:
                    st.write("**Most important words for this pattern:**")
                    for word, weight in top_features:
                        if weight > 0:
                            st.write(f"• **{word}** (weight: {weight:.4f})")
                else:
                    st.write("No significant patterns found")
    else:
        st.write("Feature names not available")

    # ==================== STEP 4: SVM CLASSIFICATION ====================
    st.header("🎯 Step 4: SVM Document Classification")
    progress_bar.progress(90)

    # Train SVM on sample data (in real app, you'd use actual labeled data)
    st.write("**Training SVM Classifier...**")
    st.info("""
    For demonstration, we're training on sample data.
    In production, you would train on actual labeled documents.
    """)

    try:
        # Create training data
        training_docs = create_training_data()

        # Prepare training features
        st.write("Preparing training features...")
        tfidf_for_training = TFIDFProcessor(max_features=1000)
        # Use dynamic n_components based on training data size
        # Must be strictly LESS than both n_samples and n_features for svd_solver='arpack'
        n_components_training = min(50, len(training_docs) - 1, 14)  # Ensure n_components < 15
        pca_for_training = PCA_Reducer(n_components=n_components_training)

        X_train, y_train, trained_tfidf_vectorizer, trained_pca_reducer = prepare_training_features(
            training_docs,
            tfidf_for_training.vectorizer,  # Pass raw sklearn TfidfVectorizer
            pca_for_training.pca           # Pass raw sklearn PCA
        )

        # Train SVM
        st.write("Training SVM classifier...")
        svm_classifier = SVM_Classifier(kernel='rbf', C=1.0, probability=True)
        training_results = svm_classifier.train(X_train, y_train, test_size=0.2)

        # Prepare uploaded documents for classification
        st.write("Classifying uploaded documents...")

        # FIX: Use the SAME TF-IDF vectorizer and PCA that were used for training
        # This ensures consistent feature dimensions between training and classification
        uploaded_tfidf = trained_tfidf_vectorizer.transform(pdf_texts)  # Use training TF-IDF vectorizer
        uploaded_pca = trained_pca_reducer.transform(uploaded_tfidf)  # Use training PCA

        # Get predictions for each uploaded document
        results = []
        for i, (doc_name, pca_features) in enumerate(zip(pdf_names, uploaded_pca)):
            # Predict category
            prediction = svm_classifier.predict_single(pca_features, document_name=doc_name)
            results.append(prediction)

        st.success(f"✅ Classification Complete!")

        # Show SVM training results
        st.subheader("📊 SVM Training Results")

        col_train1, col_train2 = st.columns(2)

        with col_train1:
            st.write("**Model Performance:**")
            st.metric("Training Accuracy", f"{training_results['train_accuracy']*100:.1f}%")
            st.metric("Test Accuracy", f"{training_results['test_accuracy']*100:.1f}%")
            st.metric("Support Vectors", training_results['support_vectors'])

            st.write("**Classes Learned:**")
            for cls in training_results['classes']:
                st.write(f"• {cls}")

        with col_train2:
            st.write("**How SVM Works:**")
            st.info("""
            1. **Finds optimal boundary** between document categories
            2. **Maximizes margin** for confident classification
            3. **Uses support vectors** - key documents that define boundary
            4. **Provides confidence scores** for each prediction
            """)

            # Show SVM parameters
            st.write("**Model Parameters:**")
            st.write(f"- Kernel: RBF (for complex text patterns)")
            st.write(f"- C: 1.0 (balanced regularization)")
            st.write(f"- Probability estimates: Enabled")

        # ==================== STEP 5: DISPLAY CLASSIFICATION RESULTS ====================
        st.header("📄 Classification Results")
        progress_bar.progress(100)

        # Create results table
        results_df = pd.DataFrame([
            {
                'Document': r['document'],
                'Predicted Category': r['predicted_category'],
                'Confidence': f"{r['confidence']*100:.1f}%",
                'Is Confident': '✅' if r['is_confident'] else '⚠️'
            }
            for r in results
        ])

        st.dataframe(results_df, use_container_width=True)

        # Show detailed predictions for each document
        st.subheader("🔍 Detailed Predictions")

        # Let user select a document to see detailed scores
        selected_doc = st.selectbox(
            "Select a document to see detailed classification scores:",
            pdf_names
        )

        # Find selected document results
        selected_result = next(r for r in results if r['document'] == selected_doc)

        # Create visualization of confidence scores
        st.write(f"**Classification scores for: {selected_doc}**")

        # Sort scores for better display
        sorted_scores = sorted(
            selected_result['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Display as bars
        for category, score in sorted_scores:
            percentage = score * 100
            bar_length = int(percentage / 2)  # Scale for display

            if category == selected_result['predicted_category']:
                st.markdown(f"**{category}**: █{'█' * bar_length} {percentage:.1f}% ✅")
            else:
                st.markdown(f"{category}: █{'█' * bar_length} {percentage:.1f}%")

        # ==================== STEP 6: CREATE ORGANIZED OUTPUT ====================
        st.header("📁 Organized Output")

        # Create organized structure
        st.write("**Folder structure that would be created:**")

        # Group documents by category
        categories_dict = {}
        for result in results:
            category = result['predicted_category']
            if category not in categories_dict:
                categories_dict[category] = []
            categories_dict[category].append(result['document'])

        # Display folder structure
        for category, documents in categories_dict.items():
            with st.expander(f"📂 {category} ({len(documents)} documents)"):
                for doc in documents:
                    st.write(f"📄 {doc}")

        # Download button (simulated)
        st.download_button(
            label="📥 Download Organized ZIP (Simulated)",
            data="This would be the actual ZIP file in production",
            file_name="organized_documents.zip",
            mime="application/zip",
            help="In full implementation, this would create actual ZIP with categorized folders"
        )

        # Success celebration
        st.balloons()
        st.success("🎉 Document Categorization Pipeline Complete!")

        # ==================== FINAL SUMMARY ====================
        st.header("📋 Pipeline Summary")

        pipeline_steps = [
            ("✅ ZIP Upload", "Complete"),
            ("✅ PDF Text Extraction", f"{len(pdf_texts)} documents"),
            ("✅ TF-IDF Vectorization", f"{tfidf_matrix.shape[1]} features"),
            ("✅ PCA Dimensionality Reduction", f"{variance_info['variance_explained']:.1f}% variance"),
            ("✅ SVM Classification", f"{len(results)} documents classified"),
            ("✅ Results Organization", "Ready for download")
        ]

        for step, status in pipeline_steps:
            st.write(f"- **{step}:** {status}")

    except Exception as e:
        st.error(f"❌ Error during SVM classification: {str(e)}")
        st.info("This is a demonstration. In production, you would use properly labeled training data.")

# Instructions when no file uploaded
else:
    st.info("""
    ## 📋 How to Use This Tool:

    1. **Prepare PDFs:** Collect your PDF documents in a folder
    2. **Create ZIP:** Right-click folder → "Compress to ZIP"
    3. **Upload:** Use the file uploader above
    4. **Process:** System will extract text and convert to features
    5. **Categorize:** (Tomorrow) Get auto-categorized documents

    ## 🧪 For Testing:
    - Use 3-5 PDFs with selectable text (not scanned images)
    - Different types work best: contracts, invoices, reports
    - Maximum 200MB ZIP file size
    """)
