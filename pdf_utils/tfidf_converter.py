import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# ADD TO EXISTING tfidf_converter.py
from sklearn.decomposition import TruncatedSVD
import numpy as np

class TFIDFProcessor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        self.feature_names = None
        self.tfidf_matrix = None
        
    def fit_transform(self, texts):
        """
        Convert texts to TF-IDF matrix
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def get_top_words(self, doc_index, top_n=10):
        """
        Get top N words for a specific document
        """
        if self.tfidf_matrix is None:
            return []
        
        doc_scores = self.tfidf_matrix[doc_index].toarray()[0]
        word_scores = list(zip(self.feature_names, doc_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:top_n]
    
    def get_document_top_words(self, texts, top_n=15):
        """
        Get top words for all documents as DataFrame
        """
        results = []
        
        for idx, text in enumerate(texts):
            top_words = self.get_top_words(idx, top_n)
            for word, score in top_words:
                if score > 0:
                    results.append({
                        'Document': f"Doc_{idx+1}",
                        'Word': word,
                        'TF-IDF Score': score
                    })
        
        return pd.DataFrame(results)
    
    def get_matrix_info(self):
        """
        Get TF-IDF matrix information
        """
        if self.tfidf_matrix is not None:
            return {
                'shape': self.tfidf_matrix.shape,
                'density': self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]),
                'non_zero_count': self.tfidf_matrix.nnz
            }
        return {}
    
class PCA_Reducer:
    def __init__(self, n_components=100):
        self.n_components = n_components
        # Use TruncatedSVD which supports sparse inputs directly
        # This is more efficient than PCA for TF-IDF sparse matrices
        self.pca = TruncatedSVD(n_components=n_components, random_state=42)
        self.explained_variance = 0
        self.components = None
        
    def fit_transform(self, tfidf_matrix):
        """
        Fit TruncatedSVD to TF-IDF matrix and transform it
        Returns: SVD-transformed features
        """
        # TruncatedSVD handles sparse matrices directly, no need to convert to dense
        pca_features = self.pca.fit_transform(tfidf_matrix)
        self.explained_variance = self.pca.explained_variance_ratio_.sum()
        self.components = self.pca.components_
        
        return pca_features
    
    def transform(self, tfidf_matrix):
        """
        Transform new TF-IDF matrix using already-fitted TruncatedSVD
        This method should be called after fit_transform() to transform new data
        
        Parameters:
        - tfidf_matrix: TF-IDF matrix to transform (should have same feature dimensions as training data)
        
        Returns: SVD-transformed features
        """
        # TruncatedSVD handles sparse matrices directly
        pca_features = self.pca.transform(tfidf_matrix)
        
        return pca_features
    
    def get_variance_info(self):
        """Get TruncatedSVD variance information"""
        if self.pca is None:
            return {}
        
        return {
            'original_features': self.pca.n_features_in_,
            'components_kept': self.n_components,
            'variance_explained': self.explained_variance * 100,
            'variance_by_component': self.pca.explained_variance_ratio_.tolist()
        }
    
    def get_top_features_per_component(self, feature_names, component_idx=0, top_n=10):
        """
        Get top contributing features for a principal component
        """
        if self.components is None:
            return []
        
        # Get weights for specified component
        component_weights = self.components[component_idx]
        
        # Get indices of top features
        top_indices = np.argsort(component_weights)[-top_n:][::-1]
        
        # Return feature names and weights
        return [(feature_names[idx], component_weights[idx]) for idx in top_indices]
