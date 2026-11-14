
from pathlib import Path
import sys

import joblib
import numpy as np
from gensim.models import Word2Vec, FastText

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Allow importing the existing Assignment 1 cleaning function
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cleaning import clean_job_description


class DocumentVectorizer:

    def __init__(self):
        # Load existing models; no training happens here
        self.tfidf = joblib.load(
            MODELS_DIR / "tfidf_vectorizer.pkl"
        )

        self.cbow = Word2Vec.load(
            str(MODELS_DIR / "word2vec_cbow.model")
        )

        self.skipgram = Word2Vec.load(
            str(MODELS_DIR / "word2vec_skipgram.model")
        )

        self.fasttext = FastText.load(
            str(MODELS_DIR / "fasttext.model")
        )

    def vectorize_document(self, raw_document, representation="tfidf"):

        if not isinstance(raw_document, str):
            raise TypeError("Document must be a string.")

        representation = representation.lower()

        if representation not in (
            "tfidf", "cbow", "skipgram", "fasttext"
        ):
            raise ValueError(
                "Choose tfidf, cbow, skipgram, or fasttext."
            )

        # Reuse the exact cleaning function from Assignment 1
        cleaned_text = clean_job_description(raw_document)

        # TF-IDF returns a sparse document vector
        if representation == "tfidf":
            return self.tfidf.transform([cleaned_text])

        # Choose the requested embedding model
        models = {
            "cbow": self.cbow,
            "skipgram": self.skipgram,
            "fasttext": self.fasttext
        }

        model = models[representation]

        tokens = cleaned_text.split()

        # Standard Word2Vec cannot generate OOV vectors.
        # fastText can construct vectors using subwords.
        if representation == "fasttext":
             tokens = [
                word for word in tokens
                if word in model.wv
    ]
        else:
            tokens = [
                word for word in tokens
                if word in model.wv.key_to_index
            ]

        # Return a zero vector if no usable tokens exist
        if not tokens:
            return np.zeros(
                model.wv.vector_size,
                dtype=np.float32
            )

        # Average the word vectors into one document vector
        word_vectors = [
            model.wv[word] for word in tokens
        ]

        document_vector = np.mean(
            word_vectors,
            axis=0
        )

        return document_vector
