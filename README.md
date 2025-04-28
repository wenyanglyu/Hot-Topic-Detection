This repository contains code for analyzing and detecting topics in blog data using various natural language processing and topic modeling techniques, as described in the research paper "Hot Topic Detection with Topic Modeling Methods" by Wenyang Lyu, Henry Hu, and Parma Nand.

## Overview

This project applies several topic modeling approaches to a large dataset of 19,320 XML-formatted blog files from an anonymous blogging site, covering a period from 2001 to 2004. The blogs are segmented by demographic categories (gender, age, and student status). The implementation includes comprehensive pre-processing of text data, application of multiple topic extraction methods, and visualization of the discovered topics.

![NMF-Topic](https://github.com/user-attachments/assets/a28bbe7e-5c07-48a8-9bad-841929635c82)

The aim is to identify the two most popular topics discussed within specific demographics:
- Males
- Females 
- Age less than 20
- Age over 20
- Students
- General population

## Features

- **Data Preparation**: Unzips and segments blog data into demographic categories
- **Text Pre-processing**: Implements robust text cleaning including:
  - Lowercasing text
  - Replacing non-ASCII characters
  - Tokenization
  - Removing stop words and punctuation
  - Spell checking
  - Stemming and lemmatization
  - Parallel processing with Dask for large datasets
- **Topic Detection Methods**:
  - Noun counting
  - Grammatical role analysis (subjects, direct objects, prepositional objects)
  - TF-IDF analysis
  - n-Gram extraction
  - Latent Dirichlet Allocation (LDA) with optimal parameter tuning
  - Non-negative Matrix Factorization (NMF)
- **Visualization**: 
  - Word clouds for topic visualization
  - Termite plots for topic term distribution
  - Topic distribution plots for most dominant topics

## Theoretical Background

The project applies two main topic modeling techniques:

- **Latent Dirichlet Allocation (LDA)**: A generative probabilistic model that treats documents as mixtures of topics and topics as mixtures of words.

- **Non-negative Matrix Factorization (NMF)**: A matrix decomposition method that breaks down document-term matrices into components representing topics.

Topic quality is evaluated using coherence scores, which measure the semantic similarity between words in a topic.

Where NPMI is the normalized pointwise mutual information between words.

## Methods Compared

The project compares several topic modeling approaches:

1. **Counting Nouns**: Identifies the most frequent nouns in the corpus
2. **Grammatical Role Analysis**: Counts subjects, direct objects, and prepositional objects
3. **TF-IDF Analysis**: Identifies important terms using TF-IDF scoring
4. **n-Gram Analysis**: Extracts common bi-grams and tri-grams
5. **LDA with CountVectorizer**: Tests various numbers of topics (10, 15, 20, 30, 40, 50, 60)
6. **NMF with TF-IDF Vectorizer**: Tests various numbers of topics (10, 15, 20, 40)

## Key Findings

- **LDA Performance**: Best coherence score (0.52) achieved with 20 topics, contrary to common belief that higher topic numbers yield better coherence
- **NMF Performance**: Best coherence score (0.59) achieved with 15 topics
- **Optimal Approach**: NMF with TF-IDF vectorization provided the most coherent and interpretable topics
- **Demographic Insights**: Each demographic showed distinct dominant topics:
  - Students: Internet technologies; Daily life reflection (0.5986 coherence)
  - Males: Job and mental health; Travel experiences (0.6036 coherence)
  - Females: Love and emotional journey; Urban life and reflections (0.5766 coherence)
  - Age > 20: Pressure and political critique; Heartbreak and emotional struggles (0.6602 coherence)
  - Age ≤ 20: Heartbreak and love; Daily activities and reflections (0.6034 coherence)
  - Everyone: Life events and reflections; Citizenship and societal expectations (0.6101 coherence)

## Installation

Install all required packages using pip:

```bash
pip install nltk spacy unidecode pyspellchecker "dask[distributed]" pyLDAvis gensim scikit-learn matplotlib seaborn textblob wordcloud
```

You'll also need to download necessary NLTK and spaCy resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import spacy
spacy.cli.download('en_core_web_sm')
```

## Usage

The notebook is organized into six main parts:

1. **Dataset Preparation**: Loading and unzipping files
2. **Pre-Processing**: Text cleaning and preparation
3. **NMF with TF-IDF**: Topic detection by demographic
4. **Basic Topic Detection**: Using noun counting, TF-IDF, n-Gram methods
5. **LDA Optimization**: Testing various topic numbers with CountVectorizer
6. **NMF Optimization**: Testing various topic numbers with TF-IDF

## Methodology Workflow

1. **Data Segmentation**: Files are categorized by demographics based on file naming patterns
2. **Pre-processing**: Text is cleaned and standardized through multiple steps
3. **Topic Modeling**: Different methods are applied to extract topics
4. **Clause Extraction**: Top documents containing dominant topics are extracted for analysis
5. **Topic Interpretation**: Word clouds and human judgment are used to interpret topics

## Reflections and Lessons Learned

- Understanding the theoretical principles of methods before parameter tuning is crucial
- NMF produces more specific and actionable topics compared to counting methods and LDA
- Topic coherence doesn't necessarily increase with higher topic numbers
- A combination of automated metrics and human judgment provides the best topic evaluation

## References

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation.
2. Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization.
3. Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures.

## Acknowledgments

The implementation is based on popular NLP libraries and techniques in topic modeling literature.
