import json
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from transformers import pipeline, set_seed

from compare_clustering_solutions import evaluate_clustering
from collections import Counter
import hdbscan

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

def cluster_requests(all_requests, min_size):
    """
        TODO: options to consider
        1. find better algorithm
        2. include UMAP and search for optimal parameters
        3. TF-IDF
            - remove stop words & punctuation
            - use lemmatization
    """
    print(f'cluster {len(all_requests)} requests, with minimum of {min_size} requests per cluster')

    # Convert requests to embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(all_requests)

    # Perform clustering
    # clusters = np.array(hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean').fit(embeddings).labels_)
    clusters = np.array(DBSCAN(eps=0.7, min_samples=5).fit(embeddings).labels_)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Remove clusters that are too small
    for cluster_num in range(n_clusters):
        cluster_mask = clusters == cluster_num

        if cluster_mask.sum() < min_size:
            clusters[cluster_mask] = -1

    return embeddings, clusters


def extract_representatives(requests, embeddings, num_rep):
    """
        TODO: options to consider
        1. PCA as a starting point looks pretty good
    """
    components = PCA(n_components=num_rep).fit(embeddings).components_

    return [
        requests[np.argmin(np.linalg.norm(embeddings - component, axis=1))]
        for component in components
    ]


def extract_cluster_representatives(all_requests, all_embeddings, all_clusters, num_rep):
    print(f'choose {num_rep} representatives for each cluster')

    return {
        cluster_num: extract_representatives(all_requests[all_clusters == cluster_num],
                                             all_embeddings[all_clusters == cluster_num],
                                             num_rep)
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def construct_name(requests: list[str]):
    article = ''


    stemmer = SnowballStemmer("english")

    def preprocess(text):
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 2:
                result.append(token)
        return ' '.join(result)

    for s in requests:
        article += (preprocess(s) + '. ')

    #print(article)
    # Set the seed for reproducibility
    set_seed(42)

    # Load the summarization model from the pipeline
    summarization_model = pipeline("summarization",model='facebook/bart-large-cnn')

    # Generate a summary for the article text
    title = summarization_model(article[:1024], max_length=11, min_length=4, early_stopping=True)[0].get("summary_text")
    title = title.split('?')[0]# + ['?' if '?' in title else ''][0]
    title = title.split('.')[0]# + ['.' if '.' in title else ''][0]
    print(title)
 
    return title
    """
        TODO: options to consider
        1. look for some common ngram that has a probable POS structure
    """
    #return "name"


def construct_cluster_names(all_requests, all_clusters):
    print(f'construct a meaningful name for each cluster')

    return {
        cluster_num: construct_name(all_requests[all_clusters == cluster_num])
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def save_cluster_results(output_file, requests, clusters, rep_dict, names_dict):
    print(f'save results in {output_file}')

    results = {
        "cluster_list": [
            {
                "cluster_name": name,
                "representative_sentences": reps,
                "requests": requests[clusters == cluster_num].tolist()
            }
            for (cluster_num, reps), name in zip(rep_dict.items(), names_dict.values())
        ],
        "unclustered": requests[clusters == -1].tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    print(f'start analyzing unrecognized requests from {data_file}')

    # Read all requests
    requests = pd.read_csv(data_file)['request'].to_numpy()

    # First, try cluster the requests
    embeddings, clusters = cluster_requests(requests, min_size)

    # Then, for each cluster extract the most representative instances
    rep_dict = extract_cluster_representatives(requests, embeddings, clusters, num_rep)

    # Finally, give each cluster a meaning full name
    names_dict = construct_cluster_names(requests, clusters)

    # Store the results
    save_cluster_results(output_file, requests, clusters, rep_dict, names_dict)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # evaluate clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
