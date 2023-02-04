import json
import re
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torch import cosine_similarity

from compare_clustering_solutions import evaluate_clustering
from collections import Counter
import hdbscan

def cluster_requests(all_requests, min_size):
    """
        TODO: options to consider
        1. find better algorithm
        2. preprocess the sentences:
            - remove stop words
            - use lemmatization
            - remove punctuation
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
    for s in requests:
        article += (s + '. ')

    from transformers import pipeline, set_seed

    # Set the seed for reproducibility
    set_seed(42)

    # Load the summarization model from the pipeline
    summarization_model = pipeline("summarization",model='facebook/bart-large-cnn')

    # Generate a summary for the article text
    title = summarization_model(article[:1024], max_length=20, min_length=5, early_stopping=True)[0].get("summary_text")
    title = title.split('?')[0] + ['?' if '?' in title else ''][0]
    title = title.split('.')[0] + ['.' if '.' in title else ''][0]
    return title
    """
        TODO: options to consider
        1. trigrams alone is not so good
        2. noun chunks works ok if the noun is repeated often, otherwise it misses. 
    """
    def is_meaningful_phrase(phrase_span):
        return sum([token.is_stop for token in phrase_span]) / len(phrase_span) < 0.5

    nlp = spacy.load("en_core_web_sm")

    phrases = Counter()
    for req in requests:
        phrases.update([c.text for c in nlp(req).noun_chunks if is_meaningful_phrase(c)])

    if len(phrases.most_common()) == 0:
        NGRAM = 3
        for req in requests:
            words = re.sub(r'[^\w\s\'-]', '', req).split()
            ngrams = [' '.join(words[i:i + NGRAM]) for i in range(len(words) - (NGRAM - 1))]
            phrases.update(ngrams)

    return phrases.most_common(1)[0][0]


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
