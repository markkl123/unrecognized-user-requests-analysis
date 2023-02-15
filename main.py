# Basic
import json
import numpy as np
import pandas as pd
import re
from time import time
from collections import Counter

# Scikit-learn
from sklearn.decomposition import PCA

# NLTK
from nltk.util import ngrams

# Gensim
from gensim.utils import simple_preprocess
from gensim.parsing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

# Transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection

# Utils
from compare_clustering_solutions import evaluate_clustering


def cluster_requests(all_requests, min_size):
    print(f'cluster {len(all_requests)} requests, with minimum of {min_size} requests per cluster')

    # Convert requests to embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(all_requests, convert_to_tensor=True)

    # Perform clustering
    grouped_clusters = community_detection(embeddings, min_community_size=min_size, threshold=0.631)

    # Construct cluster id's list
    clusters = np.full(len(embeddings), -1)
    for cluster_id, embedding_ids in enumerate(grouped_clusters, start=1):
        clusters[embedding_ids] = cluster_id

    return embeddings, clusters


def extract_representatives(requests, embeddings, num_rep):
    """
        TODO: options to consider
        1. transform embeddings
        2. select according to dimensions
    """
    components = PCA(n_components=num_rep).fit(embeddings).components_

    return list(set([
        requests[np.argmin(np.linalg.norm(embeddings - component, axis=1))]
        for component in components
    ]))


def extract_cluster_representatives(all_requests, all_embeddings, all_clusters, num_rep):
    print(f'choose {num_rep} representatives for each cluster')

    return {
        cluster_num: extract_representatives(all_requests[all_clusters == cluster_num],
                                             all_embeddings[all_clusters == cluster_num],
                                             num_rep)
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def create_cluster_dictionary(requests):
    # Remove stop words, punctuation, lower, tokenize
    preprocessed = list(map(simple_preprocess, map(remove_stopwords, requests)))

    # Build dictionary
    dictionary = Dictionary(preprocessed)

    return {word: count for word, count in dictionary.most_common()}


def calc_ngram_score(ngram, num_occurrences, word_counts):
    # (LENGTH + 1) * NUM_OCCURRENCES * WORD_COUNTS / (NUM_STOPWORDS + 1)
    return ((len(ngram) + 1)
            * num_occurrences
            * np.prod([word_counts.get(word, 1) for word in ngram])
            / (np.sum([word in STOPWORDS for word in ngram]) + 1))


def rank_ngrams(requests, word_counts):
    # Remove punctuation, lower, split
    preprocessed = [re.sub(r'[^\w\s?!-]', '', req.lower()).split() for req in requests]

    # calculating all ngram scores
    ngram_counter = Counter()

    # Count ngram occurrences
    for n in range(2, 5):
        for req in preprocessed:
            ngram_counter.update(map(tuple, ngrams(req, n)))

    # Convert to ngram scores
    return {
        ngram: calc_ngram_score(ngram, count, word_counts)
        for ngram, count in ngram_counter.most_common()
    }


def construct_name(requests):
    # Count non-stopwords occurrences
    word_counts = create_cluster_dictionary(requests)

    # Rank ngram scores
    ngram_ranking = sorted(list(rank_ngrams(requests, word_counts).items()),
                           key=lambda ngram: ngram[1],
                           reverse=True)

    # Construct name from best ngram
    return ' '.join(ngram_ranking[0][0])


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

    start = time()

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # evaluate clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])

    print(f'finished: {time() - start:.2f} seconds')
