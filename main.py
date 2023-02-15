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
    """
    Apply clustering with unknown number of clusters, and tolerate outliers.

    :param all_requests: list of strings, requests.
    :param min_size: number,  minimum cluster size.
    :return: tuple, with a list of request embeddings, and a list of cluster id's.
    """
    print(f'cluster {len(all_requests)} requests, with minimum of {min_size} requests per cluster')

    # Convert requests to embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(all_requests, convert_to_tensor=True)

    # Perform clustering
    grouped_clusters = community_detection(embeddings, min_community_size=min_size, threshold=0.631)

    # Construct clusters list
    clusters = np.full(len(embeddings), -1)
    for cluster_id, embedding_ids in enumerate(grouped_clusters, start=1):
        clusters[embedding_ids] = cluster_id

    return embeddings, clusters


def extract_representatives(requests, embeddings, num_rep):
    """
    Use PCA to identify variance in the embeddings vector space.

    :param requests: list of strings, requests.
    :param embeddings: list of vectors, request embeddings.
    :param num_rep: number, desired number of representatives.
    :return: list of up to 'num_rep' requests, i.e., representatives.
    """
    # Extract principal components, to identify variance
    components = PCA(n_components=num_rep).fit(embeddings).components_

    # Choose the embeddings that are closest to the components
    return list(set([
        requests[np.argmin(np.linalg.norm(embeddings - component, axis=1))]
        for component in components
    ]))


def extract_cluster_representatives(all_requests, all_embeddings, all_clusters, num_rep):
    """
    For each cluster, extract representative requests, satisfying the property of diversity.

    :param all_requests: list of strings, requests.
    :param all_embeddings: list of vectors, request embeddings.
    :param all_clusters: list of numbers, clusters.
    :param num_rep: number, desired number of representatives.
    :return: dictionary, cluster id to list of cluster representatives.
    """
    print(f'choose {num_rep} representatives for each cluster')

    # Extract representatives for every cluster
    return {
        cluster_num: extract_representatives(all_requests[all_clusters == cluster_num],
                                             all_embeddings[all_clusters == cluster_num],
                                             num_rep)
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def create_cluster_dictionary(requests):
    """
    Create a word-count dictionary for the meaningful words in the cluster

    :param requests: list of strings, the requests
    :return: dictionary, word to count.
    """
    # Remove stop words, punctuation, lower, tokenize
    preprocessed = map(simple_preprocess, map(remove_stopwords, requests))

    # Build dictionary
    dictionary = Dictionary(preprocessed)

    # Construct word-count dictionary
    return {word: count for word, count in dictionary.most_common()}


def calc_ngram_score(ngram, num_occurrences, word_counts):
    """
    Calculate the ngram score in its cluster.
    the formula:

        given ngram with a subset of meaningful words: {WORD_1, WORD_2, ..., WORD_N}

        (NGRAM_LENGTH + 1) * NGRAM_APPEARANCES * (WORD_COUNT_1 * WORD_COUNT_2 * ... * WORD_COUNT_N)
        =====================================================================================
                                (NUM_STOPWORDS_IN_NGRAM + 1)

        NOTE: '+ 1' was added to avoid 0-division

    :param ngram: list of string, words in ngram
    :param num_occurrences: number, appearances of ngram in cluster
    :param word_counts: dictionary, word-count of the cluster
    :return: number, ngram score
    """
    # Calc ngram score
    return ((len(ngram) + 1)
            * num_occurrences
            * np.prod([word_counts.get(word, 1) for word in ngram])
            / (np.sum([word in STOPWORDS for word in ngram]) + 1))


def rank_ngrams(requests, word_counts):
    """
    Calculate all ngrams scores in the cluster.

    :param requests: list of strings, requests.
    :param word_counts: dictionary, word-count of the cluster
    :return: dictionary, ngram to score
    """
    # Remove punctuation, lower, split
    preprocessed = [re.sub(r'[^\w\s?!-]', '', req.lower()).split() for req in requests]

    # Count ngram occurrences
    ngram_counter = Counter()
    for n in range(2, 5):
        for req in preprocessed:
            ngram_counter.update(map(tuple, ngrams(req, n)))

    # Convert to ngram scores
    return {
        ngram: calc_ngram_score(ngram, count, word_counts)
        for ngram, count in ngram_counter.most_common()
    }


def construct_name(requests):
    """
    Construct a meaningful name to the given cluster.

    :param requests: list of strings, requests.
    :return: string, meaningful name.
    """
    # Count non-stopwords occurrences
    word_counts = create_cluster_dictionary(requests)

    # Rank ngram scores
    ngram_ranking = sorted(list(rank_ngrams(requests, word_counts).items()),
                           key=lambda ngram: ngram[1],
                           reverse=True)

    # Construct name from best ngram
    return ' '.join(ngram_ranking[0][0])


def construct_cluster_names(all_requests, all_clusters):
    """
    For each cluster, Construct a meaningful name, satisfying fluency.

    :param all_requests: list of strings, requests.
    :param all_clusters: list of numbers, clusters.
    :return: dictionary, cluster id to name
    """
    print(f'construct a meaningful name for each cluster')

    # Construct name for every cluster
    return {
        cluster_num: construct_name(all_requests[all_clusters == cluster_num])
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def save_cluster_results(output_file, requests, clusters, rep_dict, names_dict):
    """
    Save results in output file.

    :param output_file: string, output file path.
    :param requests: list of strings, requests.
    :param clusters: list of numbers, clusters.
    :param rep_dict: dictionary, cluster to list of representatives
    :param names_dict: dictionary, cluster to name
    """
    print(f'save results in {output_file}')

    # Construct the final structure of the results
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

    # Store results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    """
    Analyze unrecognized request of a goal-oriented dialog system.

    :param data_file: string, input file path.
    :param output_file: string, output file path.
    :param num_rep: number, desired number of representatives to extract
    :param min_size: number, minimum number of requests in each cluster
    """
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

    # Start timer
    start = time()

    # Analyze requests
    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # evaluate clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])

    # Print execution time
    print(f'finished: {time() - start:.2f} seconds')
