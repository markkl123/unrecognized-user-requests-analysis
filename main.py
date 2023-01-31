import json
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
import math as mt
from sklearn.cluster import KMeans

class XMeans:
    def loglikelihood(self, r, rn, var, m, k):
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        l4 = rn * mt.log(rn)
        l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def __init__(self, X, kmax = 20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num

        while(1):
            ok = k

            #Improve Params
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            m = kmeans.cluster_centers_

            #Improve Structure
            #Calculate BIC
            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)

            #Split each cluster into two subclusters and calculate BIC of each splitted cluster
            sk = 2 #The number of subclusters
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))

                kmeans = KMeans(n_clusters=sk).fit(ci)
                ci_labels = kmeans.labels_
                sm = kmeans.cluster_centers_

                for l in range(sk):
                    rn = np.size(np.where(ci_labels == l))
                    var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                    nbic[i] += self.loglikelihood(r, rn, var, M, sk)

                p = sk * (M + 1)
                nbic[i] -= p/2.0*mt.log(r)

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if ok == k or k >= self.KMax:
                break


        #Calculate labels and centroids
        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels = kmeans.labels_
        self.k = k
        self.m = kmeans.cluster_centers_

def cluster_text_data(text_data):
    # Load the Sentence-BERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Convert text data to sentence embeddings using Sentence-BERT
    embeddings = model.encode(text_data)

    # Convert embeddings to a numpy array
    X = np.array(embeddings)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterer.fit(text_data)

    # Return the cluster labels for each request
    return clusterer.labels_

# # Example usage
# text_data = [
#     "this is request 1",
#     "this is request 2",
#     "request 3 is different",
#     "request 4 is similar to request 1",
#     "request 5 is unique",
# ]
# cluster_labels = cluster_text_data(text_data)
# print(cluster_labels)


def x_means(data, k_min, k_max):
    best_k = k_min
    best_inertia = float('inf')

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        if kmeans.inertia_ < best_inertia:
            best_k = k
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    return best_kmeans


# # Example usage
# X = np.array([[2, 2], [1, 1], [3,3], [100, 100],[101,99] ,[42, 50],[45,49],[47,48], [105, 100]])
# k_min = 2
# k_max = 5
# kmeans = x_means(X, k_min, k_max)
# print(kmeans.labels_)
# print(cluster_text_data(X))

def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file
    print(data_file, output_file, num_rep, min_size)
    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
