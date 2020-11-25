import numpy as np
from queue import PriorityQueue
import argparse
import random

def _parse_args():
    """
    Command-line arguments to the system. 
    :return: the parsed args bundle
    """
    parser=argparse.ArgumentParser(description='BK.py')
    parser.add_argument('-data')    #input file of data points
    parser.add_argument('-k')       #total number of desired clusters
    parser.add_argument('-s')       #specifies the maximum number of data points
    parser.add_argument('-d')       #specifies the maximum intra-cluster distance allowed
    parser.add_argument('-output')  #specifies the output file name
    args = parser.parse_args()
    return args

def icd(features, center):
    """
    Calculate the intra-cluster distance of a given cluster
    """
    intra_dist = np.array([])
    for f in features:
        distance = np.linalg.norm(center- f)
        intra_dist = np.append(intra_dist, distance)
    return np.mean(intra_dist)

def kmeans(feat):
    """
    Implement k-means algorithm as each bisecting clustering step.
    """
    feat_dim = dim
    # randomlly select initial centers
    center_idx = np.random.choice(feat,replace=False,size=2)
    center = np.array(data[center_idx])
    cluster_id = [i for i in range(2)]
    #store distancce
    intra_cluster = []
    #store clusters
    clusters = []
    #flag
    clusterChanged=True
    while clusterChanged:
        label = np.array([], dtype=int)
        dist_to_center = np.array([])
        for f in feat:
            min_distance = None
            nearest_label = None
            for i in range(2):
                c = center[i]
                distance = np.linalg.norm(data[f]- c)
                # Update the distance that is the smallest 
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    nearest_label = cluster_id[i]
            dist_to_center= np.append(dist_to_center, min_distance)
            label = np.append(label, nearest_label)
        # Calculate the mean of clusters
        cluster_mean = np.array([])
        for lab in cluster_id:
            cluster = np.array([data[i] for i in feat[label == lab]])
            cluster_mean = np.append(cluster_mean, np.mean(cluster, axis=0))
        cluster_mean = np.reshape(cluster_mean, (-1, feat_dim))
        #update the cluster center
        if np.array_equal(cluster_mean, center):
            # give the distance of clusters
            dist_cluster = []
            for i, lab in zip(range(2), cluster_id):
                idx_position = label == lab
                dist_cluster.append((sum(idx_position), idx_position))
            dist_cluster.sort(key=lambda x : x[0])
            for (_, idx_position) in dist_cluster:
                intra_cluster.append(np.mean(dist_to_center[idx_position]))
                clusters.append(feat[idx_position])
            break
        else:
            center = cluster_mean
    return intra_cluster, clusters

class Node:
    """
    Used to handle priority queue 
    """
    #create a priorityqueue used to build up the tree
    tree=PriorityQueue()
    priority_set = set()
    
    def __init__(self, feature,width):
        # Initialize all the variables
        self.data = feature
        self.leftward = None
        self.rightward= None
        self.size = len(feature)
        self.width = width
        #if reach the bottom
        self.isLeaf = False
        #if reach the end of the priorityqueue
        self.end = False       
        self.enqueue(offset=self.size)
    def enqueue(self, offset=0):
        #add one num to the priority queue
        priority = Node.sort(self,size - offset)
        Node.tree.put((priority, self))
        Node.priority_set.add(priority)
    def sort(self,priority):
        if not {priority}.issubset(Node.priority_set):
            return priority
        else:
            return Node.sort(self,priority + 1 / size)

def bisecting():
    """
    bisecting function will do the bisecting clustering and generates the dendrogram as the output
    """
    node = Node.tree.get()[1]
    #three criterias
    if node.isLeaf: 
        #stop if the tree has reached the end of node.
        node.enqueue()
        return  
    if Node.tree.qsize() >= cluster_num: 
        #stop when the tree size is larger than the maximum cluster number limitation.  
        return   
    if node.size < cluster_size:
        #stop when the node size is less than the minimum cluster size limitation
        node.enqueue()
        return 

    if node.width < intra_dist:
        #reaches the leaf
        node.isLeaf = True
        node.enqueue()
        #going down to the next node
        bisecting()

    [left_width, right_width], [left_cluster, right_cluster] = kmeans(node.data)
    
    node.leftward = Node(left_cluster, left_width)
    node.rightward = Node(right_cluster, right_width)
    bisecting() 

def print_tree():
    """
    print out the whole tree
    """
    for k in range(8):
        if storage[k]:
            line = ' '.join(storage[k])
            print(line)


def dendrogram(node, depth=0):
    """
    dendrogram will store all the data point information and other necessary information
    associated with each node in the dendrogram.
    """
    padding = depth
    if not node:
        return
    storage[depth].append(str(node.size))
    dendrogram(node.leftward, padding+1)
    dendrogram(node.rightward, padding+1)

def print_clusters(node, leaf_visited):
    """
    output the cluster index of each input data point of the data.
    The first line in the output file be the cluster index of the first input data point
    the second line in the output file should be the cluster index of the second input data point
    """
    if node.isLeaf:
        for i in node.data:
            label[i] = leaf_visited
        return leaf_visited + 1
    if node.leftward:
        leaf_visited = print_clusters(node.leftward, leaf_visited)
    if node.rightward:
        leaf_visited = print_clusters(node.rightward, leaf_visited)
    return leaf_visited

if __name__ == '__main__':
    #initialing variables given by the user
    args=_parse_args()
    data = str(args.data)   #input data file name and location
    cluster_num = int(args.k)   #maximum cluster number allowed
    cluster_size = int(args.s)  #minimum cluster size allowed
    intra_dist = float(args.d)  #maximum intra-cluster distance allowed
    output = str(args.output)   #output file name and location
    data = np.genfromtxt(data, delimiter=' ')
    size = data.shape[0]    
    dim = data.shape[1]
    feat = np.array([i for i in range(size)], dtype=int)
    label = {i: 0 for i in range(size)}
    Node.tree=PriorityQueue()
    storage = {i:[] for i in range(8)}
    root = Node(feat,200)
    bisecting()
    dendrogram(root)
    print_tree()
    while not Node.tree.empty():
        node = Node.tree.get()[1]
        node.isLeaf = True
    total_leaf_node = print_clusters(root, 0)
    labels = [v for _,v in label.items()]
    np.savetxt(output, labels, delimiter=' ')

