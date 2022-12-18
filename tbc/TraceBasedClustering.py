# -*- coding: utf-8 -*-

# Author:
#    Antonio López Martínez-Carrasco <antoniolopezmc1995@gmail.com>

"""This file contains the implementation of Trace-Based Clustering technique.
"""

from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from seaborn import heatmap
import matplotlib.pyplot as plt
from typing import Union

class TraceBasedClustering (object):
    """This class represents Trace-Based Clustering technique.

    :param k: number of partitions to generate.
    :param clustering_algorithm: traditional clustering algorithm to use. Possible values: 'kmeans'.
    :param match_function: match function to use. Possible values: 'jaccard2' (coeficient of Jaccard2), 'dice' (coeficient of Dice).
    :param mean_greater_or_equal_than: minimum mean of a column from the matrix of matches to mine the corresponding cluster.
    :param median_greater_or_equal_than: minimum median of a column from the matrix of matches to mine the corresponding cluster.
    :param criterion: measures with which compare when mining the final candidate clusters. Possible values: 'mean_and_median' (mean and median), 'mean_or_median' (mean or median), 'only_mean', 'only_median' and 'no_measure' (mine all clusters no matter tha mean or median).
    :param random_seed: random seed to generate random numbers.
    :param save_matrices_to_file: path to save both matrix of traces (T) and matrix of matches (J) joined in a single matrix (as a .csv file). Each value of this single matrix is a 2-tuple with the corresponding values from both matrices.
    """

    def __init__(self, k : int, clustering_algorithm : str, match_function : str, mean_greater_or_equal_than : float, median_greater_or_equal_than : float, criterion : str = 'no_measure', random_seed : int = 1, save_matrices_to_file : Union[str,None] = None) -> None:
        
        available_values_for_clustering_algorithm = ['kmeans']
        available_values_for_match_function = ['jaccard2', 'dice']
        available_values_for_criterion = ['mean_and_median', 'mean_or_median', 'only_mean', 'only_median', 'no_measure']
        if type(k) is not int:
            raise TypeError("Parameter 'k' must be an integer (type 'int').")
        if type(clustering_algorithm) is not str:
            raise TypeError("Parameter 'clustering_algorithm' must be a string (type 'str').")
        if clustering_algorithm not in available_values_for_clustering_algorithm:
            raise ValueError("Parameter 'clustering_algorithm' is not valid (see documentation).")
        if type(match_function) is not str:
            raise TypeError("Parameter 'match_function' must be a string (type 'str').")
        if match_function not in available_values_for_match_function:
            raise ValueError("Parameter 'match_function' is not valid (see documentation).")
        if type(mean_greater_or_equal_than) is not float:
            raise TypeError("Parameter 'mean_greater_or_equal_than' must be a float.")
        if type(median_greater_or_equal_than) is not float:
            raise TypeError("Parameter 'median_greater_or_equal_than' must be a float.")
        if type(criterion) is not str:
            raise TypeError("Parameter 'criterion' must be a string (type 'str').")
        if criterion not in available_values_for_criterion:
            raise ValueError("Parameter 'criterion' is not valid (see documentation).")
        if type(random_seed) is not int:
            raise TypeError("Parameter 'random_seed' must be an integer (type 'int').")
        if ((type(save_matrices_to_file) is not str) and (save_matrices_to_file is not None)):
            raise TypeError("The type of the parameter 'save_matrices_to_file' must be 'str' or 'NoneType'.")
        self.k = k
        self.clustering_algorithm = clustering_algorithm
        self.match_function = match_function
        self.mean_greater_or_equal_than = mean_greater_or_equal_than
        self.median_greater_or_equal_than = median_greater_or_equal_than
        self.criterion = criterion
        self.random_seed = random_seed
        self.save_matrices_to_file = save_matrices_to_file
        ##
        self._set_of_partitions = dict()
        self._matrix = DataFrame()
        self._list_of_final_candidate_clusters = []

    def _get_set_of_partitions(self) -> dict[int, list[list[int]]]:
        return self._set_of_partitions

    def _get_matrix(self) -> DataFrame:
        return self._matrix

    def _get_list_of_final_candidate_clusters(self) -> list[tuple[str,float,float]]:
        return self._list_of_final_candidate_clusters

    set_of_partitions = property(_get_set_of_partitions, None, None, "Set of partitions from the input pandas dataframe in form of python dictionary obtained after executing Trace-based clustering technique (before executing the 'fit' method, the python dictionary is empty.")
    matrix = property(_get_matrix, None, None, "Both matrix of traces (T) and matrix of matches (J) as a pandas DataFrame obtained after executing Trace-based clustering technique (before executing the 'fit' method, the pandas DataFrame is empty). Each value from the pandas DataFrame is a 2-tuple with the corresponding values from both matrices.")
    list_of_final_candidate_clusters = property(_get_list_of_final_candidate_clusters, None, None, "List of final candidate clusters obtained after executing Trace-based clustering technique (before executing the 'fit' method, the list is empty.")

    def _generateSetOfPartitions(self, pandas_dataframe : DataFrame) -> dict[int, list[list[int]]]:
        """Method to generate the set of partitions from the input pandas dataframe (in this case, in form of python dictionary).

        :param pandas_dataframe: input pandas dataframe. It MUST NOT contain missing values.
        :return: a dictionary in which every pair key-value corresponds to a partition and its clusters. A key with the number 'n' corresponds to the partition with 'n' clusters (i.e., keys go from 2 to k). Its value is a list with the clusters of this partition. Moreover, a cluster is a list with the row numbers from the pandas dataframe passed by parameter.
        """
        if type(pandas_dataframe) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        # Final dictionary.
        dictionary_of_partitions = dict()
        if self.clustering_algorithm == 'kmeans':
            # The random seed will be different in every calling to clustering algorithm.
            current_random_seed = self.random_seed
            # Generate all partitions (from 'number_of_cluster=2' to 'number_of_cluster=k').
            for number_of_clusters in range(2, self.k+1):
                # Create and run KMeans.
                kmeans_alg = KMeans(n_clusters=number_of_clusters, random_state=current_random_seed).fit(pandas_dataframe)
                # For the current partition, generate all its clusters (from 0 to 'number_of_clusters - 1').
                list_of_clusters = [] # List of lists.
                for cluster in range(0,number_of_clusters):
                    sub_dataframe = pandas_dataframe[kmeans_alg.labels_ == cluster] # Dataframe with only the elements/rows of the original pandas dataframe in the cluster number 'cluster'.
                    list_of_clusters.append( sub_dataframe.index.tolist() ) # Get the list of indexes of the sub_dataframe and add them to the list 'list_of_clusters'. 
                # current_random_seed + 3
                current_random_seed = current_random_seed + 3
                # Add partition to the dictionary.
                dictionary_of_partitions[number_of_clusters] = list_of_clusters
        else:
            raise ValueError("The clustering algorithm (" + self.clustering_algorithm + ") is not valid (see documentation).")
        return dictionary_of_partitions

    def _generateMatrix(self, set_of_partitions : dict[int, list[list[int]]]) -> DataFrame:
        """Method to generate both matrix of traces (T) and matrix of matches (J) as a pandas DataFrame from the set of partitions (returned by the function 'generateSetOfPartitions').

        :param set_of_partitions: set of partitions returned by the function 'generateSetOfPartitions'.
        :return: the matrix of traces (T) and the matrix of matches (J) joined in the same pandas DataFrame. Each value is a 2-tuple with the corresponding values from both matrices.
        """
        if type(set_of_partitions) is not dict:
            raise TypeError("Parameter 'set_of_partitions' must be a python dictionary (type 'dict').")
        # Final pandas DataFrame.
        matrix = DataFrame()
        # Iterate over the partition with more clusters ('number_of_clusters_of_partition_k = k'): from cluster 0 to cluster 'number_of_clusters_of_partition_k - 1'.
        number_of_clusters_of_partition_k = self.k
        for cluster_of_partition_k in range(0,number_of_clusters_of_partition_k):
            # Get the cluster (names/numbers of the rows of the original pandas dataframe) as a python set (type 'set').
            cluster_of_partition_k_as_a_python_set = set(set_of_partitions[self.k][cluster_of_partition_k])
            # Temporal list (list of values of the new attribute of the final pandas dataframe).
            temporal_list = []
            # Iterate over the set of previous partitions to k: from partition 2 (with 2 clusters) to partition k-1 (with k-1 clusters).
            number_of_previous_partitions = self.k-1
            for current_partition in range(2,number_of_previous_partitions+1):
                # Maximum value of match between 'cluster_of_partition_k' and the current cluster of the current partition.
                max_value_of_match = -1 # Initially -1. IMPORTANT: This value is impossible and it will be always overwritten (in other case, some error has occurred).
                cluster_with_max_value_of_match = None # Initially None. This value is impossible and it will be always overwritten (in other case, some error has occurred).
                # Iterate over the current partition: from cluster 0 to cluster 'number_of_clusters_of_current_partition - 1'.
                number_of_clusters_of_current_partition = current_partition
                for current_cluster in range(0,number_of_clusters_of_current_partition):
                    # Get the cluster (names/numbers of the rows of the original pandas dataframe) as a python set (type 'set').
                    cluster_of_current_partition_as_a_python_set = set(set_of_partitions[current_partition][current_cluster])
                    ## COMPUTE THE MATCH FUNCTION ##
                    if (self.match_function == 'jaccard2'):
                        # Obtain the intersection between the elements of 'cluster_of_partition_k_as_a_python_set' and elements of 'cluster_of_current_partition_as_a_python_set'.
                        intersection = cluster_of_partition_k_as_a_python_set.intersection(cluster_of_current_partition_as_a_python_set)
                        # Match function.
                        value_of_match = len(intersection) / len(cluster_of_current_partition_as_a_python_set)
                        # Compare with the maximum match value.
                        if (value_of_match > max_value_of_match):
                            max_value_of_match = value_of_match
                            cluster_with_max_value_of_match = current_cluster
                    elif (self.match_function == 'dice'):
                        # Obtain the intersection between the elements of 'cluster_of_partition_k_as_a_python_set' and elements of 'cluster_of_current_partition_as_a_python_set'.
                        intersection = cluster_of_partition_k_as_a_python_set.intersection(cluster_of_current_partition_as_a_python_set)
                        # Match function.
                        value_of_match = (2*len(intersection)) / (len(cluster_of_partition_k_as_a_python_set)+len(cluster_of_current_partition_as_a_python_set))
                        # Compare with the maximum match value.
                        if (value_of_match > max_value_of_match):
                            max_value_of_match = value_of_match
                            cluster_with_max_value_of_match = current_cluster
                    else:
                        raise ValueError("The match function (" + self.match_function + ") is not valid (see documentation).")
                # Assign 'max_value_of_match' and 'cluster_with_max_value_of_match' to 'temporal_list'.
                temporal_list.append( (max_value_of_match, 'cluster'+str(cluster_with_max_value_of_match)) )
            # Add a new attribute to 'matrix'.
            matrix['partition'+str(self.k)+'_cluster'+str(cluster_of_partition_k)] = Series(temporal_list)
        # Change the names of the rows to: 'partition2', 'partition3', ..., partition{k-1}.
        # - Currently, the names of the rows are: 0, 1, 2, ..., k-3.
        matrix.rename(index=(lambda x: 'partition'+str(x+2)), inplace=True)
        # Save the final matrix.
        if self.save_matrices_to_file:
            matrix.to_csv(self.save_matrices_to_file, index_label="row_name") # In this case, the row names are also stored (because they are not only numbers and contain important information).
        return matrix

    def _visualizeMatrixOfMatches(self, matrix : DataFrame, figsize : Union[tuple[int, int], None] = None, save_to_file : Union[str, None] = None) -> 'matplotlib.pyplot.axes':
        """Method to visualize the matrix of matches (using a heat map).

        :param matrix: matrix returned by the function 'generateMatrix' (the matrix of matches will be obtained from it).
        :param figsize: tuple of the form (width in inches, height in inches) used by matplotlib.
        :param save_to_file: path to save the heat map (as a .png file).
        :return: a heat map representing the matrix of matches.
        """
        if type(matrix) is not DataFrame:
            raise TypeError("Parameter 'matrix' must be a pandas DataFrame.")
        # We obtain the matrix of matches from the input matrix.
        matrix_of_matches = matrix.applymap( lambda value : value[0] )
        # To avoid long names in the final heat map, we are going to change the names of the rows and the names of the columns.
        matrix_of_matches.index = ['p'+str(i) for i in range(2,self.k)] # From partition 2 (with 2 clusters) to partition k-1 (with k-1 clusters).
        matrix_of_matches.columns = ['p'+str(self.k)+'_c'+str(i) for i in range(0,self.k)] # From cluster 0 to cluster 'number_of_clusters_of_partition_k - 1' ('number_of_clusters_of_partition_k' = k).
        # Create the heat map.
        plt.clf() # If there are previous data, clean them.
        if (figsize != None):
            plt.figure(figsize=figsize)
        heat_map_axes = heatmap(matrix_of_matches, cbar_kws = dict(use_gridspec=False,location="left"))
        heat_map_axes.set_position([heat_map_axes.get_position().x0, heat_map_axes.get_position().y0, heat_map_axes.get_position().width * 0.85, heat_map_axes.get_position().height * 0.85])
        heat_map_axes.xaxis.tick_top()
        heat_map_axes.yaxis.tick_right()
        for tick in heat_map_axes.get_xticklabels():
            tick.set_rotation(90)
        for tick in heat_map_axes.get_yticklabels():
            tick.set_rotation(0)
        # Save the heat map.
        if save_to_file:
            heat_map_axes.get_figure().savefig(save_to_file)
        return heat_map_axes

    def _mineFinalCandidateClusters(self, matrix : DataFrame) -> list[tuple[str,float,float]]:
        """Method to mine the final candidate clusters according to the mean and/or median of the columns from the matrix of matches.
        
        :param matrix: matrix returned by the function 'generateMatrix' (the matrix of matches will be obtained from it).
        :return: a python list in which each element is tuple formed by: (1) the cluster name, (2) its mean value, and (3) its median value.
        """
        if type(matrix) is not DataFrame:
            raise TypeError("Parameter 'matrix' must be a pandas DataFrame.")
        
        # Final list.
        final_list = []
        # We obtain the matrix of matches from the input matrix.
        matrix_of_matches = matrix.applymap( lambda value : value[0] )
        # Iterator over the columns of 'matrix_of_matches'.
        if (self.criterion == 'mean_and_median'):
            for column in matrix_of_matches.columns:
                # Get the current column as a Series.
                current_column = matrix_of_matches.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= self.mean_greater_or_equal_than) and (current_column_median >= self.median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (self.criterion == 'mean_or_median'):
            for column in matrix_of_matches.columns:
                # Get the current column as a Series.
                current_column = matrix_of_matches.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= self.mean_greater_or_equal_than) or (current_column_median >= self.median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (self.criterion == 'only_mean'):
            for column in matrix_of_matches.columns:
                # Get the current column as a Series.
                current_column = matrix_of_matches.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_mean >= self.mean_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (self.criterion == 'only_median'):
            for column in matrix_of_matches.columns:
                # Get the current column as a Series.
                current_column = matrix_of_matches.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Check the condition.
                if (current_column_median >= self.median_greater_or_equal_than):
                    final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        elif (self.criterion == 'no_measure'):
            for column in matrix_of_matches.columns:
                # Get the current column as a Series.
                current_column = matrix_of_matches.loc[:, column]
                # Get mean and median.
                current_column_mean = current_column.mean()
                current_column_median = current_column.median()
                # Add to the list.
                final_list.append( (column, "mean="+str(current_column_mean), "median="+str(current_column_median)) )
        else:
            raise ValueError("The criterion (" + self.criterion + ") is not valid (see documentation).")
        return final_list

    def fit(self, pandas_dataframe : DataFrame) -> None:
        """Main method to use Trace-based clustering technique.

        :param pandas_dataframe: input pandas dataframe. It MUST NOT contain missing values.
        """
        self._set_of_partitions = self._generateSetOfPartitions(pandas_dataframe)
        self._matrix = self._generateMatrix(self._set_of_partitions)
        self._list_of_final_candidate_clusters = self._mineFinalCandidateClusters(self._matrix)

    def visualizeMatrixOfMatches(self, figsize : Union[tuple[int, int], None] = None, save_to_file : Union[str, None] = None) -> 'matplotlib.pyplot.axes':
        """Method to visualize the matrix of matches (using a heat map). This method has to be called after executing 'fit' method.

        :param figsize: tuple of the form (width in inches, height in inches) used by matplotlib.
        :param save_to_file: path to save the heat map (as a .png file).
        :return: a heat map representing the matrix of matches.
        """
        return self._visualizeMatrixOfMatches(self._matrix, figsize, save_to_file)
