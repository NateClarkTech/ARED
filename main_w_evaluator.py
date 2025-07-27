# NOTE: this runs on MNIST - haven't also gotten parking lot data going yet...\

DATA_SOURCE = "MNIST"

'''
KAPPA: Paranoia Parameter
'''
KAPPA = 2

'''
QS_VAR: Query Strategy Variants
|- 0: Diameter check
|- 1: Approx. Ave Single Linkage Average 
'''
QS_VAR = 0

'''
window_size: size of the data_window window saved by ARED
|- int: larger window size means it remembers more data
|- WARNING: value must be larger than 0
'''
DATA_WINDOW_SIZE = 1000

'''
NUM_POINTS_TO_PROCESS: Number of points in dataset to process
|- -1: process all the data
|- 0-inf: process up to that number if data is available
'''
NUM_POINTS_TO_PROCESS = 2000

'''
VERBOSE_FLAGS: Array of control flags to make ARED loud or quite
|- Array, containing verbose flags for different types of messages
|- Example: VERBOSE_FLAGS = [0, 1, 2]
|- Put these numbers in the array to change which parts of ARED are very loud
|- 0: Prints dataset info and every 1000th loop of data processing
|- 1: Prints when new clusters are created and where clusters are inserted
|- 2: Prints split information, which cluster id and o_pt movements 
|- 3: Prints the labeled_data cluster_id_array and abs_index_array and data windows assigned_cluster_id buffer
|- 4: Prints the forgotten_abs_index and forgotten_point_cluster_id during subspace_partition_maintenance 
'''
VERBOSE_FLAGS = [0] #example setting [1, 2] for two verbose level control flags

# Imports ===================================
from Circular_Buffer import *
from MNIST_Data_Processing import *
from Data_Stream import *
from Oracle import *
from A_RED import *
from sklearn.datasets import fetch_openml
from collections import Counter
from Stats import *
import numpy as np

if __name__ == '__main__':
    if DATA_SOURCE == "MNIST":
        # get data ==============================
        sparsity_levels = [10000 // 2 ** n for n in range(10)]
        n_events = sum(sparsity_levels)
        X_skewed, y_skewed, X_full, y_full = load_and_skew_mnist(sparsity_levels, n_events)
        # Step 2: Identify the 2 least common digits
        digit_counts = Counter(y_skewed)
        least_common_digits = [digit for digit, _ in digit_counts.most_common()[-2:]]

        if 0 in VERBOSE_FLAGS:
            print(f"Running ARED on skewed MNIST dataset with {n_events} events")
            print(f"Least common digits: {least_common_digits} (marked as relevant)")

        # Generate relevance info
        relevance_array = generate_is_relevant(y_skewed, set(least_common_digits))
        y_w_rel = list(zip(y_skewed, relevance_array))

        # Initialize Oracle and ARED ===================================
        data_stream = Data_Stream(X_skewed, y_w_rel)
        oracle = Oracle(X_skewed, y_w_rel)
        ared = ARED(oracle, KAPPA, DATA_WINDOW_SIZE, QS_VAR, VERBOSE_FLAGS)


        # Stream and Process data =========================================
        ared.process_first_point(data_stream.stream_new_data_point())

        if NUM_POINTS_TO_PROCESS == -1:
            points_to_process = data_stream.get_remaining_num_points()
        else:
            points_to_process = NUM_POINTS_TO_PROCESS - 1

        for _ in range(points_to_process):
            if 0 in VERBOSE_FLAGS:
                if not _ % 1000:
                    print(_)
            ared.process_point(data_stream.stream_new_data_point())

        if 0 in VERBOSE_FLAGS:
            print("ARED COMPLETE")
            print(f"kappa = {KAPPA}")
            print(f"Number of points processed = {ared.data_window.abs_idx_max + 1}")
            print(f"number of classes discovered {len(ared.subspace_partition.set_of_known_labels)}")
            print(f"classes discovered {ared.subspace_partition.set_of_known_labels}")
            print(f"Number of queries: {len(ared.labeled_data.abs_idx_array)}")

            num_relevant_points_found = 0
            for cluster in ared.subspace_partition.cluster_list:
                if cluster.relevance:
                    num_relevant_points_found += len(cluster.l_pts) 

            print(f"Relevant points found {num_relevant_points_found}")
            print(f"Relevant point precentage: {num_relevant_points_found/sum(sparsity_levels[-2:])}")
