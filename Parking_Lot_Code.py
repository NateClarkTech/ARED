"""# ARED on Parking lot data"""

# print(features.shape)
# print(labels_df.shape)
# import pickle
# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# from google.colab import drive

# # === Step 1: Mount and load dataset ===
# drive.mount("/content/drive")

# features_path = "/content/drive/MyDrive/Parking_Lot_Data/features.pkl"
# labels_path = "/content/drive/MyDrive/Parking_Lot_Data/labels.csv"

# with open(features_path, 'rb') as f:
#     features = pickle.load(f)  # Expecting a list or array of 128x128 flattened frames

# labels_df = pd.read_csv(labels_path)
# print("Features loaded successfully.")
# print("Labels loaded successfully.")

# drive.flush_and_unmount()
# print("Google Drive unmounted.")

# import pickle
# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# from google.colab import drive
# import numpy as np
# from collections import Counter

# # You should define Oracle, Data_Window, Labeled_Data, Subspace_Partition, Data_Stream before running main.

# def generate_is_relevant(label_list, non_relevant_set):
#     return [label not in non_relevant_set for label in label_list]

# def main():
#     # === STEP 1: Run last cell ===

#     # === Step 2: Prepare Data ===
#     X = np.array(features)  # shape: (n_samples, 16384)
#     y = labels_df["label"].tolist()

#     assert len(X) == len(y), "Feature and label count mismatch!"

#     # Mark relevance: normal, shadow, sticks, line are NOT relevant
#     non_relevant_labels = {"normal", "shadow", "sticks", "line"}
#     relevance_array = generate_is_relevant(y, non_relevant_labels)

#     # (label, relevance)
#     y_w_rel = list(zip(y, relevance_array))

#     # === Step 3: Initialize Oracle and ARED ===
#     data_stream = Data_Stream(X, y_w_rel)
#     oracle = Oracle(X, y_w_rel)
#     ared = ARED(oracle, 1, 300, False)

#     # === Step 4: Stream data ===
#     #print(ared.data_window.abs_idx_max)

#     # feed ARED first data point to start
#     ared.process_first_point(data_stream.stream_new_data_point())
#     for _ in range(data_stream.get_remaining_num_points()):
#         #print(ared.data_window.abs_idx_max)
#         ared.process_point(data_stream.stream_new_data_point())

#     #print(len(ared.subspace_partition.cluster_list))
#     print("ARED COMPLETE")

#     average_o_pt_in_clusters = 0
#     for cluster in ared.subspace_partition.cluster_list:
#       average_o_pt_in_clusters += len(cluster.o_pts)
#     average_o_pt_in_clusters /= len(ared.subspace_partition.cluster_list)

#     print(average_o_pt_in_clusters)

#     average_o_pt_in_clusters = 0
#     clusters_w_o_pts = 0
#     for cluster in ared.subspace_partition.cluster_list:
#       if len(cluster.o_pts) != 0:
#         clusters_w_o_pts += 1
#         average_o_pt_in_clusters += len(cluster.o_pts)
#     average_o_pt_in_clusters /= clusters_w_o_pts

#     print(average_o_pt_in_clusters)

#     average_l_pt_in_clusters = 0
#     for cluster in ared.subspace_partition.cluster_list:
#       average_l_pt_in_clusters += len(cluster.l_pts)
#     average_l_pt_in_clusters /= len(ared.subspace_partition.cluster_list)

#     print(average_l_pt_in_clusters)

#     average_l_pt_in_clusters = 0
#     clusters_w_l_pts = 0
#     for cluster in ared.subspace_partition.cluster_list:
#       if len(cluster.l_pts) != 0:
#         clusters_w_l_pts += 1
#         average_l_pt_in_clusters += len(cluster.l_pts)
#     average_l_pt_in_clusters /= clusters_w_l_pts

#     print(average_l_pt_in_clusters)