from Circular_Buffer import *
import numpy as np

"""# Data Window"""

class Data_Window:
    def __init__(self, data_window_size = 1000):
        self.abs_idx_max = -1 # abs_idx_max is the absolute index most recent point inserted
        self.abs_idx_min = 0
        self.data_window_size = data_window_size
        self.assigned_cluster_id_window = Circular_Buffer(data_window_size)
        self.is_point_labeled_window = Circular_Buffer(data_window_size)
        self.data_in_window = Circular_Buffer(data_window_size)
        self.last_removed_cluster_id = None # Cluster ID of most recently forgotten point, the abs_idx of that point is abs_idx_min - 1

    def insert_data(self, data_point):
        self.data_in_window.append(data_point)
        self.abs_idx_max += 1
        self.abs_idx_min = max(0, self.abs_idx_max - self.data_window_size + 1)

        self.last_removed_cluster_id = self.assigned_cluster_id_window.append(None)

        self.is_point_labeled_window.append(False)

    def get_data_point(self, abs_index):
        if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

        dw_index = abs_index - self.abs_idx_min
        return self.data_in_window.get(dw_index)

    def update_cluster_id_at(self, abs_index, new_id):
        if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

        dw_index = abs_index - self.abs_idx_min
        self.assigned_cluster_id_window.set_at(dw_index, new_id)

    def updated_labeled_window(self, abs_index):
      if not (self.abs_idx_min <= abs_index <= self.abs_idx_max):
            raise IndexError(f"abs_index {abs_index} is out of the window range "
                             f"[{self.abs_idx_min}, {self.abs_idx_max})")

      dw_index = abs_index - self.abs_idx_min
      self.is_point_labeled_window.set_at(dw_index, True)

"""# Labeled Data"""

class Labeled_Data:
  def __init__(self):
    self.abs_idx_array = []
    self.data_array = []
    self.cluster_id_array = []
    self.label_array = []
    self.relevance_array = []

  def add_point(self, abs_idx, data_point, cluster_id, label, relevance):
    self.abs_idx_array.append(abs_idx)
    self.data_array.append(data_point)
    self.cluster_id_array.append(cluster_id)
    self.label_array.append(label)
    self.relevance_array.append(relevance)

  def get_data(self, abs_idx):
    return self.data_array[self.abs_idx_array.index(abs_idx)]

  def get_ld_index(self, abs_idx):
    return self.abs_idx_array.index(abs_idx)

"""# Subspace Partition"""

class Subspace_Partition:
    def __init__(self):        #                                                                   (l_pts)          (o_pts)
        self.cluster_list = [] # cluster is expected to be in the format of [label, relevance, [abs_idx_l_pt], [abs_idx_o_pt], diameter]
        self.set_of_known_labels = set()
        # cluster id is the cluster's index in cluster_list

    def create_new_cluster(self, label, relevance, l_pts, o_pts, labeled_data, QS_VAR):
      self.set_of_known_labels.add(label)
      self.cluster_list.append(Cluster(label, relevance, l_pts, o_pts, labeled_data, QS_VAR))

"""# Cluster"""


class Cluster:
    def __init__(self, label, relevance, l_pts, o_pts, labeled_data, QS_VAR = 0):
        self.label = label
        self.relevance = relevance
        self.l_pts = l_pts
        self.o_pts = o_pts
        self.comp_distance = 0  # QR_VAR=0: Diameter, QS_VAR=1: approx_nn_distance

        # cluster id is this cluster's position in Subspace_Partition.cluster_list

        if len(l_pts) > 1 and QS_VAR == 0:
            self.update_diameter(labeled_data)

        elif len(l_pts) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist(labeled_data)

    def add_l_pt(self, abs_idx, labeled_data, QS_VAR = 0):
        self.l_pts.append(abs_idx)

        if QS_VAR == 0:
            self.update_diameter(labeled_data)
        elif QS_VAR == 1:
            self.update_ave_nn_dist(labeled_data)

    def add_o_pt(self, abs_idx):
        self.o_pts.append(abs_idx)

    def update_diameter(self, labeled_data):
        largest_distance = 0
        for i in range(len(self.l_pts)):
            for j in range(i):
                data_l_pt_i = labeled_data.get_data(self.l_pts[i])
                data_l_pt_j = labeled_data.get_data(self.l_pts[j])
                distance = np.linalg.norm(data_l_pt_i - data_l_pt_j)
                if largest_distance < distance:
                    largest_distance = distance
        self.comp_distance = largest_distance

    def update_ave_nn_dist(self, labeled_data):
        newest_l_pt = self.l_pts[len(self.l_pts) - 1]
        closest_dist = float('inf')
        for l_pt in self.l_pts[:len(self.l_pts) - 1]:
            distance = np.linalg.norm(labeled_data.get_data(newest_l_pt) - labeled_data.get_data(l_pt))
            if distance < closest_dist:
                closest_dist = distance

        # update running average
        self.comp_distance = (self.comp_distance * (len(self.l_pts) + closest_dist)) / len(self.l_pts)

"""# ARED"""

class ARED:
    def __init__(self, oracle, kappa=1.0, data_window_size=1000, QS_VAR = 0, VERBOSE_FLAGS = []):
        self.kappa = kappa
        self.data_window = Data_Window(data_window_size)
        self.labeled_data = Labeled_Data()
        self.subspace_partition = Subspace_Partition()
        self.oracle = oracle

        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.verbose_flags = VERBOSE_FLAGS


    def process_first_point(self, data_point):

        # Insert data point into data_window
        self.data_window.insert_data(data_point)
        data_point_abs_idx = self.data_window.abs_idx_max

        # START QUERY
        label, relevance = self.query(data_point_abs_idx)
        # END QUERY

        cluster_id = 0

        # Update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(0, 0)

        # Create new cluster
        self.labeled_data.add_point(data_point_abs_idx, data_point, cluster_id, label, relevance) #cluster_id = 0
        self.subspace_partition.create_new_cluster(label, relevance, [data_point_abs_idx], [], self.labeled_data, self.QS_VAR)

        if 1 in self.verbose_flags:
            print("new cluster:", 0, [0])


    def determine_comparison_cluster(self, data_point):

      shortest_distance = np.inf
      closest = None

      for i, cluster in enumerate(self.subspace_partition.cluster_list):
        for abs_l_pt_index in cluster.l_pts:
          l_pt_data = self.labeled_data.get_data(abs_l_pt_index)
          distance = np.linalg.norm(l_pt_data - data_point)

          if distance < shortest_distance:
            shortest_distance = distance
            closest = (i, distance)

      return closest # (cluster_id, distance)


    def anomalous(self, data_point, cluster_id, distance):
        cluster = self.subspace_partition.cluster_list[cluster_id]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        self.data_window.updated_labeled_window(abs_data_index)
        # return (label, relevance) from oracle
        return self.oracle.answer_query(abs_data_index)


    # ran when we add a new o_pt to a cluster
    def add_o_pt(self, abs_idx, cluster_id):

      if 1 in self.verbose_flags:
        print("add_o_pt:", abs_idx, cluster_id)

      cluster = self.subspace_partition.cluster_list[cluster_id]
      cluster.add_o_pt(abs_idx)

      # update data_window.assigned_cluster_id_window
      self.data_window.update_cluster_id_at(abs_idx, cluster_id)


    # ran when we add a new labeled data point to a known cluster
    def add_l_pt(self, abs_idx, data_point, cluster_id):

      if 1 in self.verbose_flags:
        print("add_l_pt:", abs_idx, cluster_id)

      # update cluster in subspace partition
      cluster = self.subspace_partition.cluster_list[cluster_id]

      # get label and relevance
      label = cluster.label
      relevance = cluster.relevance

      # update data_window.assigned_cluster_id_window
      self.data_window.update_cluster_id_at(abs_idx, cluster_id)

      # update labeled_data to have the new point
      self.labeled_data.add_point(abs_idx, data_point, cluster_id, label, relevance)

      # add point to cluster, so diameter gets updated properly
      cluster.add_l_pt(abs_idx, self.labeled_data, self.QS_VAR)


    def split(self, data_point, data_point_idx, new_cluster_label, relevance, old_cluster_id):

      new_cluster_id = len(self.subspace_partition.cluster_list)
      self.labeled_data.add_point(data_point_idx, data_point, new_cluster_id, new_cluster_label, relevance)
      self.data_window.update_cluster_id_at(data_point_idx, new_cluster_id)
      self.subspace_partition.create_new_cluster(new_cluster_label, relevance, [data_point_idx], [], self.labeled_data, self.QS_VAR)

      if 1 in self.verbose_flags:
        print("new cluster:", new_cluster_id, [data_point_idx])

      # array to hold o_pt indexes during the split process
      new_cluster_o_pts_abs_inds = []
      old_cluster_o_pts_abs_inds = []

      # get o_pt indices
      o_pts_abs_inds_to_split = self.subspace_partition.cluster_list[old_cluster_id].o_pts

      if (len(o_pts_abs_inds_to_split) == 0):
        if 2 in self.verbose_flags:
            print("No o_pts to split")
        return

      # get l_pt indices
      l_pt_inds = self.subspace_partition.cluster_list[old_cluster_id].l_pts

      # o_pt_index is an abs_idx
      for o_pt_index in o_pts_abs_inds_to_split:
          o_pt = self.data_window.get_data_point(o_pt_index)

          # find the closest labeled point in the exisiting cluster
          distance_to_existing = min([
              np.linalg.norm(o_pt - self.labeled_data.get_data(l_pt_index))
              for l_pt_index in l_pt_inds
          ])

          # get the distance to the labeled point in the new cluster
          distance_to_new = np.linalg.norm(o_pt - data_point)

          # put the o_pt in the closest cluster of the two
          if distance_to_existing < distance_to_new:
              old_cluster_o_pts_abs_inds.append(o_pt_index)
          else:
              #print(distance_to_new, distance_to_existing, o_pt_index)
              new_cluster_o_pts_abs_inds.append(o_pt_index)

              # update the data window so the assigned_label_id_window is correct for window maintenance later
              self.data_window.update_cluster_id_at(o_pt_index, new_cluster_id)

      if 2 in self.verbose_flags:
        print("Split :")
        print("old_cluster_id w/ o_pts:", old_cluster_id, old_cluster_o_pts_abs_inds)
        print("new_cluster_id w/ o_pts:", new_cluster_id, new_cluster_o_pts_abs_inds)

      # put the o_pts in their correct cluster
      self.subspace_partition.cluster_list[new_cluster_id].o_pts = new_cluster_o_pts_abs_inds # update o_pts new_cluster
      self.subspace_partition.cluster_list[old_cluster_id].o_pts = old_cluster_o_pts_abs_inds # update o_pts old_cluster

    def relevance_processing(self, new_cluster_id):
        pass

    # Removing forgotten o_pts from the subspace partition
    def subspace_partition_maintenance(self, forgotten_abs_idx, forgotten_point_cluster_id):

      cluster = self.subspace_partition.cluster_list[forgotten_point_cluster_id]

      if 4 in self.verbose_flags:
          print(forgotten_abs_idx, forgotten_point_cluster_id)

      cluster.o_pts.remove(forgotten_abs_idx)

    def process_point(self, data_point):

      if 3 in self.verbose_flags:
          print("labeled id array:", self.labeled_data.cluster_id_array)
          print("labeled abs array:", self.labeled_data.abs_idx_array)
          print("data window assigned id:", self.data_window.assigned_cluster_id_window.get_array())


      is_forgotten_point_labeled = self.data_window.is_point_labeled_window.get(0) # the '0' here is the index of the oldest element in data window

      self.data_window.insert_data(data_point)
      data_point_abs_idx = self.data_window.abs_idx_max

      forgotten_abs_idx = self.data_window.abs_idx_min - 1
      forgotten_pt_cluster_id = self.data_window.last_removed_cluster_id

      # if forgotten_pt_cluster_id is NOT None (ie a point has been fogotten) do maintenance
      if forgotten_pt_cluster_id != None and not is_forgotten_point_labeled:
        self.subspace_partition_maintenance(forgotten_abs_idx, forgotten_pt_cluster_id)

      # START DETERMINE COMPARISON CLUSTER

      comp_cluster_id, distance = self.determine_comparison_cluster(data_point)

      relevant = self.subspace_partition.cluster_list[comp_cluster_id].relevance

      # END DETERMINE COMPARISON CLUSTER

      # START NOT RELEVANT

      if not relevant:
        # START NOT ANOMALOUS
        if not self.anomalous(data_point, comp_cluster_id, distance):

          self.add_o_pt(data_point_abs_idx, comp_cluster_id)

          return # Data point processed, END Function

        # END NOT ANOMALOUS

      #END NOT RELEVANT

      # START QUERY
      label, relevant = self.query(data_point_abs_idx)
      # END QUERY

      # START NOT NEW LABEL
      comp_cluster_label = self.subspace_partition.cluster_list[comp_cluster_id].label
      label_is_same = (label == comp_cluster_label)

      if label_is_same: # if not a new label
        self.add_l_pt(data_point_abs_idx, data_point, comp_cluster_id)

        return # Data point processed, END Function

      # END NOT NEW LABEL

      # START NEW LABEL
      # create new cluster with the split o_pts

      self.split(data_point, data_point_abs_idx, label, relevant, comp_cluster_id)

      # END NEW LABEL

      # START RELEVANCE PROCESSING
      if relevant:
        self.relevance_processing(len(self.subspace_partition.cluster_list) - 1)
      # END RELEVANCE PROCESSING

      # POINT PROCESSED