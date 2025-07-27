"""# Data Stream"""

class Data_Stream:
  def __init__(self, X, y):
    self.X = X #[[data]]
    self.y = y #[[label, relevant]]
    self.stream_counter = 0

  def stream_new_data_point(self):
    data_point = self.X[self.stream_counter]
    self.stream_counter += 1
    return data_point

  def get_remaining_num_points(self):
    return len(self.X) - self.stream_counter