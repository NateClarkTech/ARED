"""# Oracle"""

class Oracle:
  def __init__(self, X, y):
    self.X = X #[[data]]
    self.y = y #[[label, relevant]]

  def answer_query(self, abs_index):
    label = self.y[abs_index][0]
    relevance = self.y[abs_index][1]
    return label, relevance