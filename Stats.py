"""# Stats"""

class Stats:
    def __init__(self, ared):
        self.num_queries = len(ared.labeled_data.abs_idx_array)
        self.num_queries_by_time = []

        highest_idx = ared.data_window.abs_idx_max

        last_idx = -1
        for i, query_abs_idx in enumerate(ared.labeled_data.abs_idx_array):
            diff = query_abs_idx - last_idx
            while 0 < diff:
                self.num_queries_by_time.append(i + 1)
                diff += -1

            last_idx = query_abs_idx

        diff = highest_idx - last_idx
        i = self.num_queries_by_time[len(self.num_queries_by_time) - 1]
        while 0 < diff:
            self.num_queries_by_time.append(i + 1)
            diff += -1