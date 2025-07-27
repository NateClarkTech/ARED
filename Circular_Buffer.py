"""# Circular Buffer"""

class Circular_Buffer:
    def __init__(self, size):
        self.buffer = [None] * size
        self.size = size
        self.start = 0  # Points to the oldest element
        self.count = 0  # Number of elements currently in the buffer

    def append(self, value):
        index = (self.start + self.count) % self.size
        if self.count < self.size:
            self.buffer[index] = value
            self.count += 1
            return None  # No value was overwritten
        else:
            overwritten = self.buffer[self.start]
            self.buffer[self.start] = value
            self.start = (self.start + 1) % self.size
            return overwritten

    def set_at(self, index, value):
        """Set value at a relative index within the buffer (0 = oldest)."""
        if index < 0 or index >= self.count:
            raise IndexError(f"Index out of bounds in circular buffer {index} {value}")
        real_index = (self.start + index) % self.size
        self.buffer[real_index] = value

    def get(self, idx):
      """Return the element at the given relative index (0 is the oldest element)."""
      if idx < 0 or idx >= self.count:
          raise IndexError(f"Index out of bounds in circular buffer. {idx}")
      return self.buffer[(self.start + idx) % self.size]

    def print_array(self):
        """Print the contents of the buffer in order from oldest to newest."""
        elements = [self.buffer[(self.start + i) % self.size] for i in range(self.count)]
        print(elements)

    def get_array(self):
        return [self.buffer[(self.start + i) % self.size] for i in range(self.count)]

    def __repr__(self):
        return f"Circular_Buffer({self.get()})"
