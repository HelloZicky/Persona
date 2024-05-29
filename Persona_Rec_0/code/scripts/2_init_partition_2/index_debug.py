import numpy as np

train_lens = 3
test_lens = 6
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[:train_lens])
print(a[-test_lens:])