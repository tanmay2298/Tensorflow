import numpy as np
my_list = [1, 2, 3]

arr = np.array(my_list)
print(arr)
print(np.arange(0, 10))
print(np.arange(0, 10, 2))
print(np.zeros(5))
print(np.zeros((3, 5)))
print(np.ones(3))
print(np.linspace(0, 2, 5))
print(np.random.randint(0, 5, (2, 2)))
np.random.seed(101)
arr = np.array(np.random.randint(0, 100))
print(arr.max())