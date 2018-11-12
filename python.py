# This tool computes the cartesian product of input iterables. 
from itertools import product
print(list(product([1,2,3],[3,4])))
output-[(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)]


# This tool returns successive  length permutations of elements in an iterable.
>>> print (list(permutations(['1','2','3'],2)))
output-[('1', '2'), ('1', '3'), ('2', '1'), ('2', '3'), ('3', '1'), ('3', '2')]

# This tool returns the  length subsequences of elements from the input iterable.
>>> print list(combinations('12345',2))
output-[('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '3'), ('2', '4'), ('2', '5'), ('3', '4'), ('3', '5'), ('4', '5')]
