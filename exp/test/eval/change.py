import os
import numpy as np

filenames = os.listdir()
filenames.sort()
counter1 = 0
counter2 = 0
for name in filenames:
	print(name)
	if 'round' in name:
		new_name = str(counter1) + "_test_round.npy"
		os.rename(name, new_name)
		counter1 += 1
	elif 'bernoulli' in name:
		new_name = str(counter2) + "_test_bernoulli.npy"
		os.rename(name, new_name)
		counter2 += 1
	# new_name = str(counter) + ".npy"