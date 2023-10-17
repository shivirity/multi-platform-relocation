import numpy as np
import pickle

from test_case import test_case_25

"""
with open(
        r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_c_array.pkl', 'rb'
) as file:
    arr_c_array = pickle.load(file)
with open(
        r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_s_array.pkl', 'rb'
) as file:
    arr_s_array = pickle.load(file)
with open(
        r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\dep_c_array.pkl', 'rb'
) as file:
    dep_c_array = pickle.load(file)
with open(
        r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\dep_s_array.pkl', 'rb'
) as file:
    dep_s_array = pickle.load(file)
"""

arr_s_array, dep_s_array, arr_c_array, dep_c_array = \
    test_case_25['lambda_s_array'], test_case_25['mu_s_array'], test_case_25['lambda_c_array'], test_case_25['mu_c_array']

test_s_array = arr_s_array - dep_s_array
test_c_array = arr_c_array - dep_c_array

test_s_hour_array, test_c_hour_array = np.zeros((48, test_s_array.shape[1])), np.zeros((48, test_s_array.shape[1]))
for i in range(48):
    for j in range(test_s_array.shape[1]):
        test_s_hour_array[i, j] = np.sum(test_s_array[i * 6:i * 6 + 6, j])
        test_c_hour_array[i, j] = np.sum(test_c_array[i * 6:i * 6 + 6, j])

test_hour_array = test_s_hour_array + test_c_hour_array