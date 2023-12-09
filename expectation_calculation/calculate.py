import numpy as np
import pickle
import time

from scipy.linalg import expm
from simulation.consts import CAP_S, CAP_C, MIN_STEP, MIN_RUN_STEP, RE_START_T, RE_END_T, SIM_END_T, NUM_STATIONS
from test_file.test_case import test_case_25

PREFERENCE = 1

# dep
mu_s_array, mu_c_array = test_case_25['mu_s_array'], test_case_25['mu_c_array']
mu_array = mu_s_array + mu_c_array
# arr
lambda_s_array, lambda_c_array = test_case_25['lambda_s_array'], test_case_25['lambda_c_array']


def get_target_state(num_s: int, num_c: int) -> int:
    return round(num_s * (CAP_C + 1) + num_c)


def get_transition_matrix(sel_s: int, sel_t: int, mode: str) -> np.ndarray:
    """get the transition matrix for the selected station and time"""
    if mode == 'multi':
        rand = np.zeros(shape=((CAP_S + 1) * (CAP_C + 1), (CAP_S + 1) * (CAP_C + 1)))
        for i in range(rand.shape[0]):
            num_s, num_c = i // (CAP_C + 1), i % (CAP_C + 1)
            if num_s == 0:
                rand[i, get_target_state(num_s + 1, num_c)] = lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                if num_c == 0:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1] - \
                                 lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                elif num_c == CAP_C:
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s + 1, num_c)] - rand[i, get_target_state(num_s, num_c - 1)]
                else:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s + 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c + 1)] - \
                                 rand[i, get_target_state(num_s, num_c - 1)]
            elif num_s == CAP_S:
                rand[i, get_target_state(num_s - 1, num_c)] = num_s / (num_s + PREFERENCE * num_c) * \
                                                              mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                if num_c == 0:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c + 1)]
                elif num_c == CAP_C:
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c - 1)]
                else:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c + 1)] - \
                                 rand[i, get_target_state(num_s, num_c - 1)]
            else:
                rand[i, get_target_state(num_s + 1, num_c)] = lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                rand[i, get_target_state(num_s - 1, num_c)] = num_s / (num_s + PREFERENCE * num_c) * \
                                                              mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                if num_c == 0:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s + 1, num_c)] - \
                                 rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c + 1)]
                elif num_c == CAP_C:
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s + 1, num_c)] - \
                                 rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c - 1)]
                else:
                    rand[i, get_target_state(num_s, num_c + 1)] = lambda_c_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, get_target_state(num_s, num_c - 1)] = PREFERENCE * num_c / (num_s + PREFERENCE * num_c) * \
                                                                  mu_array[round(sel_t / MIN_STEP), sel_s - 1]
                    rand[i, i] = - rand[i, get_target_state(num_s + 1, num_c)] - \
                                 rand[i, get_target_state(num_s - 1, num_c)] - \
                                 rand[i, get_target_state(num_s, num_c + 1)] - \
                                 rand[i, get_target_state(num_s, num_c - 1)]
    elif mode == 'single':
        rand = np.zeros(shape=(CAP_S + 1, CAP_S + 1))
        for i in range(rand.shape[0]):
            num_s = i
            if num_s == 0:
                rand[i, num_s + 1] = lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                rand[i, i] = - lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1]
            elif num_s == CAP_S:
                rand[i, num_s - 1] = mu_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                rand[i, i] = - mu_s_array[round(sel_t / MIN_STEP), sel_s - 1]
            else:
                rand[i, num_s + 1] = lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                rand[i, num_s - 1] = mu_s_array[round(sel_t / MIN_STEP), sel_s - 1]
                rand[i, i] = - lambda_s_array[round(sel_t / MIN_STEP), sel_s - 1] \
                             - mu_s_array[round(sel_t / MIN_STEP), sel_s - 1]
    else:
        assert False, 'mode error'

    return rand


def get_expectation(mode: str) -> tuple:
    """calculate the expected inventory level (EIo(t0, t, Io, Ir), EIr(t0, t, Io, Ir))
    and the expected satisfied demands ESD(t0, t, Io, Ir)"""
    if mode == 'multi':
        EI_s_array = np.zeros(
            shape=(
            NUM_STATIONS, round((RE_END_T - RE_START_T) / MIN_RUN_STEP), round((SIM_END_T - RE_START_T) / MIN_RUN_STEP + 1),
            CAP_S + 1, CAP_C + 1))
        EI_c_array = np.zeros(
            shape=(
            NUM_STATIONS, round((RE_END_T - RE_START_T) / MIN_RUN_STEP), round((SIM_END_T - RE_START_T) / MIN_RUN_STEP + 1),
            CAP_S + 1, CAP_C + 1))
        ESD_array = np.zeros(
            shape=(
            NUM_STATIONS, round((RE_END_T - RE_START_T) / MIN_RUN_STEP), round((SIM_END_T - RE_START_T) / MIN_RUN_STEP + 1),
            CAP_S + 1, CAP_C + 1))

        start = time.time()

        rate_list = []
        for m in range(41):
            for n in range(81):
                if m + n == 0:
                    rate_list.append(0)
                else:
                    rate_list.append(m / (m + PREFERENCE * n))
        rate_array = np.array(rate_list).reshape((-1, 1))

        for sel_s in range(1, 26):

            p_mat_list = []

            for tmp in range(RE_START_T, SIM_END_T, MIN_STEP):
                p_mat_list.append(expm(get_transition_matrix(sel_s=sel_s, sel_t=tmp, mode=mode)))
            # for _ in range(96):
            #     p_mat_list.append(np.eye((CAP_S + 1) * (CAP_C + 1)))

            end_p_mat = time.time()
            print(f'p_mat_generated for station {sel_s}')
            print(f'time used: {end_p_mat - start}')

            for t0 in range(RE_START_T, RE_END_T, MIN_RUN_STEP):
                pi_mat = np.eye((CAP_S + 1) * (CAP_C + 1))
                for t in range(t0, SIM_END_T + 1, MIN_STEP):
                    if t == t0:
                        for num_s in range(CAP_S + 1):
                            for num_c in range(CAP_C + 1):
                                EI_s_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                    (t - RE_START_T) / MIN_RUN_STEP), num_s, num_c] = num_s
                                EI_c_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                    (t - RE_START_T) / MIN_RUN_STEP), num_s, num_c] = num_c
                                ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                    (t - RE_START_T) / MIN_RUN_STEP), num_s, num_c] = 0
                    else:
                        pi_mat = np.dot(pi_mat, p_mat_list[round((t - RE_START_T) / MIN_STEP - 1)])
                        if (t - t0) % MIN_RUN_STEP == 0:
                            for num_s in range(CAP_S + 1):
                                for num_c in range(CAP_C + 1):
                                    EI_s_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                        (t - RE_START_T) / MIN_RUN_STEP), num_s, num_c] = \
                                        np.dot(pi_mat[get_target_state(num_s, num_c), :],
                                               np.arange(CAP_S + 1).repeat(CAP_C + 1).T)
                                    EI_c_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                        (t - RE_START_T) / MIN_RUN_STEP), num_s, num_c] = \
                                        np.dot(pi_mat[get_target_state(num_s, num_c), :],
                                               np.tile(np.arange(CAP_C + 1), CAP_S + 1).T)
                                    ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP),
                                    round((t - RE_START_T) / MIN_RUN_STEP):, num_s, num_c] += \
                                        np.dot(pi_mat[get_target_state(num_s, num_c), :], rate_array) * mu_array[
                                            round(t / MIN_STEP), sel_s - 1]
                        else:
                            for num_s in range(CAP_S + 1):
                                for num_c in range(CAP_C + 1):
                                    ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP),
                                    round((t + MIN_STEP - RE_START_T) / MIN_RUN_STEP):, num_s, num_c] += \
                                        np.dot(pi_mat[get_target_state(num_s, num_c), :], rate_array) * mu_array[round(t / MIN_STEP), sel_s - 1]

            end_calculation = time.time()
            print(f'calculation done for station {sel_s}')
            print(f'time used: {end_calculation - end_p_mat}')

    elif mode == 'single':
        EI_s_array = np.zeros(
            shape=(
                NUM_STATIONS, round((RE_END_T - RE_START_T) / MIN_RUN_STEP),
                round((SIM_END_T - RE_START_T) / MIN_RUN_STEP + 1),
                CAP_S + 1))
        EI_c_array = 0
        ESD_array = np.zeros(
            shape=(
                NUM_STATIONS, round((RE_END_T - RE_START_T) / MIN_RUN_STEP),
                round((SIM_END_T - RE_START_T) / MIN_RUN_STEP + 1),
                CAP_S + 1))

        start = time.time()

        rate_list = []
        for m in range(41):
            if m == 0:
                rate_list.append(0)
            else:
                rate_list.append(1)
        rate_array = np.array(rate_list).reshape((-1, 1))

        for sel_s in range(1, 26):

            p_mat_list = []

            for tmp in range(RE_START_T, SIM_END_T, MIN_STEP):
                p_mat_list.append(expm(get_transition_matrix(sel_s=sel_s, sel_t=tmp, mode=mode)))
            # for _ in range(96):
            #     p_mat_list.append(np.eye((CAP_S + 1) * (CAP_C + 1)))

            end_p_mat = time.time()
            print(f'p_mat_generated for station {sel_s}')
            print(f'time used: {end_p_mat - start}')

            for t0 in range(RE_START_T, RE_END_T, MIN_RUN_STEP):
                pi_mat = np.eye(CAP_S + 1)
                for t in range(t0, SIM_END_T + 1, MIN_STEP):
                    if t == t0:
                        for num_s in range(CAP_S + 1):
                            EI_s_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                (t - RE_START_T) / MIN_RUN_STEP), num_s] = num_s
                            ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                (t - RE_START_T) / MIN_RUN_STEP), num_s] = 0
                    else:
                        pi_mat = np.dot(pi_mat, p_mat_list[round((t - RE_START_T) / MIN_STEP - 1)])
                        if (t - t0) % MIN_RUN_STEP == 0:
                            for num_s in range(CAP_S + 1):
                                EI_s_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round(
                                    (t - RE_START_T) / MIN_RUN_STEP), num_s] = np.dot(pi_mat[num_s, :], np.arange(CAP_S + 1).T)
                                ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP), round((t - RE_START_T) / MIN_RUN_STEP):, num_s] += \
                                    np.dot(pi_mat[num_s, :], rate_array) * mu_s_array[round(t / MIN_STEP), sel_s - 1]
                        else:
                            for num_s in range(CAP_S + 1):
                                ESD_array[sel_s - 1, round((t0 - RE_START_T) / MIN_RUN_STEP),
                                round((t + MIN_STEP - RE_START_T) / MIN_RUN_STEP):, num_s] += \
                                    np.dot(pi_mat[num_s, :], rate_array) * mu_s_array[round(t / MIN_STEP), sel_s - 1]

            end_calculation = time.time()
            print(f'calculation done for station {sel_s}')
            print(f'time used: {end_calculation - end_p_mat}')

    else:
        assert False, 'mode error'

    return EI_s_array, EI_c_array, ESD_array


if __name__ == '__main__':

    calculate_mode = 'single'

    if calculate_mode == 'multi':

        a, b, c = get_expectation(mode=calculate_mode)

        with open('EI_s_array_multi.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('EI_c_array_multi.pkl', 'wb') as f:
            pickle.dump(b, f)
        with open('ESD_array_multi.pkl', 'wb') as f:
            pickle.dump(c, f)

    elif calculate_mode == 'single':

        a, _, c = get_expectation(mode=calculate_mode)

        with open('EI_s_array_single.pkl', 'wb') as f:
            pickle.dump(a, f)
        with open('ESD_array_single.pkl', 'wb') as f:
            pickle.dump(c, f)

    else:
        assert False, 'mode error'
