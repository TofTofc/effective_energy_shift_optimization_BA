import numba
import numpy as np
from numba import njit

from versions.aa_compress.resize import add_excess_value, add_deficit_value, insert_excess_value

@numba.njit(nogil=True, inline="always", fastmath=True)
def remove_node(node_idx, head_idx, ll_prev, ll_next):
    """
    Remove a node from the double linked list by updating adjacent pointers

    Returns new head_idx if prev was removed
    """

    prev_node = ll_prev[node_idx]
    next_node = ll_next[node_idx]

    if prev_node != -1:
        ll_next[prev_node] = next_node
    else:
        head_idx = next_node

    if next_node != -1:
        ll_prev[next_node] = prev_node

    return head_idx

@numba.njit(nogil=True, inline="always", fastmath=True)
def init(excess_array, deficit_array):
    """
    Fills out the state mask and creates a doubly linked list of phase groups

    phase_meta:
    0: size_excess
    1: size_deficit
    2: mask[0] (Boolean: Has phase Excess?)
    3: mask[1] (Boolean: Has phase Deficit?)

    data arrays:
    0: starting height
    1: energy excess or deficit
    """

    n = excess_array.shape[0]
    # Not smaller than 2
    initial_capacity = 2

    data_excess = np.empty((n, initial_capacity, 2), dtype=np.float64)
    data_deficit = np.empty((n, initial_capacity, 2), dtype=np.float64)

    phase_meta = np.zeros((n, 4), dtype=np.uint8)
    phase_meta[:, 0] = 1
    phase_meta[:, 1] = 1

    # Double Linked List
    ll_prev = np.full(n, -1, dtype=np.int32)
    ll_next = np.full(n, -1, dtype=np.int32)
    ll_start = np.zeros(n, dtype=np.int32)
    ll_end = np.zeros(n, dtype=np.int32)

    e_counter = 0
    d_counter = 0

    node_ptr = -1

    # 0: Balanced 1: Excess 2: Deficit
    last_type = 0

    for i in range(n):
        data_excess[i, 0, 0] = 0
        data_excess[i, 0, 1] = excess_array[i]
        data_deficit[i, 0, 0] = 0
        data_deficit[i, 0, 1] = deficit_array[i]

        e_ex = excess_array[i]
        e_def = deficit_array[i]

        diff = e_ex - e_def
        current_type = 0

        if diff > 1e-12:
            current_type = 1
            e_counter += 1
            phase_meta[i, 2] = 1
            data_excess[i, 0, 1] = e_def
            data_excess[i, 1, 0] = e_def
            data_excess[i, 1, 1] = e_ex - e_def
            phase_meta[i, 0] = 2

        elif diff < -1e-12:
            current_type = 2
            d_counter += 1
            phase_meta[i, 3] = 1
            data_deficit[i, 0, 1] = e_ex
            data_deficit[i, 1, 0] = e_ex
            data_deficit[i, 1, 1] = e_def - e_ex
            phase_meta[i, 1] = 2

        if current_type != 0:
            if current_type == last_type:

                ll_end[node_ptr] = i
            else:

                new_node = node_ptr + 1
                ll_start[new_node] = i
                ll_end[new_node] = i

                if node_ptr != -1:

                    ll_next[node_ptr] = new_node
                    ll_prev[new_node] = node_ptr

                node_ptr = new_node
                last_type = current_type
        else:

            last_type = 0

    num_nodes = node_ptr + 1
    return e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes

@njit(nogil = True, fastmath = True)
def process_phases(excess_array, deficit_array, start_times):

    # Provides the initial states for each Phase object and balances them
    e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes = init(excess_array, deficit_array)



    return \
    (
        phase_meta[:, 0], phase_meta[:, 1],
        data_excess[:, :, 0], data_deficit[:, :, 0],
        data_excess[:, :, 1], data_deficit[:, :, 1],
        phase_meta[:, 3:].T
    )