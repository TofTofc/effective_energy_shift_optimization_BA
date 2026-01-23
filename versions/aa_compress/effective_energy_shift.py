import numba
import numpy as np
from numba import njit

from versions.aa_compress.resize import insert_excess_value, insert_deficit_value

@numba.njit(nogil=True, inline="always")
def find_head(ll_prev, ll_next, num_nodes):

    head = -1
    n = len(ll_next)
    for i in range(n):
        prev_i = ll_prev[i]
        next_i = ll_next[i]
        if 0 <= prev_i < n and 0 <= next_i < n:
            if ll_next[prev_i] == i and (next_i != i or num_nodes == 1):
                head = i
                break
    return head

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

    if num_nodes > 0:

        ll_next[node_ptr] = 0
        ll_prev[0] = node_ptr

        if num_nodes > 1:

            idx_f = ll_start[0]
            idx_l = ll_start[node_ptr]

            if (phase_meta[idx_f, 2] == phase_meta[idx_l, 2]) and (phase_meta[idx_f, 3] == phase_meta[idx_l, 3]):

                ll_start[0] = ll_start[node_ptr]
                pn = ll_prev[node_ptr]
                ll_next[pn] = 0
                ll_prev[0] = pn
                num_nodes -= 1

    return e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes


@numba.njit(nogil=True, inline="always", fastmath=True)
def compress(e_counter, d_counter,
             phase_meta, data_excess, data_deficit,
             ll_prev, ll_next, ll_start, ll_end, num_nodes):

    n = phase_meta.shape[0]

    if num_nodes <= 0:
        return (e_counter, d_counter, phase_meta,
                data_excess, data_deficit,
                ll_prev, ll_next, ll_start, ll_end, num_nodes)

    current = find_head(ll_prev, ll_next, num_nodes)
    processed_count = 0
    original_num_nodes = num_nodes

    while processed_count < original_num_nodes:

        is_excess = (
            phase_meta[ll_start[current], 2] == 1 and
            phase_meta[ll_start[current], 3] == 0
        )

        next_node = ll_next[current]

        if is_excess:

            target_idx = ll_start[next_node]
            start = ll_start[current]
            end = ll_end[current]

            nodes_in_block = (
                (end - start + 1)
                if start <= end
                else (n - start + end + 1)
            )

            target_has_uncovered = phase_meta[target_idx, 2] == 1

            idx = end

            for _ in range(nodes_in_block):

                num_balanced = phase_meta[idx, 1]

                for p_idx in range(num_balanced, phase_meta[idx, 0]):

                    start_height = data_excess[idx, p_idx, 0]
                    energy = data_excess[idx, p_idx, 1]

                    t_p_cnt = phase_meta[target_idx, 0]

                    if target_has_uncovered:

                        t_last = t_p_cnt - 1
                        target_end_height = (
                            data_excess[target_idx, t_last, 0] +
                            data_excess[target_idx, t_last, 1]
                        )

                        if target_end_height >= start_height:
                            data_excess[target_idx, t_last, 1] += energy
                        else:
                            data_excess, phase_meta = insert_excess_value(
                                target_idx,
                                t_p_cnt,
                                start_height,
                                energy,
                                data_excess,
                                phase_meta
                            )
                    else:
                        t_last = t_p_cnt - 1
                        target_end_height = (
                            data_excess[target_idx, t_last, 0] +
                            data_excess[target_idx, t_last, 1]
                        )

                        start_height = max(start_height, target_end_height)

                        data_excess, phase_meta = insert_excess_value(
                            target_idx,
                            t_p_cnt,
                            start_height,
                            energy,
                            data_excess,
                            phase_meta
                        )

                        target_has_uncovered = True

                phase_meta[idx, 0] = phase_meta[idx, 1]
                phase_meta[idx, 2] = 0
                phase_meta[idx, 3] = 0

                idx = (idx - 1 + n) % n

            p_node = ll_prev[current]
            n_node = ll_next[current]
            ll_next[p_node] = n_node
            ll_prev[n_node] = p_node

            num_nodes -= 1
            current = n_node

        else:

            base_idx = ll_start[current]
            start = ll_start[current]
            end = ll_end[current]

            if end != start:

                nodes_to_proc = (
                    (end - start)
                    if start < end
                    else (n - start + end)
                )

                idx = (start + 1) % n

                for _ in range(nodes_to_proc):

                    num_balanced = phase_meta[idx, 0]

                    for p_idx in range(num_balanced, phase_meta[idx, 1]):

                        start_height = data_deficit[idx, p_idx, 0]
                        energy = data_deficit[idx, p_idx, 1]

                        t_p_cnt = phase_meta[base_idx, 1]
                        t_last = t_p_cnt - 1

                        target_end_height = (
                            data_deficit[base_idx, t_last, 0] +
                            data_deficit[base_idx, t_last, 1]
                        )

                        if target_end_height >= start_height:
                            data_deficit[base_idx, t_last, 1] += energy
                        else:
                            start_height = max(start_height, target_end_height)
                            data_deficit, phase_meta = insert_deficit_value(
                                base_idx,
                                t_p_cnt,
                                start_height,
                                energy,
                                data_deficit,
                                phase_meta
                            )

                    phase_meta[idx, 1] = phase_meta[idx, 0]
                    phase_meta[idx, 2] = 0
                    phase_meta[idx, 3] = 0

                    idx = (idx + 1) % n

                ll_end[current] = ll_start[current]

            current = ll_next[current]

        processed_count += 1

    return (e_counter, d_counter, phase_meta,
            data_excess, data_deficit,
            ll_prev, ll_next, ll_start, ll_end, num_nodes)

@numba.njit(nogil=True, inline="always", fastmath=True)
def balance(phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, num_nodes):

    if num_nodes <= 0:
        return phase_meta, data_excess, data_deficit

    curr = find_head(ll_prev, ll_next, num_nodes)
    for _ in range(num_nodes):

        idx = ll_start[curr]

        p_ex = 0
        p_def = 0

        while p_ex < phase_meta[idx, 0] and p_def < phase_meta[idx, 1]:

            height_ex = data_excess[idx, p_ex, 0]
            height_def = data_deficit[idx, p_def, 0]

            if abs(height_ex - height_def) > 1e-12:
                new_h = max(height_ex, height_def)
                data_excess[idx, p_ex, 0] = new_h
                data_deficit[idx, p_def, 0] = new_h

            energy_ex = data_excess[idx, p_ex, 1]
            energy_def = data_deficit[idx, p_def, 1]
            diff = energy_ex - energy_def

            if abs(diff) < 1e-12:

                p_ex += 1
                p_def += 1

            elif diff > 1e-12:

                data_excess[idx, p_ex, 1] = energy_def
                rem_energy = diff
                rem_start = data_excess[idx, p_ex, 0] + energy_def

                data_excess, phase_meta = insert_excess_value(idx, p_ex + 1, rem_start, rem_energy, data_excess, phase_meta)

                p_ex += 1
                p_def += 1

            else:

                data_deficit[idx, p_def, 1] = energy_ex
                rem_energy = abs(diff)
                rem_start = data_deficit[idx, p_def, 0] + energy_ex
                data_deficit, phase_meta = insert_deficit_value(idx, p_def + 1, rem_start, rem_energy, data_deficit, phase_meta)
                p_ex += 1
                p_def += 1

        has_remaining_ex = p_ex < phase_meta[idx, 0]
        has_remaining_def = p_def < phase_meta[idx, 1]

        if not has_remaining_ex and not has_remaining_def:

            phase_meta[idx, 2] = 0
            phase_meta[idx, 3] = 0

        elif has_remaining_ex and not has_remaining_def:

            phase_meta[idx, 2] = 1
            phase_meta[idx, 3] = 0

        elif has_remaining_def and not has_remaining_ex:

            phase_meta[idx, 2] = 0
            phase_meta[idx, 3] = 1

        curr = ll_next[curr]

    return phase_meta, data_excess, data_deficit


@numba.njit(nogil=True, inline="always", fastmath=True)
def merge_linked_list(phase_meta, ll_prev, ll_next, ll_start, ll_end, num_nodes):

    if num_nodes <= 1:
        return num_nodes

    curr = find_head(ll_prev, ll_next, num_nodes)
    processed = 0

    while processed < num_nodes+1:

        nxt = ll_next[curr]

        idx_curr = ll_start[curr]
        idx_nxt = ll_start[nxt]

        type_curr = 1 if phase_meta[idx_curr, 2] == 1 else 2
        type_nxt = 1 if phase_meta[idx_nxt, 2] == 1 else 2

        if type_curr == type_nxt and curr != nxt:

            ll_end[curr] = ll_end[nxt]

            new_next = ll_next[nxt]
            ll_next[curr] = new_next
            ll_prev[new_next] = curr

            num_nodes -= 1

        else:

            curr = nxt
            processed += 1

    return num_nodes


@njit(nogil = True, fastmath = True)
def process_phases(excess_array, deficit_array, start_times):

    n = excess_array.shape[0]

    # Provides the initial states for each Phase object and balances them
    e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes = init(excess_array, deficit_array)

    #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)
    #debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)

    while num_nodes > 1:

        #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)

        e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes = compress(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)

        #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)
        #debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)

        phase_meta, data_excess, data_deficit = balance(phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, num_nodes)

        #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)
        #debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)

        num_nodes = merge_linked_list(phase_meta, ll_prev, ll_next, ll_start, ll_end, num_nodes)

        #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)
        #debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)


    #debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n)
    #debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes)

    return \
    (
        phase_meta[:, 0], phase_meta[:, 1],
        data_excess[:, :, 0], data_deficit[:, :, 0],
        data_excess[:, :, 1], data_deficit[:, :, 1],
        phase_meta[:, 3:].T
    )











"""

def debug_linked_list(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes, n):
    print("\n" + "=" * 85)
    print(f"{'DEBUG: CIRCULAR LINKED LIST (WITH WRAP-AROUND)':^85}")
    print("=" * 85)
    print(f"Nodes: {num_nodes:<5} | Array Size: {n:<5} | Ex: {e_counter:<5} | Def: {d_counter}")
    print("-" * 85)
    print(f"{'Node ID':<8} | {'Range (Start-End)':<22} | {'Length':<8} | {'Prev':<6} | {'Next':<6} | {'Typ'}")
    print("-" * 85)

    if num_nodes <= 0:
        print("KEINE KNOTEN VORHANDEN")
        return

    curr = -1
    for i in range(len(ll_next)):
        if ll_next[ll_prev[i]] == i and (ll_next[i] != i or num_nodes == 1):
            curr = i
            break

    if curr == -1:
        print("FEHLER: Kein gültiger Einstiegspunkt gefunden")
        return

    for _ in range(num_nodes):
        s = ll_start[curr]
        e = ll_end[curr]

        actual_count = 0
        temp_idx = s

        loop_limit = n
        step = 0

        while step < loop_limit:

            if phase_meta[temp_idx, 2] == 1 or phase_meta[temp_idx, 3] == 1:
                actual_count += 1

            if temp_idx == e:
                break

            temp_idx = (temp_idx + 1) % n
            step += 1

        is_ex = phase_meta[s, 2] == 1 and phase_meta[s, 3] == 0
        n_type = "EXCESS " if is_ex else "DEFICIT"

        range_str = f"{s:>3} -> {e:>3}"
        print(f"{curr:<8} | {range_str:<22} | {actual_count:<8} | {ll_prev[curr]:<6} | {ll_next[curr]:<6} | {n_type}")

        curr = ll_next[curr]
        if curr == -1: break

    print("=" * 85 + "\n")


def debug_all_phases(e_counter, d_counter, phase_meta, data_excess, data_deficit, ll_prev, ll_next, ll_start, ll_end, num_nodes):
    print("\n" + "=" * 115)
    print(f"{'PHYSICAL PHASE MONITOR (DETAILED)':^115}")
    print("=" * 115)
    print(f"{'ID':<5} | {'Ex_Cnt':<6} | {'Def_Cnt':<7} | {'Sum_Ex':<12} | {'Sum_Def':<12} | {'Status':<10} | {'Details (Values)'}")
    print("-" * 115)

    total_sum_ex = 0.0
    total_sum_def = 0.0

    for i in range(phase_meta.shape[0]):
        ex_cnt = int(phase_meta[i, 0])
        def_cnt = int(phase_meta[i, 1])
        f_ex = phase_meta[i, 2]
        f_def = phase_meta[i, 3]

        ex_values = []
        sum_ex = 0.0
        for p in range(ex_cnt):
            val = data_excess[i, p, 1]
            sum_ex += val
            ex_values.append(f"{val:.5f}")

        def_values = []
        sum_def = 0.0
        for p in range(def_cnt):
            val = data_deficit[i, p, 1]
            sum_def += val
            def_values.append(f"{val:.5f}")

        total_sum_ex += sum_ex
        total_sum_def += sum_def

        if f_ex == 1 and f_def == 0:
            status = "EXCESS"
        elif f_ex == 0 and f_def == 1:
            status = "DEFICIT"
        elif f_ex == 1 and f_def == 1:
            status = "BOTH"
        else:
            status = "BALANCED"

        if ex_cnt > 0 or def_cnt > 0 or f_ex > 0 or f_def > 0:

            print(f"{i:<5} | {ex_cnt:<6} | {def_cnt:<7} | {sum_ex:<12.4f} | {sum_def:<12.4f} | {status:<10} | ", end="")

            ex_str = "[" + ", ".join(ex_values) + "]" if ex_values else "[]"
            def_str = "[" + ", ".join(def_values) + "]" if def_values else "[]"
            print(f"Ex: {ex_str} Def: {def_str}")

    print("-" * 115)
    print(f"{'TOTAL':<5} | {'':<6} | {'':<7} | {total_sum_ex:<12.4f} | {total_sum_def:<12.4f} |")
    print("=" * 115 + "\n")

"""