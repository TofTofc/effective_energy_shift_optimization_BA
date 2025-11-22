import sys

import numpy as np
from numba import njit
from numba.typed import List

from versions.ai_ic10_numba_test import efes_dataclasses



def calculate_virtual_excess(current_phase, next_phase):
    """
    Erzeugt einen Virtual Excess pro current_phases + next_phases Paar
    Starthöhe ist max aus eigener Excess Starthöhe und Excess Endhöhe der nächsten Phase
    """

    overflow_content = current_phase.energy_excess[current_phase.size_excess - 1]
    overflow_start = current_phase.starts_excess[current_phase.size_excess - 1]

    blocking_excess_content = next_phase.energy_excess[next_phase.size_excess - 1]
    blocking_excess_start = next_phase.starts_excess[next_phase.size_excess - 1]
    blocking_excess_end = blocking_excess_start + blocking_excess_content

    virtual_excess_start = max(overflow_start, blocking_excess_end)
    virtual_excess_content = overflow_content
    virtual_excess_id = current_phase.excess_ids[current_phase.size_excess - 1]

    return virtual_excess_start, virtual_excess_content, virtual_excess_id


def move_overflow_njit_index(phases, is_unbalanced_array, i):

    n = len(is_unbalanced_array[0])

    # current_phase und next_phase bestimmen
    current_phase = phases[i]
    next_idx = (i + 1) % n
    next_phase = phases[next_idx]

    # Balance next_phase
    balance_phase(phases[next_idx])

    last_e_current = current_phase.size_excess - 1
    last_d_current = current_phase.size_deficit - 1

    last_e_next = next_phase.size_excess - 1
    last_d_next = next_phase.size_deficit - 1

    # Teste ob überhaupt overflow vorhanden ist zum moven

    # Verschmelze den aktuellen Überschuss von next_phase mit dem neuen von current_phase
    next_phase.energy_excess[last_e_next] += current_phase.energy_excess[last_e_current]

    # Passe Start Excess von current_phase an, damit Überschüsse korrekt ankommen
    current_phase.starts_excess[last_e_current - 1] += current_phase.energy_excess[last_e_current]

    # Entferne den kompletten Überschuss von current_phase
    current_phase.remove_excess(-1)

    return phases, is_unbalanced_array


def balance_phase(phase: efes_dataclasses.Phase):

    last_e = phase.size_excess - 1
    last_d = phase.size_deficit - 1

    last_start_excess = phase.starts_excess[last_e]
    last_start_deficit = phase.starts_deficit[last_d]

    last_energy_excess = phase.energy_excess[last_e]
    last_energy_deficit = phase.energy_deficit[last_d]

    start_max = max(last_start_excess, last_start_deficit)

    phase.starts_excess[last_e] = start_max
    last_start_excess = start_max
    phase.starts_deficit[last_d] = start_max
    last_start_deficit = start_max


    #Falls Excess > Deficit -> Deficit Balanced (False)
    if phase.energy_excess[last_e] > phase.energy_deficit[last_d]:

        # Endzeitpunkt des letzten Deficit-Pakets
        new_start = last_start_deficit + last_energy_deficit

        # Energie Überschuss
        energy_excess = last_energy_excess - last_energy_deficit

        # Bisher letzter Excess wird auf das Niveau des Deficits gebracht -> Excess geht in ein neues Excess Objekt
        phase.energy_excess[last_e] = last_energy_deficit
        last_energy_excess = last_energy_deficit

        phase.append_excess(new_start, energy_excess, False, phase.excess_ids[last_e])


        # Falls Deficit > Excess -> Excess Balanced (False)
    else:

        # Endzeitpunkt des letzten Excess-Pakets
        new_start = last_start_excess + last_energy_excess

        # Energie Defizit
        energy_deficit = last_energy_deficit - last_energy_excess

        # Bisher letzter Deficit wird auf das Niveau des Excess gebracht -> Deficit geht in ein neues Deficit Objekt
        phase.energy_deficit[last_d] = last_energy_excess
        last_energy_deficit = last_energy_excess

        phase.append_deficit(new_start, energy_deficit, False)


def process_phases_njit(phases_typed_list):

    n = len(phases_typed_list)

    is_unbalanced_array = np.ones((2, n), dtype=np.bool_)
    """
        Maske mit 2 Spalten (Excess Spalte + Deficit Spalte) und n Zeilen 
        Initial auf True 
        True -> unbalanced
        False -> balanced  

    """
    i = 0

    while True:

        #print(is_unbalanced_array)
        print(i)
        print("_________________")

        # Ruft nur balance_phase auf für den aktuellen index, falls Excess und Deficit unbalanced sind
        if is_unbalanced_array[0, i] and is_unbalanced_array[1, i]:

            balance_phase(phases_typed_list[i])

        # Abbruch Bedingung
        excess_row_is_balanced = True
        deficit_row_is_balanced = True

        for x in range(n):

            if is_unbalanced_array[0, x] == True:
                excess_row_is_balanced = False

            if is_unbalanced_array[1, x] == True:
                deficit_row_is_balanced = False

        if excess_row_is_balanced or deficit_row_is_balanced:
            break
        # Ruft shift für den aktuellen Index auf
        phases_typed_list, is_unbalanced_array = move_overflow_njit_index(phases_typed_list, is_unbalanced_array, i)

        i = (i + 1) % n

    return phases_typed_list, is_unbalanced_array


def process_phases(energy_excess: np.ndarray, energy_deficit: np.ndarray, start_time_phases):

    phases_list = List()
    for ex, de, t in zip(energy_excess, energy_deficit, start_time_phases):
        phases_list.append(efes_dataclasses.Phase(ex, de, id=t))

    phases_out, mask_out = process_phases_njit(phases_list)

    return dict(phases=phases_out, mask=mask_out)
