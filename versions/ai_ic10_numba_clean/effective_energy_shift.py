import numpy as np
from numba import njit
from numba.typed import List

from versions.ai_ic10_numba_clean import efes_dataclasses


@njit
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

    phase.excess_balanced[last_e] = True

    #Falls Excess und Deficit gleich hoch -> Beide Einträge Balanced (False)
    if last_energy_excess == last_energy_deficit:

        phase.deficit_balanced[last_d] = True
        return False, False

    #Falls Excess > Deficit -> Deficit Balanced (False)
    if phase.energy_excess[last_e] > phase.energy_deficit[last_d]:

        # Deficit gedeckt → markiere als balanced
        phase.deficit_balanced[last_d] = True

        # Endzeitpunkt des letzten Deficit-Pakets
        new_start = last_start_deficit + last_energy_deficit

        # Energie Überschuss
        energy_excess = last_energy_excess - last_energy_deficit

        # Bisher letzter Excess wird auf das Niveau des Deficits gebracht -> Excess geht in ein neues Excess Objekt
        phase.energy_excess[last_e] = last_energy_deficit
        last_energy_excess = last_energy_deficit

        phase.append_excess(new_start, energy_excess, False, phase.excess_ids[last_e])

        return True, False

    # Falls Deficit > Excess -> Excess Balanced (False)
    else:

        # Endzeitpunkt des letzten Excess-Pakets
        new_start = last_start_excess + last_energy_excess

        # Energie Defizit
        energy_deficit = last_energy_deficit - last_energy_excess

        # Bisher letzter Deficit wird auf das Niveau des Excess gebracht -> Deficit geht in ein neues Deficit Objekt
        phase.energy_deficit[last_d] = last_energy_excess
        last_energy_deficit = last_energy_excess

        # Das aktuell letzte Deficit Objekt ist nun Balanced, da der noch zu erfüllende Teil im neuen Deficit Objekt ist
        phase.deficit_balanced[last_d] = True

        phase.append_deficit(new_start, energy_deficit, False)

        return False, True


@njit
def balance_phases_njit(phases, is_unbalanced_array):
    """
    Ruft für alle Phasen jeweils balance_phase auf
    Setzt den Eintrag in is_unbalanced_array dementsprechend
    """

    n = len(phases)

    for i in range(n):

        # Ruft nur balance_phase auf, falls Excess und Deficit unbalanced sind
        if is_unbalanced_array[0, i] and is_unbalanced_array[1, i]:

            excess_is_balanced, deficit_is_balanced = balance_phase(phases[i])

            is_unbalanced_array[0, i] = excess_is_balanced
            is_unbalanced_array[1, i] = deficit_is_balanced

    return phases, is_unbalanced_array


@njit
def calculate_virtual_excess(current_phases, next_phases):
    """
    Erzeugt einen Virtual Excess pro current_phases + next_phases Paar
    Starthöhe ist max aus eigener Excess Starthöhe und Excess Endhöhe der nächsten Phase
    """

    m = len(current_phases)
    virtual_excess_starts = np.empty(m, dtype=np.float64)
    virtual_excess_contents = np.empty(m, dtype=np.float64)
    virtual_excess_ids = np.empty(m, dtype=np.int64)

    for i in range(m):

        overflow_content = current_phases[i].energy_excess[current_phases[i].size_excess - 1]
        overflow_start = current_phases[i].starts_excess[current_phases[i].size_excess - 1]

        blocking_excess_content = next_phases[i].energy_excess[next_phases[i].size_excess - 1]
        blocking_excess_start = next_phases[i].starts_excess[next_phases[i].size_excess - 1]
        blocking_excess_end = blocking_excess_start + blocking_excess_content

        virtual_excess_starts[i] = max(overflow_start, blocking_excess_end)
        virtual_excess_contents[i] = overflow_content
        virtual_excess_ids[i] = current_phases[i].excess_ids[current_phases[i].size_excess - 1]

    return virtual_excess_starts, virtual_excess_contents, virtual_excess_ids


@njit
def move_overflow_njit(phases, is_unbalanced_array):

    n = len(phases)

    # is_unbalanced_array_shifted[i] bekommt den Wert von is_unbalanced_array[0, i-1]
    # i=0 greift auf das letzte Element zu
    is_unbalanced_array_shifted = np.empty(n, dtype=np.bool_)
    for i in range(n):
        is_unbalanced_array_shifted[i] = is_unbalanced_array[0, (i - 1) % n]

    #Anzahl der Unbalanced Excess Werte
    count_excess_unbalanced = 0
    for i in range(n):
        if is_unbalanced_array[0, i]:
            count_excess_unbalanced += 1

    # Indizes der Nachfolger aller unbalanced Excess Einträge (zyklisch)
    next_indices = np.empty(count_excess_unbalanced, dtype=np.int64)
    j = 0
    for i in range(n):
        if is_unbalanced_array[0, i]:
            next_indices[j] = (i + 1) % n
            j += 1

    # Speichert alle unbalanced Excess Einträge in current_phases
    current_phases = List()
    for i in range(n):
        if is_unbalanced_array[0, i]:
            current_phases.append(phases[i])

    # Speichert die Nachfolger aller unbalanced Excess Einträge in next_phases
    next_phases = List()
    for k in range(len(next_indices)):
        idx = next_indices[k]
        next_phases.append(phases[idx])

    # Erstellt Listen aller Virtuellen Excess
    m = len(current_phases)
    v_e_start_arr, v_e_content_arr, v_e_id_arr = calculate_virtual_excess(current_phases, next_phases)

    # Füge die Überschüsse der aktuellen Phasen in den nächsten Phasen ein
    for i in range(m):
        next_phases[i].append_excess(v_e_start_arr[i], v_e_content_arr[i], False, v_e_id_arr[i])

    # Lösche alte Excess Pakete die durch neue Excess Pakete in der nächsten Phase ersetzt wurden
    for i in range(n):
        if is_unbalanced_array[0, i] and is_unbalanced_array_shifted[i]:
            phases[i].remove_excess(-2)
        elif is_unbalanced_array[0, i] and not is_unbalanced_array_shifted[i]:
            phases[i].remove_excess(-1)

    # Passe Maske an auf die neuen Werte
    for i in range(n):
        is_unbalanced_array[0, i] = is_unbalanced_array_shifted[i]

    return phases, is_unbalanced_array


@njit
def process_phases_njit(phases_typed_list):

    n = len(phases_typed_list)
    """
        Maske mit 2 Spalten (Excess Spalte + Deficit Spalte) und n Zeilen 
        Initial auf True 
        True -> unbalanced
        False -> balanced  
        
    """
    is_unbalanced_array = np.ones((2, n), dtype=np.bool_)

    while True:
        phases_typed_list, is_unbalanced_array = balance_phases_njit(phases_typed_list, is_unbalanced_array)

        """
            Überprüft, ob entweder alle Excess oder alle Deficit Einträge Balanced sind 
            -> Bricht ab 
        """
        excess_row_is_balanced = True
        deficit_row_is_balanced = True

        for i in range(n):

            if is_unbalanced_array[0, i] == True:
                excess_row_is_balanced = False

            if is_unbalanced_array[1, i] == True:
                deficit_row_is_balanced = False

        if excess_row_is_balanced or deficit_row_is_balanced:
            break

        phases_typed_list, is_unbalanced_array = move_overflow_njit(phases_typed_list, is_unbalanced_array)

    return phases_typed_list


def process_phases(energy_excess: np.ndarray, energy_deficit: np.ndarray, start_time_phases,):

    phases_list = List()
    for ex, de, t in zip(energy_excess, energy_deficit, start_time_phases):
        phases_list.append(efes_dataclasses.Phase(ex, de, id=t))

    phases_out = process_phases_njit(phases_list)

    return phases_out
