import sys

import numpy as np
from numba import njit
from numba.typed import List

from versions.new_version import efes_dataclasses

@njit
def get_next_excess_index(phases, idx, state_mask):
    """
    Returns the idx of the next phase with excess overflow
    """
    i = idx + 1
    n = len(phases)

    while True:
        if state_mask[i] == 1:
            return i
        i = (i + 1) % n


@njit
def get_next_non_balanced_phase(phases, idx, is_balanced_mask):

    """
    Returns the next non balanced phase starting from idx using is_balanced_mask
    """
    sys.exit("NOCH NICHT IMPLEMENTIERT")

@njit
def move_excess(current_phase, next_phase, max_height):
    """
    Move excess from current phase to next phase
    The starting position of the new Excess in next phase is at least max height high
    After this, the current phase is now perfectly balanced
    """

    sys.exit("NOCH NICHT IMPLEMENTIERT")

@njit
def init(phases, state_mask, max_height_array):
    """
    Fills out the state mask:
        -  1 for excess > deficit
        - -1 for excess < deficit
        -  0 for excess = deficit

    For each 0:
    Also sets the correct height entry for max_height_array

    Balances all Excesses and Deficits:
        - If E = D: Nothing
        - If E > D: 2 Excess Entries in the Excess list one covers the Deficit one is the overflow Excess
        - If E < D: See above

    Returns the tuple: (Number of 1 in total, Number of -1 in total)
    """

    e_counter = 0
    d_counter = 0

    for i in range(len(phases)):

        excess  = phases[i].get_energy_excess(-1)
        deficit = phases[i].get_energy_deficit(-1)

        if excess > deficit:
            state_mask[i] = 1
            e_counter += 1

            # Balancing:

            # Create new Excess paket
            excess_start = deficit
            excess_content = excess - deficit
            excess_id = phases[i].id

            phases[i].append_excess(excess_start, excess_content, excess_id)

            # change energy Excess of old Excess entry to match deficit
            phases[i].energy_excess[0] = deficit

        elif excess < deficit:
            state_mask[i] = -1
            d_counter += 1

            # Balancing:

            # Create new Deficit paket
            deficit_start = excess
            deficit_content = deficit - excess

            phases[i].append_deficit(deficit_start, deficit_content)

            # change energy Deficit of old Deficit entry to match excess
            phases[i].energy_deficit[0] = excess
        else:
            state_mask[i] = 0
            max_height_array[i] = excess

    return e_counter, d_counter

@njit
def process_phases_njit(phases):

    n = len(phases)

    # Mask for the state of the phases:
    #  1 = Excess > Deficit
    #  0 = Excess = Deficit
    # -1 = Excess < Deficit
    state_mask = np.zeros(n)

    # Counters for how many E > D and E < D we have
    e_counter = 0
    d_counter = 0

    # Saves the max_height of all balanced Phases
    max_height_array = np.zeros(n)

    # Provides the initial states for each Phase object and balances them
    e_counter, d_counter = init(phases, state_mask, max_height_array)

    # TODO: FÜR DAS ERSTE EXCESS ÜBERSCHUSS PHASE OBJEKT MUSS MAN DEN VIRTUAL EXCESS SPEICHERN, DAMIT WENN AUF DIE PHASE
    # TODO: EXCESS DRAUF KOMMT MAN DIE KORREKTE HÖHE HAT

    # start with an excess overflow right away
    idx = get_next_excess_index(phases, 0, state_mask)
    start_idx = idx

    while True:

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # For each Phase there are 3 possibilities

        #1. Excess > Deficit
        if state_mask[idx] == 1:

            current_phase = phases[idx]
            next_phase = get_next_non_balanced_phase(phases, idx, state_mask)

            # Moves the Excess from the current Phase to the next non perfectly balanced phase
            # max height is the max height of the skipped balanced phases
            # TODO: THIS ALSO CAN REDUCE D_COUNTER AND BALANCE AN DEFICIT NEXT PHASE OR IS NOT ENOUGH TO FILL IT
            move_excess(phases[current_phase, next_phase, max_height_array[idx]])

            # current phase is now balanced => reduce e_counter
            state_mask[idx] = 0
            e_counter -= 1

        #2. Excess = Deficit
        if state_mask[idx] == 0:

            # Not needed because every balanced Phase gets filled in initially
            # or gets marked by E > D case
            #state_mask[idx] = 0
            pass

        #3. Excess < Deficit:
        # Nothing to move here so pass
        else:
            pass

        # Index goes to the next Excess
        idx = get_next_excess_index(phases, idx, state_mask)

    return phases

# TODO: PROBLEM LISTE
# - TODO: WENN DAS PROGRAMM DEN LETZTEN DEFICIT FÜLLT KANN ES SEIN, DAS DAFÜR NICHT ALLE EXCESSE IN DER GENUTZTEN EXCESS GRUPPE BENUTZT WERDEN,
#   TODO: FALLS DIES PASSIERT, KÖNNEN POSITIONEN DER EXCESSE FALSCH SEIN, DA SIE IN DER NORMALEN IMPLEMENTIERUNG EIG NOCH NICHT ES BIS HIERHIN GESCHAFFT HÄTTEN
# Idee: Speichere für jedes Excess Packet ab, wo es herkam um es dann dahin zurück zu schicken, wo es mit der eig implementierung hingekommen wäre
# Problem: Die Höhe dieses Paketes wäre dann wieder potentiell anderes als seine aktuelle Position
# Idee: Speichere den Verlauf der Höhen des Paketes ab. Nehem Daraus die richtige Höhe

# - TODO: THEORETISCH KANN DER ALGORITHMUS BEENDET WERDEN BEVOR JEDES EXCESS PACKET SICH BEWEGT HAT
#   TODO: DAMIT BEWEGEN SICH DIESE NICHT ANGEFASSEN EXCESS PACKETE NIEMALS WEITER UND SIND AUF IHRER START POSITION
#   TODO: OBWOHL SIE EIG WEITER SEIN SOLLTEN
# Idee: Speichere Die Anzahl der Moves ab die das Packet mit dem längsten Weg gegangen ist und gehen den Weg dann für alle nicht bewegten Packete

# Zentrales Problem: Die Move Anzahl eines Excess Packetes kann zu hoch oder zu niedrig sein
# und damit das Packet zu weit oder zu kurz weitergereicht werden

@njit
def process_phases(excess_array, deficit_array, start_times):

    n = len(excess_array)
    phases_list = List()
    for i in range(n):
        phase = efes_dataclasses.Phase(excess_array[i], deficit_array[i], start_times[i], 10)
        phases_list.append(phase)

    return process_phases_njit(phases_list)
