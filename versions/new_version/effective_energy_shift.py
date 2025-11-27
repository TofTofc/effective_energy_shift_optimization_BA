import sys

import numpy as np
from numba import njit
from numba.typed import List
from scipy.stats import energy_distance

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
def get_next_non_balanced_phase(phases, idx, state_mask):

    """
     Returns the idx of the next phase wich is not balanced
    """
    i = idx + 1
    n = len(phases)

    while True:
        if state_mask[i] != 0:
            return i
        i = (i + 1) % n

@njit
def move_excess(phases, current_phase_idx, next_phase_idx, max_height_array, state_mask, e_counter, d_counter):
    """
    Move excess from current phase to next phase
    The starting position of the new Excess in next phase is at least max height
    Max height is extracted from max height array between current phase idx and next phase idx
    The current phase is now perfectly balanced ( e_counter - 1 and state_mask entry changed to 0)
    Next Phase might become balanced if it was a Deficit Overflow which was fully filled
    """

    current_phase = phases[current_phase_idx]
    next_phase = phases[next_phase_idx]

    # Get the max height inbetween the two phases
    max_height = np.amax(max_height_array[current_phase_idx + 1 : next_phase_idx])

    # Overflow content is Overflow Excess
    overflow_content = current_phase.get_energy_excess(-1)

    # Max of the current start height and the max height of all skipped Phases
    overflow_start = max (current_phase.get_starts_excess(-1), max_height)

    # Consider the height of the end of the last Excess in the next phase
    blocking_excess_content = next_phase.get_energy_excess(-1)
    blocking_excess_start = next_phase.get_starts_excess(-1)

    excess_start = max(overflow_start, blocking_excess_start + blocking_excess_content)
    excess_content = overflow_content
    excess_id = current_phase.get_excess_id(-1)

    # Add Excess to next Phase
    next_phase.append_excess(excess_start, excess_content, excess_id)

    # Now 4 things can happen:

    # 1. Our Excess Overflow gets added to an Excess Overflow Packet
    # -> save info about how many Excess Packet have to be moved (since you cant fuse them
    # Due to Possible gaps)

    if state_mask[next_phase_idx] == 1:

        next_phase.number_of_excess_not_covered += 1

    else:
        # 2. Our Excess Overflow gets added to an Deficit Overflow Packet and its not enough to cover it
        # -> Create new Deficit Entry that covers the newly added Excess

        if  overflow_content < next_phase.get_energy_deficit(-1):

            e_counter, d_counter = balance_phase(phases, next_phase_idx, state_mask, max_height_array, e_counter, d_counter)

        # 3. As 2 but it perfectly covers the Deficit
        # -> Next phase is now an Balanced Phase

        elif  overflow_content == next_phase.get_energy_deficit(-1):

            state_mask[next_phase_idx] = 0

        # 4. As 2 but we have more Excess than Deficit in next phase
        # -> next phase is now an Excess Phase
        # + Split the incoming Excess In 2. One covers the deficit in next phase the other one is the unmatched Excess

        else:
            e_counter, d_counter = balance_phase(phases, next_phase_idx, state_mask, max_height_array, e_counter, d_counter)


    # Remove Excess from current Phase
    current_phase.remove_excess(-1)

    # Current phase is now balanced
    state_mask[current_phase_idx] = 0
    e_counter -= 1


    return e_counter, d_counter


@njit
def balance_phase(phases, i, state_mask, max_height_array, e_counter, d_counter):
    """
    Balances the last Excess and Deficit of a Phase with each other.
    """

    # TODO: UMSCHREIBEN SODASS ICH NICHT DIE IF ELIF UND ELSE GANZ OBEN HABE SONDERN WEITER UNTEN IM ABLAUF

    n = phases[i].number_of_excess_not_covered
    all_excess = phases[i].get_energy_excess_all()

    total_unfilled_excess = np.sum(all_excess[len(all_excess) - n:]) if n > 0 else 0.0

    excess = phases[i].get_energy_excess(-1)
    deficit = phases[i].get_energy_deficit(-1)

    total = phases[i].size_excess


    if total_unfilled_excess > deficit:

        if state_mask[i] != 1:
            if state_mask[i] == -1:
                d_counter -= 1
            state_mask[i] = 1
            e_counter += 1

        # Balancing:
        """
        # Create new Excess paket
        excess_start = deficit
        excess_content = excess - deficit
        excess_id = phases[i].id

        phases[i].append_excess(excess_start, excess_content, excess_id)

        # change energy Excess of old Excess entry to match deficit
        phases[i].energy_excess[-2] = deficit
        """

        # TODO

    elif total_unfilled_excess < deficit:

        if state_mask[i] != -1:
            if state_mask[i] == 1:
                e_counter -= 1
            state_mask[i] = -1
            d_counter += 1

        # Balancing:
        """
        # Create new Deficit paket
        deficit_start = excess
        deficit_content = deficit - excess

        phases[i].append_deficit(deficit_start, deficit_content)

        # change energy Deficit of old Deficit entry to match excess
        phases[i].energy_deficit[-2] = excess
         """

        start_idx = total - n
        remaining_deficit = deficit

        # Iterates over all unfitted excesses
        for idx in range(start_idx, total):

            current_excess = phases[i].get_energy_excess[idx]
            current_excess_start = phases[i].get_starts_excess(idx)

            # Sets the starting height of the last deficit to the height of the current unfitted excess
            phases[i].set_starts_deficit(-1, phases[i].get_starts_excess(idx))

            # Sets the Deficit of the last deficit to the current_excess
            phases[i].set_energy_deficit(-1, current_excess)

            energy_remaining = remaining_deficit - current_excess

            if idx == total - 1:
                new_start = current_excess + current_excess_start

            else:
                # gets changed in next iteration to the correct value
                new_start = 0

            # Creates new unfilled deficit
            phases[i].append_deficit(new_start, energy_remaining)

            remaining_deficit -= current_excess
            phases[i].number_of_excess_not_covered -= 1


    else:

        #TODO: Passe die Höhe an. Die Deficit Blöcke müssen auf der selben höhe sein wie die Excess Blöcke

        if state_mask[i] == -1:
            d_counter -= 1
        if state_mask[i] == 1:
            e_counter -= 1

        state_mask[i] = 0
        max_height_array[i] = excess + phases[i].get_starts_excess(-1)

        phases[i].number_of_excess_not_covered = 0

    return e_counter, d_counter

@njit
def init(phases, state_mask, max_height_array):
    """
    Fills out the state mask:
        -  1 for excess > deficit
        - -1 for excess < deficit
        -  0 for excess = deficit

    For each 0:
    Also sets the correct height entry for max_height_array

    Returns the tuple: (Number of 1 in total, Number of -1 in total)
    """

    e_counter = 0
    d_counter = 0

    for i in range(len(phases)):

        current_excess_array_size = len(phases[i].get_energy_excess_all())
        current_deficit_array_size = len(phases[i].get_energy_deficit_all())

        if current_excess_array_size > current_deficit_array_size:
            e_counter += 1
            state_mask[i] = 1

        elif current_excess_array_size < current_deficit_array_size:
            d_counter += 1
            state_mask[i] = -1

        else:
            state_mask[i] = 0
            max_height_array[i] = phases[i].get_energy_excess(-1)


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

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return phases

    # start with an excess overflow right away
    idx = get_next_excess_index(phases, 0, state_mask)
    start_idx = idx

    #TODO: bis hier getestet

    while True:

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        # For each Phase there are 3 possibilities

        #1. Excess > Deficit
        next_phase_idx = get_next_non_balanced_phase(phases, idx, state_mask)

        # Moves the Excess from the current Phase to the next non perfectly balanced phase
        e_counter, d_counter = move_excess(phases, idx, next_phase_idx, max_height_array, state_mask, e_counter, d_counter)

        # Stop when either no more Excesses to move or no more Deficits to fill
        if e_counter == 0 or d_counter == 0:
            break

        #2. Excess = Deficit
        # Nothing to move here

        #3. Excess < Deficit:
        # Nothing to move here

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





@njit
def print_helper(phases, e_counter, d_counter, max_height_array, state_mask):

    for phase in phases:
        print(phase.get_energy_excess_all())
        print(phase.get_starts_excess_all())
        print(phase.get_energy_deficit_all())
        print(phase.get_starts_deficit_all())
        print(phase.number_of_excess_not_covered)
        print("_____________________")
    print("_____________________")
    print("e_counter: ", e_counter)
    print("d_counter: ", d_counter)
    print("_____________________")
    print("max_height_array: ", max_height_array)
    print("_____________________")
    print("state_mask: ", state_mask)

    return 0