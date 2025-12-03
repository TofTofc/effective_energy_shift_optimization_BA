import numpy as np
from numba import njit
from numba.typed import List

from versions.new_version_fusion import efes_dataclasses

@njit
def get_next_excess_index(phases, idx, state_mask):
    """
    Returns the idx of the next phase with excess overflow
    """
    n = len(phases)
    i = (idx + 1) % n

    while True:
        if state_mask[i] == 1:
            return i
        i = (i + 1) % n


@njit
def get_next_non_balanced_phase(phases, idx, state_mask):

    """
     Returns the idx of the next phase wich is not balanced
    """
    n = len(phases)
    i = (idx + 1) % n

    while True:
        if state_mask[i] != 0:
            return i
        i = (i + 1) % n

@njit
def move_excess(phases, current_phase_idx, next_phase_idx, max_height_array, state_mask, e_counter, d_counter):

    """
    Moves all uncovered excess packets from current -> next phase

    1. Compute max height of skipped phases (0 if no phases skipped)
    2. Take all uncovered excess packets from current phase
    3. For each of those packets: new start = max(orig_start, max_skipped_height, end_of_last_in_next)
    4. Insert packet into next phase via append_excess
    4.1 merge the new excess in the next phase if possible
    5. Update uncovered excess counters in both phases
    6. Remove transferred packets from current phase
    7. Mark current phase as balanced: state_mask=0, e_counter--, update max_height
    8. Handle new packets in next phase by calling balance_phase
    Returns updated (e_counter, d_counter).
    """

    current_phase = phases[current_phase_idx]
    next_phase = phases[next_phase_idx]

    # Get the max height inbetween the two phases
    if next_phase_idx > current_phase_idx + 1:
        max_height = np.amax(max_height_array[current_phase_idx + 1: next_phase_idx])
    else:
        max_height = 0

    # Get start index of not covered excesses
    n = phases[current_phase_idx].number_of_excess_not_covered
    total = phases[current_phase_idx].size_excess
    start_idx = total - n

    # Iterates over all uncovered excesses
    for idx in range(start_idx, total):

        # Get the excess content from the current uncovered excess
        overflow_content = current_phase.get_energy_excess(idx)

        # Max of the current start height and the max height of all skipped Phases
        overflow_start = max(current_phase.get_starts_excess(idx), max_height)

        can_merge = False

        last_excess_end_height = next_phase.get_starts_excess(-1) + next_phase.get_energy_excess(-1)
        # computed start for the moved packet (before appending)
        excess_start = max(overflow_start, last_excess_end_height)
        # merge conditions:
        # 1. there is at least one uncovered excess in next phase
        # 2. start of moved packet equals end of last excess in next phase
        if (next_phase.number_of_excess_not_covered > 0) and (excess_start == last_excess_end_height):
            can_merge = True

        excess_id = current_phase.get_excess_id(idx)

        if can_merge:
            # Merge: increase energy of last excess in next_phase
            prev_energy = next_phase.get_energy_excess(-1)
            next_phase.set_energy_excess(-1, prev_energy + overflow_content)
            # not increment next_phase.number_of_excess_not_covered because we no new packet
        else:
            # Add Excess to next Phase
            next_phase.append_excess(excess_start, overflow_content, excess_id)
            # Increment the Number of not covered excess packets in the next phase
            next_phase.number_of_excess_not_covered += 1

        # remove one uncovered excess from current phase
        current_phase.number_of_excess_not_covered -= 1

    # separate removal of excesses in current phase
    for idx in range(start_idx, total):
        current_phase.remove_excess(-1)

    # Current phase is now balanced
    state_mask[current_phase_idx] = 0

    # Change the max_height_array entry
    max_height_array[current_phase_idx] = phases[current_phase_idx].get_starts_excess(-1) + phases[current_phase_idx].get_energy_excess(-1)

    e_counter -= 1

    e_counter, d_counter = balance_phase(phases, next_phase_idx, state_mask, max_height_array, e_counter, d_counter)

    return e_counter, d_counter


@njit
def balance_phase(phases, i, state_mask, max_height_array, e_counter, d_counter):
    """
    Balances newly moved excess packets for the phase

    Behaviour:
    0. If there is no uncovered deficit block, nothing to do for balancing (number_of_excess_not_covered was already changed properly)

    1. Start at the first not covered excess and iterate over all of them
    2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess
    3. For each current excess vs the uncovered deficit one of 3 happens:
        a: excess < deficit:
            - Split the deficit: the lower part is matched to the excess (becomes covered)
              the upper part remains as a smaller deficit starting at the end of the matched pair
            - number_of_excess_not_covered --
            - Move to the next excess and continue
       b: excess > deficit:
            - Split the excess: the lower part covers the deficit, the remaining excess stays
              (with its start adjusted to the end of the covered portion)
            - Update: deficit counter --, excess counter++, state_mask[i] = 1
            - return
       c: excess == deficit:
            - If last uncovered excess: state_mask[i] = 0, deficit counter --
              set max_height_array[i] to the end height of this block and return
              number_of_excess_not_covered --
            - If not last excess: state_mask[i] = 1, deficit counter--, excess counter++
              number_of_excess_not_covered -- and return

    4. Return updated (e_counter, d_counter)
    """

    phase = phases[i]

    # 0. no uncovered deficit block -> nothing to do
    if state_mask[i] == 1:
        return e_counter, d_counter

    else:
        #  1. Start at the first not covered excess and iterate over all of them
        n = phase.number_of_excess_not_covered
        total = phase.size_excess
        start_idx = total - n

        for idx in range(start_idx, total):

            # 2. Raise the start of the last deficit packet (uncovered one) to the start of the current excess
            phase.set_starts_deficit(-1, phase.get_starts_excess(idx))

            # 3. For each current excess vs the uncovered deficit one of 3 happens:

            # a: excess < deficit:
            if phase.get_energy_excess(idx) < phase.get_energy_deficit(-1):
                # Split the deficit

                # New start is excess height + start
                new_start = phase.get_starts_excess(idx) + phase.get_energy_excess(idx)

                # Remaining deficit is current deficit - energy excess
                energy_remaining = phase.get_energy_deficit(-1) - phase.get_energy_excess(idx)

                phase.append_deficit(new_start, energy_remaining)

                # Change Deficit of lower packet
                phase.set_energy_deficit(-2, phase.get_energy_excess(idx))

                # number_of_excess_not_covered--
                phase.number_of_excess_not_covered -= 1

                # Move to the next excess and continue
                continue

            # b: excess > deficit
            elif phase.get_energy_excess(idx) > phase.get_energy_deficit(-1):

                excess_energy = phase.get_energy_excess(idx)
                deficit_energy = phase.get_energy_deficit(-1)
                deficit_start = phase.get_starts_deficit(-1)

                # computed start for the remaining excess
                new_start = deficit_start + deficit_energy
                energy_remaining = excess_energy - deficit_energy

                # set lower packet to cover the deficit
                phase.set_energy_excess(idx, deficit_energy)

                # insert remaining excess after the covered excess and NOT at the end to keep correct sequence
                # phase.append_excess(new_start, energy_remaining, phase.get_excess_id(idx))
                phase.insert_excess(idx + 1, new_start, energy_remaining, phase.get_excess_id(idx))

                # update counters and mark phase as still having excess
                d_counter -= 1
                e_counter += 1
                state_mask[i] = 1

                # return updated counters
                return e_counter, d_counter

            # c: excess == deficit:
            else:

                # If last (uncovered) excess
                if idx == total-1:

                    #state_mask[i] = 0, deficit counter --
                    state_mask[i] = 0
                    d_counter -= 1

                    # set max_height_array[i] to the end height of this block
                    max_height_array[i] = phase.get_energy_excess(idx) + phase.get_starts_excess(idx)

                    # number_of_excess_not_covered --
                    phase.number_of_excess_not_covered -= 1

                    return e_counter, d_counter

                # If not last (uncovered) excess
                else:

                    # state_mask[i] = 1, deficit counter--, excess counter++
                    state_mask[i] = 1
                    d_counter -= 1
                    e_counter += 1

                    # number_of_excess_not_covered --
                    phase.number_of_excess_not_covered -= 1

                    return e_counter, d_counter

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
            max_height_array[i] = phases[i].get_starts_excess(-1) + phases[i].get_energy_excess(-1)


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

    # Return when we either start with no Excess or no Deficit
    if e_counter == 0 or d_counter == 0:
        return phases

    # start with an excess overflow right away
    idx = get_next_excess_index(phases, 0, state_mask)
    start_idx = idx

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

        #2. Excess = Deficit (cant happen)
        # Nothing to move here

        #3. Excess < Deficit (cant happen)
        # Nothing to move here

        # Index goes to the next Excess
        idx = get_next_excess_index(phases, idx, state_mask)

    #print_helper(phases, e_counter, d_counter, max_height_array, state_mask)

    return phases

# Wenn 3 Excess Pakete das letzte Deficit Paket füllen,
# dann kann es sein, das nicht alle der 3 E überschüsse im original so weit gekommen wären
# Wenn nur 2 der 3 E Pakete gereicht hätten, dann wäre das letzte E Paket einen Schritt hinter den anderen

# Wenn die letzte Phase eine E Überschuss Phase ist aber davor alle D Überschüsse ausgeglichen wurden,
# dann hat sich das letzte E Packet nicht bewegt obwohl es sich im original bewegt hätte
# Idee: Speichere die Schrittzahl eines Paketes und gehe die Schritte die fehlen am Ende

# Die Move Anzahl eines Excess Paketes kann zu hoch oder zu niedrig sein
# und damit das Packet zu weit oder zu kurz weitergereicht werden
# Idee: Anzahl der Bewegungen eines Paketes speichern und die Info benutzen um am Ende aufzuräumen?
# Problem: Evt sind die Höheninfos dabei ein Problem

# Die Höheninfo der Pakete geht teilweise verloren, wenn an den Anfang zurück gesprungen wird
# Bsp 1. Phase ist E Phase. Dann loopt man von der letzten Phase zurück zum Anfang
# Dabei wird von der 1. Phase nur die Höhe des Balancierten Teiles betrachtet, nicht die Höhe des Excess Überschusses
# Die Höhe des Überschusses wäre beim original aber relevant, falls die letzte Phase einen Überschuss hat
# und dieser dann ja auf die Höhe des Überschusses platziert wird

# Laufzeit wird ab ca 1 millionen langsamer wegen RAM Mangel und Swapping.
# Variabel je nach Auslastung des RAMs durch andere Nutzung
# Am besten wäre konstante RAM Nutzung für Vergleichbarkeit

# Fusion weglassen um Weg der einzel Pakete zu tracken?
# Problem: Worst Case nur mit Fusion optimal
# Idee: Fange im Worst Case in der Mitte an und gehe mit dem Index immer 1 zurück
# Damit werden dann in der Mitte balancierte Phasen erzeugt
# Diese können dann geskipped werden
# Problem: Im Allgemeinem Fall nicht sinnvoll

# Evt den Startpunkt verschieben
# Beim anfang (der größten / einer) Excess Kette starten
# Aktuell kann z.b. der Eintrag vor dem Start auf den Start einen Überschuss übertragen
# Wenn man so startet, das man keinen Excess Übertragen bekommt, löst das evt Probleme

# Evt Mischung aus original und neuer Version z.B. erst original für x Schritte und dann die neue Version

# worst case result stimmt. average case ähnlich aber definitiv nicht gleich

@njit
def process_phases(excess_array, deficit_array, start_times):

    n = len(excess_array)
    phases_list = List()
    for i in range(n):
        phase = efes_dataclasses.Phase(excess_array[i], deficit_array[i], start_times[i])
        phases_list.append(phase)

    result = process_phases_njit(phases_list)

    return result





@njit
def print_helper(phases, e_counter, d_counter, max_height_array, state_mask):

    for phase in phases:
        print("Energy Excess: ")
        print(phase.get_energy_excess_all())
        print("Energy Excess Starts: ")
        print(phase.get_starts_excess_all())
        print("Energy Deficit: ")
        print(phase.get_energy_deficit_all())
        print("Energy Deficit Starts: ")
        print(phase.get_starts_deficit_all())
        print("Number of Excess Not Covered: ")
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

@njit
def dbg_sum_excess(phases):
    s = 0.0
    for i, p in enumerate(phases):
        for v in p.get_energy_excess_all():
            s += v
    print("Sum of excess: ", s)
    return s

@njit
def dbg_sum_deficit(phases):
    s = 0.0
    for i, p in enumerate(phases):
        for v in p.get_energy_deficit_all():
            s += v
    print("Sum of deficit: ", s)
    return s