# parameters
# eps: for connect disconnect
# tau: for interruption
import numpy as np

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from numba import njit


class Event:
    """Represents a topological event (connect, disconnect, appear, disappear)
    in the sweep-line simulation.

    Attributes:
        event (str): The type of event ("appear", "connect", "disconnect", "disappear").
        trajectory (Optional[int]): The ID of the interacting trajectory, if any.
        t (Optional[int]): The point index where the event occurs.
    """

    def __init__(self, event: str, trajectory: Optional[int] = None, t: Optional[int] = None):
        self.event = event
        self.trajectory = trajectory
        self.t = t


@njit(cache=True)
def check_epsilon_distance(p1: np.ndarray, p2: np.ndarray, eps_sq: float) -> bool:
    """Check if the squared Euclidean distance between two 3D points is <= eps_sq."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return (dx * dx + dy * dy + dz * dz) <= eps_sq


@njit(cache=True)
def _try_expand_direction(t_source: np.ndarray, t_target: np.ndarray, flag_source: np.ndarray, flag_target: np.ndarray,
                          anchor_idx: int, first_target: int, last_target: int,
                          eps_sq: float, bounds_checks: np.ndarray) -> Tuple[int, bool, bool]:
    """Try to expand the matched region by matching anchor_idx in t_source
    against candidate indices near [first_target, last_target] in t_target.

    Returns:
        (matched_target_idx, is_candidate, found_match)
    """
    candidates = np.array([first_target - 1, last_target + 1,
                           first_target + 1, last_target - 1], dtype=np.int32)

    for i in range(4):
        cand_idx = candidates[i]
        if (bounds_checks[i]
                and not flag_target[cand_idx]
                and check_epsilon_distance(t_source[anchor_idx],
                                           t_target[cand_idx], eps_sq)):
            return cand_idx, True, True

    # Fallback: scan existing matched range for an epsilon match.
    for each in range(first_target, last_target + 1):
        if check_epsilon_distance(t_source[anchor_idx],
                                  t_target[each], eps_sq):
            return each, False, True

    return -1, False, False


@njit(cache=True)
def _apply_target_boundary_update(matched: int, first_target: int, last_target: int,
                                  is_candidate: bool) -> Tuple[int, int]:
    """Compute updated target boundaries after a candidate match."""
    if not is_candidate:
        return first_target, last_target
    if matched == first_target - 1:
        return first_target - 1, last_target
    elif matched == last_target + 1:
        return first_target, last_target + 1
    elif matched == first_target + 1:
        return first_target + 1, last_target
    elif matched == last_target - 1:
        return first_target, last_target - 1
    return first_target, last_target


def find_connect_disconnect_events(
    t1_id: int, t2_id: int, t1: np.ndarray, t2: np.ndarray, eps: float, alpha: float
) -> Tuple[Dict[int, List[Event]], Dict[int, List[Event]]]:
    """Find connect and disconnect events between two trajectories.

    Scans the points of two trajectories to find segments where they are
    within epsilon distance of each other. Returns event dictionaries for
    each trajectory detailing when they connect and disconnect.

    Args:
        t1_id: Identifier for the first trajectory.
        t2_id: Identifier for the second trajectory.
        t1: Array of 3D points for the first trajectory.
        t2: Array of 3D points for the second trajectory.
        eps: Epsilon distance threshold.
        alpha: Persistence distance threshold (for filtering close events).

    Returns:
        A tuple of two dictionaries (events_t1, events_t2), mapping point
        indices to a list of Event objects.
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    events_t1: Dict[int, List[Event]] = {}
    events_t2: Dict[int, List[Event]] = {}
    ti = 0
    flag_t1 = np.zeros(len(t1), dtype=np.bool_)
    flag_t2 = np.zeros(len(t2), dtype=np.bool_)

    eps_sq = eps * eps
    alpha_sq = alpha * alpha

    while ti < (len(t1)):
        tj = 0
        while tj < (len(t2)):
            if not flag_t1[ti] and not flag_t2[tj] and check_epsilon_distance(t1[ti], t2[tj], eps_sq):
                # print(ti,"Start",tj)
                flag_t1[ti] = True
                flag_t2[tj] = True
                flag_insert = True
                first_i = ti
                last_i = ti
                first_j = tj
                last_j = tj
                while (flag_insert and last_j < len(t2) and first_j >= 0):
                    flag_insert = False

                    # case first_i - 1 insert:
                    if (first_i - 1 >= 0) and not flag_t1[first_i - 1]:
                        bounds = np.array([first_j - 1 >= 0,
                                           last_j + 1 < len(t2),
                                           first_j + 1 < len(t2),
                                           last_j - 1 >= 0], dtype=np.bool_)
                        matched, is_cand, found = _try_expand_direction(
                            t1, t2, flag_t1, flag_t2,
                            first_i - 1, first_j, last_j, eps_sq, bounds)
                        if found:
                            first_i -= 1
                            flag_t1[first_i] = True
                            flag_t2[matched] = True
                            first_j, last_j = _apply_target_boundary_update(
                                matched, first_j, last_j, is_cand)
                            flag_insert = True

                    # case last_i + 1 insert:
                    if (last_i + 1 < len(t1)) and not flag_t1[last_i + 1]:
                        bounds = np.array([first_j - 1 > 0,
                                           last_j + 1 < len(t2),
                                           first_j + 1 < len(t2),
                                           last_j - 1 > 0], dtype=np.bool_)
                        matched, is_cand, found = _try_expand_direction(
                            t1, t2, flag_t1, flag_t2,
                            last_i + 1, first_j, last_j, eps_sq, bounds)
                        if found:
                            last_i += 1
                            flag_t1[last_i] = True
                            flag_t2[matched] = True
                            first_j, last_j = _apply_target_boundary_update(
                                matched, first_j, last_j, is_cand)
                            flag_insert = True

                    # case first_j - 1 insert:
                    if (first_j - 1 >= 0) and not flag_t2[first_j - 1]:
                        bounds = np.array([first_i - 1 >= 0,
                                           last_i + 1 < len(t1),
                                           first_i + 1 < len(t1),
                                           last_i - 1 >= 0], dtype=np.bool_)
                        matched, is_cand, found = _try_expand_direction(
                            t2, t1, flag_t2, flag_t1,
                            first_j - 1, first_i, last_i, eps_sq, bounds)
                        if found:
                            first_j -= 1
                            flag_t2[first_j] = True
                            flag_t1[matched] = True
                            first_i, last_i = _apply_target_boundary_update(
                                matched, first_i, last_i, is_cand)
                            flag_insert = True

                    # case last_j + 1 insert:
                    if (last_j + 1 < len(t2)) and not flag_t2[last_j + 1]:
                        bounds = np.array([first_i - 1 >= 0,
                                           last_i + 1 < len(t1),
                                           first_i + 1 < len(t1),
                                           last_i - 1 >= 0], dtype=np.bool_)
                        matched, is_cand, found = _try_expand_direction(
                            t2, t1, flag_t2, flag_t1,
                            last_j + 1, first_i, last_i, eps_sq, bounds)
                        if found:
                            last_j += 1
                            flag_t2[last_j] = True
                            flag_t1[matched] = True
                            first_i, last_i = _apply_target_boundary_update(
                                matched, first_i, last_i, is_cand)
                            flag_insert = True

                # connect event
                if events_t1.get(first_i):
                    events_t1[first_i].append(Event("connect", t2_id, first_j))
                else:
                    events_t1[first_i] = [Event("connect", t2_id, first_j)]

                if events_t2.get(first_j):
                    events_t2[first_j].append(Event("connect", t1_id, first_i))
                else:
                    events_t2[first_j] = [Event("connect", t1_id, first_i)]
                # disconnect event
                if last_i + 1 < len(t1):
                    if events_t1.get(last_i + 1):
                        events_t1[last_i +
                                  1].append(Event("disconnect", t2_id, last_j + 1))
                    else:
                        events_t1[last_i +
                                  1] = [Event("disconnect", t2_id, last_j + 1)]
                if last_j + 1 < len(t2):
                    if events_t2.get(last_j + 1):
                        events_t2[last_j +
                                  1].append(Event("disconnect", t1_id, last_i + 1))
                    else:
                        events_t2[last_j +
                                  1] = [Event("disconnect", t1_id, last_i + 1)]

                ti = last_i
                break
            else:
                tj += 1
        ti += 1

    # code to take care of alpha parameter
    # Explicitly sort the dictionaries by sweep-line index (key) to ensure the
    # alpha filter correctly compares spatially consecutive events.
    events_t1 = {k: events_t1[k] for k in sorted(events_t1.keys())}
    events_t2 = {k: events_t2[k] for k in sorted(events_t2.keys())}

    keys_to_remove1 = []
    keys_to_remove2 = []
    event_pos1 = [t1[key] for key in events_t1.keys()]
    event_pos2 = [t2[key] for key in events_t2.keys()]
    # check epsilon distance from consecutive points, if less than alpha then remove
    for i in range(len(event_pos1)):
        if i + 1 < len(event_pos1) and check_epsilon_distance(event_pos1[i], event_pos1[i + 1], alpha_sq):
            keys_to_remove1.append(list(events_t1.keys())[i])
            keys_to_remove1.append(list(events_t1.keys())[i + 1])
    for i in range(len(event_pos2)):
        if i + 1 < len(event_pos2) and check_epsilon_distance(event_pos2[i], event_pos2[i + 1], alpha_sq):
            keys_to_remove2.append(list(events_t2.keys())[i])
            keys_to_remove2.append(list(events_t2.keys())[i + 1])

    # remove all the keys in events_t1 and events_t2
    for key in list(set(keys_to_remove1)):
        del events_t1[key]
    for key in list(set(keys_to_remove2)):
        del events_t2[key]

    return events_t1, events_t2
