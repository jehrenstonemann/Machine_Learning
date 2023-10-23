import heapq
import numpy as np


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    result = 0
    for i in range(3):
        for j in range(3):
            if np.reshape(from_state, (3, 3))[i][j] == 0:
                continue
            x = int(np.where(np.reshape(to_state, (3, 3)) == np.reshape(from_state, (3, 3))[i][j])[0])
            y = int(np.where(np.reshape(to_state, (3, 3)) == np.reshape(from_state, (3, 3))[i][j])[1])
            result += abs(i - x) + abs(j - y)
    return result


def print_succ(state):
    succ_states = get_succ(state)
    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    i1_row, i1_col = int(state.index(0) / 3), state.index(0) % 3
    i2_row, i2_col = int(state.index(0, state.index(0) + 1) / 3), state.index(0, state.index(0) + 1) % 3
    result = []
    for i, s in enumerate(state):
        if s != 0:
            c1 = (i1_row - 1 == int(i / 3) or int(i / 3) == i1_row + 1) and (i1_col == i % 3)
            c2 = (i1_col - 1 == i % 3 or i % 3 == i1_col + 1) and (i1_row ==int(i / 3))
            c3 = (i2_row - 1 == int(i / 3) or int(i / 3) == i2_row + 1) and (i2_col == i % 3)
            c4 = (i2_col - 1 == i % 3 or i % 3 == i2_col + 1) and (i2_row == int(i / 3))
            if c1 or c2:
                succ1 = state[:]
                succ1[i] = 0
                succ1[state.index(0)] = s
                result.append(succ1)
            if c3 or c4:
                succ2 = state[:]
                succ2[i] = 0
                succ2[state.index(0, state.index(0) + 1)] = s
                result.append(succ2)
    return sorted(result)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    pq = []
    p_index = -1
    heapq.heappush(pq, (0 + get_manhattan_distance(state), state, (0, get_manhattan_distance(state), p_index)))
    max_length = 0
    open = {}
    close = {}
    parent_dict = {}
    open[str(state)] = 0
    parent_dict[str(state)] = -1
    # while length pq not reach zero
    while len(pq) > 0:
        max_length = max(max_length, len(pq))
        pop = heapq.heappop(pq)
        if str(pop[1]) in open:
            open.pop(str(pop[1]))
        close[str(pop[1])] = pop[2][0]
        if pop[1] == goal_state:
            path = []
            result = 0
            path.append(pop[1])
            par = parent_dict[str(pop[1])]
            while par[2][2] != -1:
                path.append(par[1])
                par = parent_dict[str(par[1])]
            path.append(state)
            for i in range(len(path) - 1, -1, -1):
                print(path[i], "h=" + str(get_manhattan_distance(path[i])), "moves: " + str(result))
                result += 1
            print("Max queue length: " + str(max_length))
            exit()
        else:
            for element in get_succ(pop[1]):
                g = pop[2][0] + 1
                if str(element) not in open and str(element) not in close:
                    p_index += 1
                    heapq.heappush(pq, (
                        g + get_manhattan_distance(element), element, (g, get_manhattan_distance(element), p_index)))
                    open[str(element)] = g
                    parent_dict[str(element)] = pop
                elif str(element) in open:
                    if g < open[str(element)]:
                        parent_dict[str(element)] = pop
                    p_index += 1
                    heapq.heappush(pq, (
                        g + get_manhattan_distance(element), element, (g, get_manhattan_distance(element), p_index)))
                    open[str(element)] = g
                else:
                    if g < close[str(element)]:
                        p_index += 1
                        heapq.heappush(pq, (
                            g + get_manhattan_distance(element), element,
                            (g, get_manhattan_distance(element), p_index)))
                        open[str(element)] = g
                        parent_dict[str(element)] = pop
