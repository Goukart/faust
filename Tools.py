import numpy as np
import time       # Measure execution time


def columnify(iterable):
    # First convert everything to its str
    strings = [str(x) for x in iterable]  # repr seems to be for debugging
    # Now pad all the strings to match the widest
    widest = max(len(x) for x in strings)
    padded = [x.ljust(widest) for x in strings]
    return padded


def colprint(iterable, width=120):
    columns = columnify(iterable)
    colwidth = len(columns[0]) + 2
    perline = (width - 4) // colwidth
    for i, column in enumerate(columns):
        print(column, end='\t')
        if i % perline == perline - 1:
            print('\n', end='')
    print("\n")


def error_print(function_name, error_message):
    print("---------------------------------------")
    # ToDo format so: ---- // <name> \\ ---- [...] ---- \\ <name> // ----
    print(f"in {function_name}")
    print(error_message)
    print("---------------------------------------")


def array_is_equal(array_1, array_2):
    n_array_1 = len(array_1)
    n_array_2 = len(array_2)

    # Guarding ifs
    if n_array_1 != n_array_2:
        error_print(array_is_equal.__name__, f"Array not same lengths: {n_array_1} ≠ {n_array_2}")
        return False

    # sort both the arrays
    array_1.sort()
    array_2.sort()

    # traverse each index of arrays
    for i in range(n_array_1):
        if (type(array_1[i]) in (list, np.ndarray)) and (type(array_2[i]) in (list, np.ndarray)):
            if not array_is_equal(array_1[i], array_2[i]):
                return False
        else:
            # for same index if the value in the sorted arrays are different return false
            if array_1[i] != array_2[i]:
                error_print(array_is_equal.__name__, f"Item at i={i} not equal: [{array_1[i]}] ≠ [{array_2[i]}]")
                return False

    # if none of the above conditions satisfied return true
    return True


# pass two functions and pit them against each other to see which one is faster
# ToDo, make variable parameter list?
def cmp_runtime(old, opt, parameter, n=10):
    times_old = 0
    times_opt = 0
    output = []
    for i in range(n):
        # old methode
        start = time.time()
        output.append(old(parameter))
        end = time.time()
        times_old += end - start

        # optimized methode
        start = time.time()
        output.append(opt(parameter))
        end = time.time()
        times_opt += end - start

    old = times_old / n
    print("average time: ", old)
    print("outputs:\n", output)
    opt = times_opt / n
    print("average time: ", opt)
    print("outputs:\n", output)

    # ToDo proper calculation of percenutal increase decrease
    if opt > old:
        quality = "slower"
        percent = (100 / old) * opt
    else:
        quality = "faster"
        percent = (100 / opt) * old
    print(f"optimization is {np.round(percent)}% {quality} than initial procedure")
