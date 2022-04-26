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
    # if the length of arrays are different return false
    n_array_1 = len(array_1)
    n_array_2 = len(array_2)
    if n_array_1 != n_array_2:
        error_print(array_is_equal.__name__, f"Array not same lengths: {n_array_1} ≠ {n_array_2}")
        return False
    else:
        # sort both the arrays
        array_1.sort()
        array_2.sort()
        # traverse each index of arrays
        for i in range(n_array_1):
            # for same index if the value in the sorted arrays are different return false
            if array_1[i] != array_2[i]:
                error_print(array_is_equal.__name__,
                                 f"Item at i={i} not equal: [{array_1[i]}] ≠ [{array_2[i]}]")
                return False

    # if none of the above conditions satisfied return true
    return True
