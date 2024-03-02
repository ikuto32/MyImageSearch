import numpy as np


def format_search_query(search_query_obj) -> str:
    return np.array2string(search_query_obj, separator=', ')


def parse_search_query(search_query_text: str):
    try:
        return np.fromstring(search_query_text.strip('[]'), sep=',', dtype=np.float32).reshape(1, -1)
    except ValueError:
        print("Invalid input format for a numpy array.")
        return None


def main():
    print("Enter search query A (comma-separated):")
    array_A = parse_search_query("")

    print("Enter search query B (comma-separated):")
    array_B = parse_search_query("")

    print("Enter search query C (comma-separated):")
    array_C = parse_search_query(input())

    if array_A is not None and array_B is not None and array_C is not None:
        array_Y = array_A - array_B + array_C
        print("Calculated search query Y is as follows:")
        print(format_search_query(array_Y))
    else:
        print("There was an error in the array input.")


if __name__ == "__main__":
    main()