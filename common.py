from csv import reader


def read(filename):
    with open(filename, "r") as file:
        return [list(map(float, line)) for line in reader(file)]
