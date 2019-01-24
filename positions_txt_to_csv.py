import sys
import csv
import numpy as np


class Converter:
    @staticmethod
    def convert_positions(test_path):
        with open(test_path + '/result_object_animation.txt',
                  newline='') as csv_input, \
             open(test_path + '/positions.csv', 'w',
                  newline='') as csv_output:

            params_reader = csv.reader(csv_input, delimiter=' ')
            params_writer = csv.writer(csv_output, delimiter=',')
            index = 0

            for row in params_reader:
                indexed_row = np.concatenate(([index], [float(x) for x in row]),
                                             axis=None)
                params_writer.writerow(indexed_row)
                index += 1


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Wrong number of arguments!')
        print('Expected format: python positions_txt_to_csv.py <test_path>')
        exit(1)

    Converter.convert_positions(args[1])
