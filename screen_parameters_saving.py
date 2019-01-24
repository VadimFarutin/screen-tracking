import sys
import csv


class ScreenParametersSaver:
    @staticmethod
    def save_screen_parameters(test_path, width, height):
        with open(test_path + '/screen_parameters.csv', 'w', newline='') as csv_file:
            params_writer = csv.writer(csv_file, delimiter=',')
            params_writer.writerow([width] + [height])


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        print('Wrong number of arguments!')
        print('Expected format: python screen_parameters_saving.py <test_path> <width> <height>')
        exit(1)

    ScreenParametersSaver.save_screen_parameters(args[1], args[2], args[3])
