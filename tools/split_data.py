import argparse
import random


def sample_file(input_file, output_file, percentage, seed):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Calculate the number of lines to sample
    num_lines = int(len(lines) * (percentage / 100))

    # Randomly sample the lines
    sampled_lines = random.sample(lines, num_lines)

    # Write the sampled lines to the output file
    with open(output_file, 'w') as f:
        f.writelines(sampled_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly sample lines from a file.')
    parser.add_argument('--input_file',
                        required=True,
                        help='Path to the input text file')
    parser.add_argument('--output_file',
                        required=True,
                        help='Path to the output text file')
    parser.add_argument('--percentage',
                        type=float,
                        required=True,
                        help='Percentage of lines to sample')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    sample_file(args.input_file, args.output_file, args.percentage, args.seed)
