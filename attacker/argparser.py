import argparse


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')

    parser.add_argument('--seed', type=int, default=14,
                        help='seed')

    return parser.parse_args()
