import argparse

parser = argparse.ArgumentParser(description="Parse bool")
parser.add_argument("--do", default=False, type=bool,
                                        help="Flag to do something")
args = parser.parse_args()
print(args)
