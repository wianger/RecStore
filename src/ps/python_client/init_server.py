from client import ParameterClient as Client
import argparse


"""
key_size 12543669
"""

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_size", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15000)
    args = parser.parse_args()
    return args

def main():
    args = parse()
    client = Client(args.host, args.port, 0, 0)
    client.LoadFakeData(args.key_size)
    print("init done")

if __name__ == "__main__":
    main()
