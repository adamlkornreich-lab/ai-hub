import argparse

from . import PROVIDERS, ask, ask_all

parser = argparse.ArgumentParser(prog="ai_hub", description="Ask multiple AIs at once")
parser.add_argument("prompt")
parser.add_argument("-p", "--provider", choices=list(PROVIDERS), help="default: all configured")
args = parser.parse_args()

if args.provider:
    print(ask(args.provider, args.prompt))
else:
    for name, reply in ask_all(args.prompt).items():
        print(f"=== {name} ===\n{reply}\n")
