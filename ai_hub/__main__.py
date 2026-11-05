import argparse
import sys

from . import PROVIDERS, ask, ask_all, configured, default_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_hub", description="Ask multiple AIs at once")
    parser.add_argument("prompt", nargs="?", help="prompt text (default: read from stdin)")
    parser.add_argument("-p", "--provider", choices=list(PROVIDERS),
                        help="default: all configured")
    parser.add_argument("-m", "--model", help="model override (requires --provider)")
    parser.add_argument("-s", "--system", help="system prompt")
    parser.add_argument("--list", action="store_true",
                        help="show providers, default models and which are configured")
    args = parser.parse_args(argv)

    if args.list:
        have = set(configured())
        for p in PROVIDERS:
            print(f"{p:10} {default_model(p):24} {'configured' if p in have else '-'}")
        return 0
    if args.model and not args.provider:
        parser.error("--model requires --provider")

    prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt:
        parser.error("no prompt given")

    if args.provider:
        try:
            print(ask(args.provider, prompt, args.model, args.system))
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    replies = ask_all(prompt, args.system)
    if not replies:
        envs = ", ".join(env for env, _ in PROVIDERS.values())
        print(f"No providers configured; set one of {envs}", file=sys.stderr)
        return 1
    for name, reply in replies.items():
        print(f"=== {name} ===\n{reply}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
