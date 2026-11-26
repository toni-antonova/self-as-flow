import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import portability
import sensing_results
import tracking_results


def main():
    parser = argparse.ArgumentParser(description="Calculate the results reported in Self as flow.")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Directory of processed inputs.")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="Directory for result CSVs.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tracking_results.write_results(args.data, args.output)
    portability.run(args.data, args.output)
    sensing_results.reproduce(args.data, args.output)


if __name__ == "__main__":
    main()
