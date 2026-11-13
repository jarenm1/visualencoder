import argparse
from pathlib import Path

from . import distill, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="visenc")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="render a dataset split from the city scene")
    data.add_argument("--scene", type=Path, default=Path("assets/city"), help="output of scene/build_city.py")
    data.add_argument("--out", type=Path, required=True, help="new split directory")
    data.add_argument("--seed", type=int, required=True, help="pose/appearance seed; use distinct seeds per split")
    data.add_argument("--count", type=int, required=True)
    data.add_argument("--width", type=int, default=320)
    data.add_argument("--height", type=int, default=240)
    data.add_argument("--random-views", type=int, default=2, help="random appearances per pose (plus canonical)")
    data.add_argument("--workers", type=int, default=8, help="parallel renderer processes")
    data.add_argument("--exclude-region", type=float, nargs=4, metavar=("X0", "Z0", "X1", "Z1"),
                      help="world rectangle no pose may fall in (spatial holdout)")  # fmt: skip

    distill.add_args(sub.add_parser("distill", help="distil a DINOv3 teacher into a ViT student"))
    evaluate.add_args(sub.add_parser("eval", help="probe frozen encoders against simulator ground truth"))

    args = parser.parse_args()
    if args.command == "data":
        from .data import generate

        generate(
            args.scene, args.out, args.seed, args.count, args.width, args.height,
            args.random_views, args.workers, tuple(args.exclude_region) if args.exclude_region else None,
        )  # fmt: skip
    elif args.command == "distill":
        distill.train(args)
    else:
        evaluate.run(args)


if __name__ == "__main__":
    main()
