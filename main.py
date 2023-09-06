from converter import convert_files
import argparse
import os


def init_argparse() -> argparse.ArgumentParser:
    # create a parser object
    parser = argparse.ArgumentParser(
        description="Generates svg files from textures for use in the scratch project: 'Textured tri fill'"
    )

    # add arguments
    parser.add_argument("-H", "--height", nargs=1, default=64, type=int,
                        help="Height in pixels of triangle")

    parser.add_argument("-b", "--bbsize", nargs=1, default=2048, type=int,
                        help="Size of bounding box in pixels")

    parser.add_argument("-S", "--displacement", nargs=1,
                        default=4, type=int,
                        help="Displacement of triangle in multiples of it's height from bounding box")

    parser.add_argument("-SF", "--scalefactor", nargs=1,
                        default=0.25, type=int,
                        help="Scale factor to apply to costume")

    parser.add_argument("-f", "--flipped", nargs=1,
                        default=False, type=bool,
                        help="Scale factor to apply to costume")


    parser.add_argument("-s", "--source", nargs='*',
                        default=[
                            "test textures/chrome_cat.png",
                            "test textures/test texture.png"
                        ], type=str, help="Input textures separated by spaces")

    parser.add_argument("-o", "--out", nargs=1, default="", type=str,
                        help="Output directory")

    parser.add_argument("-D", "--debug", nargs=1, default=False, type=bool,
                        help="Set to 1, to export a debug image")
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    convert_files(
        args.height,
        args.displacement,
        args.bbsize,
        args.scalefactor,
        args.source,
        args.out,
        args.debug,
        args.flipped
    )
