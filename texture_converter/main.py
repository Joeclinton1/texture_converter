from converter import convert_to_sub_textures
import argparse


def init_argparse() -> argparse.ArgumentParser:
    # create a parser object
    parser = argparse.ArgumentParser(
        description="Generates svg files from textures for use in the scratch project: 'Textured tri fill'"
    )

    # add arguments
    parser.add_argument("-d", "--dim", nargs=2, default=[64, 64], type=int,
                        help="Width and height in pixels of output image. def")

    parser.add_argument("-b", "--bbsize", nargs=1, default=2048, type=int,
                        help="Size of bounding box in pixels")

    parser.add_argument("-s", "--source", nargs='*', default=["test texture"], type=str,
                        help="Input textures separated by spaces")

    parser.add_argument("-o", "--out", nargs=1, default="sub textures", type=str,
                        help="Output directory")

    parser.add_argument("-m", "--mode", nargs=1, default="SPLIT", type=str,
                        help="Mode to use for converting texture to subtextures. Can be 'SPLIT' or 'OVERLAP'")

    parser.add_argument("-D", "--debug", nargs=1, default=0, type=int,
                        help="Set to 1, to export a debug image")
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    convert_to_sub_textures(*args.dim, args.bbsize, args.source, args.out, args.mode, args.debug)
