import cv2
import numpy as np
from lxml import etree as ET
import base64
import os
from math import sqrt, tan, pi
from cairosvg import svg2png


def load_and_preprocess_image(path, w, h):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    return cv2.resize(im, (w, h))


def generate_STTF_from_image(im, original_h):
    w, h = im.shape[0], im.shape[1]

    # Create our placeholder top left and top right images
    triangles = {"tl": im, "br": im[:]}
    triangles["tl"] = cv2.rotate(triangles["tl"], cv2.ROTATE_180)

    # Equilateral triangle transformation parameters
    SRC_POINTS = [[w, h], [w, 0], [0, 0], [0, h]]
    equilateral_triangle_height = h * sqrt(3) / 2
    DST_POINTS = [
        [w, h],  # Bottom-right corner
        [w + -1 / 2 * w, h - equilateral_triangle_height],  # Top vertex of the equilateral triangle
        [-1 / 2 * w, h - equilateral_triangle_height],  # Left vertex of the equilateral triangle at the offset height
        [0, h]  # Bottom-left corner
    ]

    # we use a matrix transformation via svg to avoid blurring the triangle.
    M = cv2.getPerspectiveTransform(np.float32(SRC_POINTS), np.float32(DST_POINTS))
    M = ' '.join(map(str, [M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2]]))

    # Transformation prefix for each triangle orientation
    CENTER = (w / 2, h - 1 / 3 * original_h)
    # CENTER = (w / 2, 52)
    # STR_TRANS = f"translate({w * -0.25} {h * (1 - 3 ** 0.5 / 2) * 0.5})"
    STR_TRANS = f"translate(0 {-(h - equilateral_triangle_height)})"
    STR_SKEW = f"matrix({M})"
    STR_ROT1 = f"rotate(-120 {CENTER[0]} {CENTER[1]})"
    STR_ROT2 = f"rotate(120 {CENTER[0]} {CENTER[1]})"
    TRANSFORM_PREFIXES = {
        "_a": f'{STR_TRANS} {STR_SKEW}',
        "_b": f'{STR_TRANS} {STR_ROT1} {STR_SKEW}',
        "_c": f'{STR_TRANS} {STR_ROT2} {STR_SKEW}'
    }

    # Generate the transformed triangles dictionary
    transformed_triangles = {
        ori + suffix: (img, prefix)
        for ori, img in triangles.items()
        for suffix, prefix in TRANSFORM_PREFIXES.items()
    }

    return transformed_triangles


def export_tri_svg(im, file_path, transform, bb_size, w, h, S, scale_factor, DEBUG, is_flipped):
    # Adjusting for scale
    bb_size *= scale_factor
    w *= scale_factor
    h *= scale_factor
    offset= 1*scale_factor

    viewbox = (0, 0, bb_size, bb_size)
    root = ET.Element(
        "svg",
        width=f"{bb_size}px",
        height=f"{bb_size}px",
        xmlns="http://www.w3.org/2000/svg",
        stroke="none",
        viewBox="%s %s %s %s" % viewbox,
        nsmap={"xlink": "http://www.w3.org/1999/xlink"},
        version="1.1",
    )
    # base64 image
    im_b64 = base64.b64encode(cv2.imencode('.png', im)[1]).decode()

    # group for clipping
    clipping_group = ET.SubElement(
        root,
        "g",
        {"clip-path": "url(#cut-off-bottom)"}
    )

    # y position at bottom if not flipped else at top
    y_trans = bb_size - S * h - h if not is_flipped else S * h
    # image
    ET.SubElement(
        clipping_group,
        "image",
        {
            ET.QName("http://www.w3.org/1999/xlink", "href"): f"data:image/png;base64,{im_b64}",
            "stroke-width": "1",

        },
        width=f"{im.shape[1]}px",
        height=f"{im.shape[0]}px",
        x="0",
        y="0",
        transform=f'translate({bb_size / 2 - w / 2} {y_trans}) '
                  f'scale({w / im.shape[1]} {w / im.shape[1]}) '
                  + transform,
        fill="#000000",
        preserveAspectRatio="none",

    )

    # transparent bounding circle
    ET.SubElement(
        root,
        "circle",
        {"opacity": "1" if DEBUG else "0"},
        cx="50%",
        cy="50%",
        r="50%",
        fill="none",
        stroke="red",
        strokeWidth="3px",
    )

    # triangle
    x1 = bb_size / 2
    y1 = bb_size - S * h - h
    x2 = bb_size / 2 + h / sqrt(3)
    y2 = bb_size - S * h
    x3 = bb_size / 2 - h / sqrt(3)
    y3 = bb_size - S * h

    clip_triangle = '<polygon fill="none" stroke="red" opacity="0.5" stroke-width="0.25px" ' \
                    f'points="' \
                    f'{x1},{y1 - offset * tan(pi / 3)} ' \
                    f'{x2 + offset * tan(pi / 3)},{y2 + offset} ' \
                    f'{x3 - offset * tan(pi / 3)},{y3 + offset}" />'

    clip_path = ET.fromstring(f'''
        <defs xmlns="http://www.w3.org/2000/svg">
            <clipPath id="cut-off-bottom">
               {clip_triangle}
            </clipPath>
        </defs>
    ''')
    root.append(clip_path)

    if DEBUG:
        # debug triangle
        # triangle = '<polygon fill="none" stroke="red" opacity="0.5" strokeWidth="5px" ' \
        #            f'points="{x1},{y1} {x2},{y2} {x3},{y3}" />'
        triangle = clip_triangle
        root.append(ET.fromstring(triangle))

    tree = ET.ElementTree(root)
    tree.write(file_path + '.svg')


def convert_files(h, S, bb_size, scale_factor, source_paths, OUT, is_debug, is_flipped):
    OUT = OUT + "/" if OUT != "" else ""
    for path in source_paths:
        filename, ext = os.path.splitext(os.path.basename(path))
        if ext != ".png":
            raise ValueError("Image must be in PNG format")

        # Create output and debug directory
        OUT_DIR = f"{OUT}out/{filename}"
        os.makedirs(OUT_DIR, exist_ok=True)

        if is_debug:
            DEBUG_DIR = f"{OUT}out/{filename}/debug"

        im = load_and_preprocess_image(path, int(h * 2 / sqrt(3)), int(h * 2 / sqrt(3)))
        tris = generate_STTF_from_image(im, h)

        for ori, tri in tris.items():
            export_tri_svg(
                im=tri[0],
                file_path=f"{OUT_DIR}/{ori}",
                transform=tri[1],
                bb_size=bb_size,
                w=h * 2 / sqrt(3),
                h=h,
                S=S,
                scale_factor=scale_factor,
                DEBUG=is_debug,
                is_flipped=is_flipped
            )

        if is_debug:
            cv2.imwrite(f"{DEBUG_DIR}/tri br_a.png", tris["br_a"][0])

            # Debug triangle outputs
            for tri_id in ["br_a", "br_b", "br_c"]:
                debug_svg = open(f"{OUT_DIR}/{tri_id}.svg").read()
                svg2png(bytestring=debug_svg, write_to=f"{DEBUG_DIR}/svg {tri_id}.png")
