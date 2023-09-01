import cv2
import cv2 as cv
import numpy as np
from lxml import etree as ET
import base64
import os
from math import atan, degrees


def apply_mask_to_alpha(im, mask):
    alpha_mask = np.concatenate([np.ones((*mask.shape, 3), dtype='uint8'), mask.reshape(*mask.shape, 1)], axis=2)
    return im * alpha_mask


def cut_img_into_2_tris(im, masks):
    masks = [cv.resize(mask, im.shape[:2]) / 255 for mask in masks]
    return {
        "tl": apply_mask_to_alpha(im, masks[0]),
        "br": apply_mask_to_alpha(im, masks[1])
    }


def pack_ims(ims):
    return {ori: [im, ""] for ori, im in ims.items()}


def transform(im, s, r):
    h, w = im.shape[:2]
    if r:
        rotate_matrix = cv2.getRotationMatrix2D(center=(w * r[1][0], h * r[1][1]), angle=r[0], scale=1)
        im = cv2.warpAffine(src=im, M=rotate_matrix, dsize=(w, h))

    if s:
        src = [[w, h], [w, 0], [0, 0], [0, h]]
        dst = [[w, h], [w + s[0] * w, s[1] * h], [s[0] * w, s[1] * h], [0, h]]
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        im = cv2.warpPerspective(im, M, (w, h))

    return im


def export_tri_svg(im, file_path, transform, bb_size, w, h, S):
    use_local_origin = False
    trans = f'translate({- w / 2} {S - bb_size / 2})' if use_local_origin else f''

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
        transform=trans
    )
    # base64 image
    im_b64 = base64.b64encode(cv.imencode('.png', im)[1]).decode()

    # image
    ET.SubElement(
        root,
        "image",
        {
            ET.QName("http://www.w3.org/1999/xlink", "href"): f"data:image/png;base64,{im_b64}",
            "stroke-width": "1"
        },
        width=f"{im.shape[1]}px",
        height=f"{im.shape[0]}px",
        x="0",
        y="0",
        transform=f'translate({bb_size / 2 - w / 2} {bb_size - S - h}) '
                  f'scale({w / im.shape[1]} {h / im.shape[0]}) '
                  + transform,
        fill="#000000",
        preserveAspectRatio="none"
    )

    # transparent bounding rect
    ET.SubElement(
        root,
        "path",
        {"fill-opacity": "0"},
        fill="#d40000",
        d=f"M0,{bb_size}v-{bb_size}h{bb_size}v{bb_size}z",
    )

    tree = ET.ElementTree(root)
    tree.write(file_path + '.svg')


def convert_to_sub_textures(w, h, bb_size, path, OUT, DEBUG):
    S = h * 1.5
    OUT = OUT + "/" if OUT else ""
    for path in path:
        filename, ext = os.path.basename(path).split(".")
        if ext != "png":
            raise "Image must be in PNG format"

        # Create output directory if it doesn't exist
        OUT_DIR = f"{OUT}out/{filename}"
        if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR)

        # Load texture
        im = cv.imread(path, cv.IMREAD_UNCHANGED)
        im = cv.cvtColor(im, cv.COLOR_RGB2RGBA)
        im = cv.resize(im, (w * 2, h * 2))

        br_mask = np.rot90(np.tri(*im.shape[:2], k=1, dtype=int)) * 255
        tl_mask = np.rot90(br_mask, k=2)

        # Cut texture along diagonal into two right angled tris | □ ⟶ ◸◿
        ims = pack_ims(cut_img_into_2_tris(im, (tl_mask, br_mask)))

        # rotate top left so that
        ims["tl"][0] = cv2.rotate(ims["tl"][0], cv2.ROTATE_180)

        # transform the right angled triangles into equilateral ones

        src = [[w, h], [w, 0], [0, 0], [0, h]]
        dst = [[w, h], [w + -1 / 2 * w, (1 - 3 ** (1 / 2) / 2) * h], [-1 / 2 * w, (1 - 3 ** (1 / 2) / 2) * h],
               [0, h]]
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        M2 = ' '.join(map(str, [M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2]]))

        trans = f'matrix({M2}) '
        ims = {ori: (im[0], im[1] + trans) for ori, im in ims.items()}

        # create two additional triangles. One rotated 60° the other 120°
        # center = [w*2*0.75, h*2*(3 ** (1 / 2) / 2)*(1 - 3 ** (1 / 2) / 6)]
        center = [w * 2 * 0.75, 82.5]
        trans = f'translate({w * 2 * -1 / 2 / 2} {h * 2 * (1 - 3 ** (1 / 2) / 2) / 2})'
        a = {ori + "_a": (im[0], trans + im[1]) for ori, im in ims.items()}
        b = {ori + "_b": (im[0], trans + f'rotate(-120 {center[0]} {center[1]})' + im[1]) for ori, im in
             ims.items()}
        c = {ori + "_c": (im[0], trans + f'rotate(120 {center[0]} {center[1]})' + im[1]) for ori, im in
             ims.items()}
        ims = a | b | c

        # Export tri svg files
        if DEBUG:
            DEBUG_DIR = f"{OUT}out/{filename}/debug"
            if not os.path.isdir(DEBUG_DIR):
                os.makedirs(DEBUG_DIR)

            cv.imwrite(f"{DEBUG_DIR}/mask_br.png", br_mask)
            cv.imwrite(f"{DEBUG_DIR}/mask_tl.png", tl_mask)
            cv.imwrite(f"{DEBUG_DIR}/tri br a.png", ims["br_a"][0])
            cv.imwrite(f"{DEBUG_DIR}/tri tl a.png", ims["tl_a"][0])
            print([im[1] for im in [ims["br_a"]]])

        for ori, pack_im in ims.items():
            export_tri_svg(pack_im[0], f"{OUT_DIR}/{filename}_{ori}", pack_im[1], bb_size, w, h, S)
