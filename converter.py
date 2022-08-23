import cv2
import cv2 as cv
import numpy as np
from lxml import etree as ET
import base64
import os


def apply_mask_to_alpha(im, mask):
    alpha_mask = np.concatenate([np.ones((*mask.shape, 3), dtype='uint8'), mask.reshape(*mask.shape, 1)], axis=2)
    return im * alpha_mask


def cut_img_into_2_tris(im, masks):
    masks = [cv.resize(mask, im.shape[:2])[:, :, 0] / 255 for mask in masks]
    return {
        "tl": apply_mask_to_alpha(im, masks[0]),
        "br": apply_mask_to_alpha(im, masks[1])
    }


def apply_gradient(im, dir):
    gradient_mask = np.tile(np.linspace((dir + 1) / 2, (dir - 1) / -2, im.shape[1]), (im.shape[0], 1))
    return apply_mask_to_alpha(im, gradient_mask)


def transform(im, s, r):
    h, w = im.shape[:2]
    if r:
        rotate_matrix = cv2.getRotationMatrix2D(center=(h * r[1][0], w * r[1][1]), angle=r[0], scale=1)
        im = cv2.warpAffine(src=im, M=rotate_matrix, dsize=(w, h))

    if s:
        src = [[w, h], [w, 0], [0, 0], [0, h]]
        dst = [[w, h], [w + s[0] * w, s[1] * h], [s[0] * w, s[1] * h], [0, h]]
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        im = cv2.warpPerspective(im, M, (w, h))

    return im


def export_tri_svg(im, file_path, is_sheared, bb_size, w, h, S):
    viewbox = (0, 0, bb_size, bb_size)
    root = ET.Element(
        "svg",
        width=f"{bb_size}px",
        height=f"{bb_size}px",
        xmlns="http://www.w3.org/2000/svg",
        stroke="none",
        viewBox="%s %s %s %s" % viewbox,
        nsmap={"xlink": "http://www.w3.org/1999/xlink"},
        version="1.1"
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
        transform=f"translate({bb_size / 2 - w} {bb_size - S - h})"
                  f"scale({w / im.shape[1]} {h / im.shape[0]})"
                  f"skewX({is_sheared * 45})",
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

def convert_to_sub_textures(w, h, bb_size, path, OUT, MODE, DEBUG):
    S = h * 1.5
    OUT = OUT + "/" if OUT else ""
    for path in path:
        filename, ext = os.path.basename(path).split(".")
        if ext != "png":
            raise "Image must be in PNG format"

        # Create output directory if it doesn't exist
        OUT_DIR = f"{OUT}out/{filename}/{MODE}"
        if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR)

        # Load texture
        im = cv.imread(path, cv.IMREAD_UNCHANGED)
        im = cv.cvtColor(im, cv.COLOR_RGB2RGBA)
        im = cv.resize(im, (w * 2, h * 2))

        # split triangle if SPLIT else apply gradient
        new_ims = {}
        if MODE == "SPLIT":
            # load alpha masks
            tl_mask = cv.imread("alpha masks/tl_alpha_mask.png", cv.IMREAD_UNCHANGED)
            br_mask = cv.imread("alpha masks/br_alpha_mask.png", cv.IMREAD_UNCHANGED)

            # Cut texture along diagonal into two right angled tris | □ ⟶ ◸◿
            new_ims = cut_img_into_2_tris(im, (tl_mask, br_mask))
            # Create rotated and sheared copies
            new_ims["tl"] = transform(new_ims["tl"], None, (180, (0.5, 0.5)))
        else:
            # Create quads with gradient in middle
            new_ims = {"l": apply_gradient(im, 1), "r": apply_gradient(im, -1)}

        new_ims = new_ims | {ori + "_rot90": transform(im, (1, 0), (-90, (0.5, 0.5))) for ori, im in new_ims.items()}
        new_ims = new_ims | {ori + "_shear": im for ori, im in new_ims.items()}

        # Export tri svg files
        if DEBUG:
            cv.imwrite("test texture(tri).png", new_ims["tl"])
            cv.waitKey(0)

        for ori, tri in new_ims.items():
            export_tri_svg(tri, f"{OUT_DIR}/{ori}", "shear" in ori, bb_size, w, h, S)