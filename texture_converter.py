import cv2
import cv2 as cv
import numpy as np
from lxml import etree as ET
import base64
from math import sqrt

# constants
w, h = (100, 100)  # width and height of one cut right angle triangle.
bb_size = 4000
S = h*1.5


def apply_mask_to_alpha(im, mask):
    alpha_mask = np.concatenate([np.ones((*mask.shape, 3), dtype='uint8'), mask.reshape(*mask.shape, 1)], axis=2)
    return im * alpha_mask


def cut_img_into_4_tris(im):
    mask_BL = np.tril(np.ones(im.shape[:2], dtype='uint8'))
    mask_TL = np.rot90(np.tril(np.ones(im.shape[:2][::-1], dtype='uint8')), -1)

    l_tri = apply_mask_to_alpha(im, mask_BL * mask_TL)
    r_tri = apply_mask_to_alpha(im, (1 - mask_BL) * (1 - mask_TL))
    b_tri = apply_mask_to_alpha(im, mask_BL * (1 - mask_TL))
    t_tri = apply_mask_to_alpha(im, (1 - mask_BL) * mask_TL)

    return {"left": l_tri, "right": r_tri, "bottom": b_tri, "top": t_tri}


def align_tri_to_baseline(ori, im):
    # add padding to image to allow for rotation without cutoff edges
    pad_x = int((im.shape[1]/2)*(sqrt(2)-1))
    pad_y = int((im.shape[0]/2)*(sqrt(2)-1))
    im = cv.copyMakeBorder(im,pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, None, value = 0)


    R = {"left": -45, "right": 135, "top": -45, "bottom": 135}
    height, width = im.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=R[ori], scale=1)
    rotated_image = cv2.warpAffine(src=im, M=rotate_matrix, dsize=(width, height))
    cropped_image = rotated_image[:height//2]
    return cropped_image

def export_tri_svg(im, file_path):
    viewbox = (0, 0, bb_size, bb_size)
    root = ET.Element(
        "svg",
        width=f"4000px",
        height=f"4000px",
        xmlns="http://www.w3.org/2000/svg",
        stroke="none",
        viewBox="%s %s %s %s" % viewbox,
        nsmap = {"xlink": "http://www.w3.org/1999/xlink"},
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
        transform=f"translate({bb_size / 2 - w} {bb_size / 2 - h}) scale({w * 2 / im.shape[1]} {h / im.shape[0]})",
        fill="#000000",
        preserveAspectRatio="none"
    )

    # transparent bounding rect
    ET.SubElement(
        root,
        "path",
        {"fill-opacity": "0"},
        fill = "#d40000",
        d= f"M0,{bb_size/2+S}v-{bb_size}h{bb_size}v{bb_size}z",
    )

    tree = ET.ElementTree(root)
    tree.write(file_path + '.svg')


if __name__ == "__main__":
    texture_filename = "test texture"  # must be png
    im = cv.imread(texture_filename + ".png", cv.IMREAD_UNCHANGED)
    im = cv.cvtColor(im, cv.COLOR_RGB2RGBA)
    tris = cut_img_into_4_tris(im)
    aligned_tris = {ori: align_tri_to_baseline(ori, im) for ori, im in tris.items()}
    # cv.imwrite(texture_filename+"(tri).png",aligned_tris[0])
    # cv.waitKey(0)

    for ori, tri in aligned_tris.items():
        export_tri_svg(tri, f"texture tris/tri_{ori}")
