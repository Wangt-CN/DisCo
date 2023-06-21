import numpy as np
import base64
import cv2
import os
from .common import ensure_directory
from .common  import network_input_to_image
from .common  import encoded_from_img
import matplotlib.pyplot as plt
from random import random
import logging
from .common  import is_pil_image
from PIL import Image


def cvimg_to_pil(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return Image.fromarray(im)

def copy_make_border(im, top, bottom, left, right):
    if is_pil_image(im):
        w, h = im.size
        w2 = w + left + right
        h2 = top + bottom + h
        im2 = Image.new('RGB', (w2, h2))
        im2.paste(im, (left, top, left + w, top + h))
        return im2
    else:
        im_squared = cv2.copyMakeBorder(im, top=top, bottom=bottom, left=left, right=right,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return im_squared

def im_rescale(im, target_size):
    if is_pil_image(im):
        w, h = im.size
        if w > h:
            if w == target_size:
                return im, 1
            w2 = target_size
            h2 = (w2 * h + w - 1) // w
            im_scale = 1. * w2 / w
        else:
            if h == target_size:
                return im, 1
            h2 = target_size
            w2 = (h2 * w + h - 1) // h
            im_scale = 1. * h2 / h
        #im_resized = im.resize((w2, h2), PIL.Image.BILINEAR)
        im_resized = im.resize((w2, h2))
        return im_resized, im_scale
    else:
        im_size_max = max(im.shape[0:2])
        if target_size == im_size_max:
            return im, 1
        im_scale = float(target_size) / float(im_size_max)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        return im_resized, im_scale

def gen_colors(num_real_classes):
    colors = []
    for c in range(num_real_classes):
        colors.append(np.random.rand(3))
    return colors

def draw_rects(rects, im=None, add_label=True, style=None):
    if im is None:
        im = np.zeros((1000, 1000, 3), dtype=np.uint8)
    probs = None
    if all('conf' in r for r in rects):
        probs = [r['conf'] for r in rects]
    draw_bb(im,
            [r['rect'] for r in rects],
            [r['class'] for r in rects],
            probs=probs,
            draw_label=add_label,
            style=style)
    return im

def put_text(im, text, bottomleft=(0,100),
        color=(255,255,255), font_scale=0.5,
        font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if hasattr(cv2, 'UMat'):
        im2 = cv2.putText(cv2.UMat(im),text,bottomleft,
                font,font_scale, color,
                thickness=font_thickness)
        im[:] = im2.get()
    else:
        cv2.putText(im,text,bottomleft,
                font,font_scale, color,
                thickness=font_thickness)
    return cv2.getTextSize(text, font, font_scale, font_thickness)[0]

def show_net_input_image(data, mean_value=[104, 117, 123], std_value=[1, 1, 1],
        save_to_file=None):
    all_image = network_input_to_image(data, mean_value, std_value)
    for i in range(len(all_image)):
        if save_to_file:
            save_image(all_image[i], save_to_file+'{}.jpg'.format(i))
        else:
            show_image(all_image[i])

def show_net_input(data, label, max_image_to_show=None,
                mean_value=[104, 117, 123], std_value=[1, 1, 1],
                save_to_file=None, draw_label=True):
    all_image = network_input_to_image(data, mean_value, std_value)
    num_image = data.shape[0]
    num_rect = label.shape[1] // 5
    if max_image_to_show:
        num_image = min(max_image_to_show, num_image)
    im_height = all_image[0].shape[0]
    im_width = all_image[0].shape[1]
    for i in range(num_image):
        rects = []
        txts = []
        for j in range(num_rect):
            if label[i, j * 5] == 0:
                break
            cx, cy, w, h = label[i, (j * 5 + 0) : (j * 5 + 4)]
            txt = str(label[i, j * 5 + 4])
            cx = cx * (im_width - 1)
            cy = cy * (im_height - 1)
            w = w * (im_width - 1)
            h = h * (im_height - 1)
            rects.append((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 *
                h))
            txts.append(txt)
        draw_bb(all_image[i], rects, txts, draw_label=draw_label)
        if save_to_file:
            save_image(all_image[i], save_to_file+'{}.jpg'.format(i))
        else:
            show_image(all_image[i])

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=None):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    if gap is None:
        gap = thickness * 3
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def draw_dotted_rect(img,pt1,pt2,color,thickness=1):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style='dotted')

__label_to_color = {}
__gold_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
        (0, 255, 255),
        ]

def rectangle(img, *args, **kwargs):
    if hasattr(cv2, 'UMat'):
        im2 = cv2.rectangle(cv2.UMat(img), *args, **kwargs)
        img[:] = im2.get()
    else:
        cv2.rectangle(img, *args, **kwargs)

def draw_bb(im, all_rect, all_label,
        probs=None,
        color=None,
        font_scale=None,
        font_thickness=None,
        #rect_thickness=2,
        draw_label=True,
        style=None):
    '''
    all_rect: x0, y0, x1, y1
    '''
    ref = sum(im.shape[:2]) // 2

    if font_scale is None:
        font_scale = ref / 500.
    if font_thickness is None:
        font_thickness = max(ref // 300, 1)
    rect_thickness = max(ref // 250, 1)
    # in python3, it is float, and we need to convert it to integer
    font_thickness = int(font_thickness)
    rect_thickness = int(rect_thickness)

    dist_label = set(all_label)
    if color is None:
        color = {}
        color = __label_to_color
        for l in dist_label:
            if l in color:
                continue
            if len(__gold_colors) > 0:
                color[l] = __gold_colors.pop()
    for i, l in enumerate(dist_label):
        if l in color:
            continue
        color[l] = (random() * 255., random() * 255, random() * 255)

    if type(all_rect) is list:
        assert len(all_rect) == len(all_label)
    elif type(all_rect) is np.ndarray:
        assert all_rect.shape[0] == len(all_label)
        assert all_rect.shape[1] == 4
    else:
        assert False

    all_filled_region = []
    all_put_text = []
    placed_position = {}
    for i in range(len(all_label)):
        rect = all_rect[i]
        label = all_label[i]
        if style == 'dotted':
            draw_dotted_rect(im, (int(rect[0]), int(rect[1])),
                    (int(rect[2]), int(rect[3])),
                    color[label],
                    thickness=rect_thickness)
        else:
            assert style is None
            rectangle(im, (int(rect[0]), int(rect[1])),
                    (int(rect[2]), int(rect[3])), color[label],
                    thickness=rect_thickness)
        if probs is not None:
            if draw_label:
                label_in_image = '{}-{:.2f}'.format(label, probs[i])
            else:
                label_in_image = '{:.2f}'.format(probs[i])
        else:
            if draw_label:
                label_in_image = '{}'.format(label)

        if draw_label or probs is not None:
            (text_width, text_height), _ = cv2.getTextSize(label_in_image, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_thickness)
            text_left = int(rect[0] + 2)
            left_top = (int(rect[0]), int(rect[1]))
            if left_top in placed_position:
                text_bottom = placed_position[left_top][-1] + text_height
                placed_position[left_top].append(text_bottom)
            else:
                text_bottom = int(rect[1]) + text_height
                placed_position[left_top] = [text_bottom]

            all_filled_region.append(((text_left, text_bottom - text_height),
                    (text_left + text_width, text_bottom + 5), (75, 75, 75)))
            all_put_text.append((label_in_image, (text_left, text_bottom),
                color[label]))

    for left_top, right_bottom, c in all_filled_region:
        rectangle(im, left_top, right_bottom,
                c,
                thickness=-1)
    for label_in_image, (text_left, text_bottom), c in all_put_text:
        put_text(im,
                label_in_image,
                (text_left, text_bottom),
                c,
                font_scale,
                font_thickness)

def save_image(im, file_name, quality=None):
    ensure_directory(os.path.dirname(file_name))
    if quality is None:
        return cv2.imwrite(file_name, im)
    else:
        return cv2.imwrite(file_name, im, [int(cv2.IMWRITE_JPEG_QUALITY),
            quality])


def load_image(file_name):
    return cv2.imread(file_name)

def load_image_by_pil(file_name, respect_exif=False):
    image = Image.open(file_name).convert('RGB')
    if respect_exif:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    return image

def pil_to_cvim(pil_image):
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1]
    return open_cv_image

def show_image(im):
    show_images([im], 1, 1)

def show_images(all_image, num_rows=None, num_cols=None,
        titles=None, out_fname=None):
    plt.figure(1)
    if num_rows is None and num_cols is None:
        num_rows = 1
        num_cols = len(all_image)

    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if k >= len(all_image):
                break
            plt.subplot(num_rows, num_cols, k + 1)
            if is_pil_image(all_image[k]):
                plt.imshow(np.asarray(all_image[k]))
            else:
                if len(all_image[k].shape) == 3:
                    plt.imshow(cv2.cvtColor(all_image[k],
                        cv2.COLOR_BGR2RGB))
                else:
                    # grey image
                    assert len(all_image[k].shape) == 2
                    plt.imshow(np.repeat(all_image[k][:, :, np.newaxis], 3, axis=2))
            if titles is not None:
                plt.title(titles[k])
            k = k + 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if out_fname is None:
        plt.show()
    else:
        plt.savefig(out_fname)
    plt.close()

def bytes_to_img_array(img_bytes, check_channel=True):
    """ Convert bytes to image array of shape h*w*c in BGR order
    Ensure the image is valid with 3 channels if check_channel == True
    NOTE: use > py3.5 to load webp image with OpenCV
    """
    import imghdr
    import imageio
    import sys

    is_py2 = sys.version_info.major == 2
    if is_py2:
        from StringIO import StringIO as BytesIO
    else:
        from io import BytesIO

    t = imghdr.what('', h = img_bytes)
    imarr = None
    if t == "gif":
        gif = imageio.mimread(BytesIO(img_bytes))
        imarr = gif[0]
    else: # ["jpg", "jpeg", "png", "webp"]
        try:
            imarr = imageio.imread(BytesIO(img_bytes))
        except (ValueError, SyntaxError) as e:
            return None

    if imarr is None:
        return None
    # ensure dtype is uint8
    if imarr.dtype is not np.dtype('uint8'):
        # NOTE: no easy way to convert other dtype to uint8 color scale
        # info = np.iinfo(imarr.dtype) # Get the information of the incoming image type
        # imarr = imarr.astype(np.float64) / info.max # normalize the imarr to 0 - 1
        # imarr = 255 * imarr # Now scale by 255
        # imarr = imarr.astype(np.uint8)
        return None

    # conver grayscale
    if len(imarr.shape) == 2:
        imarr = cv2.cvtColor(imarr, cv2.COLOR_GRAY2RGB)
    if len(imarr.shape) != 3:
        return None

    h, w, c = imarr.shape
    # convert form RGBA to BGRA
    if c == 3:
        imarr = imarr[:, :, (2, 1, 0)]
    elif c == 4:
        imarr = imarr[:, :, (2, 1, 0, 3)]
    else:
        return None

    if check_channel and c == 4:
        imarr = bgra_to_bgr_img_arr(imarr)

    if imarr.max() - imarr.min() < 5:
        return None

    return imarr

def file_to_base64_img(fpath, check_channel=True):
    """ Read image file, converts to base64 encoded string
    """
    with open(fpath, 'rb') as fp:
        img_bytes = fp.read()
    imarr = bytes_to_img_array(img_bytes, check_channel=check_channel)
    return encoded_from_img(imarr)

def bgra_to_bgr_img_arr(img_arr):
    """ Convert BGRA to BGR, and transparent part to white
    if using opencv built-in cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB),
    transparent part can be any color
    """
    h, w, c = img_arr.shape
    assert(c == 4)
    alpha_channel = img_arr[:, :, 3]
    trans_thres = max(alpha_channel.max() // 2, 1)
    _, mask = cv2.threshold(alpha_channel, trans_thres, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img_arr[:, :, :3]
    new_img_arr = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    return new_img_arr
