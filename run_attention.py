import PIL
import numpy
from matplotlib import patches

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def non_max_suppression(dim, result, c_image):
    x = []
    y = []
    image_height, image_width = result.shape[:2]
    for i in range(0, image_height - dim, dim):
        for j in range(0, image_width - dim, dim):
            index = np.argmax(result[i:i + dim, j:j + dim])
            a = np.amax(result[i:i + dim, j:j + dim])
            if a > 100:
                x.append(index // dim + i)
                y.append(index % dim + j)
                c_image[index // dim + i, index % dim + j] = 0
    return x, y


def red_picture(im_red, filter_kernel, c_image):
    grad = sg.convolve2d(im_red, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(12, 30))
    ax_orig.imshow(im_red)
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_mag.imshow(np.absolute(grad))
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()
    ax_ang.imshow(np.absolute(result))
    ax_ang.set_title('Gradient orientation')
    ax_ang.set_axis_off()
    fig.show()
    x_red, y_red = non_max_suppression(14, result, c_image)
    return x_red, y_red


def green_picture(im_green, filter_kernel, c_image):
    grad = sg.convolve2d(im_green, filter_kernel, boundary='symm', mode='same')
    result = ndimage.maximum_filter(grad, size=5)
    fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(12, 30))
    ax_orig.imshow(im_green)
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_mag.imshow(np.absolute(grad))
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()
    ax_ang.imshow(np.absolute(result))
    ax_ang.set_title('Gradient orientation')
    ax_ang.set_axis_off()
    fig.show()
    x_green, y_green = non_max_suppression(18, result, c_image)
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.set_axis_off()
    max_mag.imshow(np.absolute(c_image))
    fig.show()
    return x_green, y_green


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]
    filter_kernel = [[-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                        -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                        -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324,
                        11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                        -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                        -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324],
                       [-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                        -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324]]
    x_red, y_red = red_picture(im_red, filter_kernel, c_image)
    x_green, y_green = green_picture(im_green, filter_kernel, c_image)
    print(x_red)
    print(y_red)
    print(x_green)
    print(y_green)
    # x = np.arange(-100, 100, 20) + c_image.shape[1] / 2
    # y_red = [c_image.shape[0] / 2 - 120] * len(x)
    # y_green = [c_image.shape[0] / 2 - 100] * len(x)
    # return x, y_red, x, y_green
    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)
    find_tfl_lights(image, some_threshold=42)
    # red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'pic'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
