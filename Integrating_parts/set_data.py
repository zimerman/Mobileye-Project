

def get_crop_images(image, coord):

    crop_image = image[coord[0]:coord[0] + 81, coord[1]:coord[1] + 81]
    return crop_image


def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0