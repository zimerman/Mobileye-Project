from Light_sources import *
from Network import *
from Frame_container import FrameContainer
import tensorflow.keras
from set_data import *
from SFM import *


class TFL:

    def __init__(self, pkl_path):

        with open(pkl_path, 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')
        self.focal = self.data['flx']
        self.pp = self.data['principle_point']
        self.prev_frame = None

    def run_product(self, image_path, frame_index):

        current_frame = FrameContainer(image_path)

        '''part1'''
        candidate, auxiliary = self.coord_of_source_lights(image_path)

        '''part2'''
        traffic, auxiliary_traffic = self.get_tfl_coord(candidate, auxiliary, image_path)
        current_frame.traffic_light = traffic

        '''part3'''
        dists = []
        if self.prev_frame:
            current_frame.EM = np.dot(self.data['egomotion_' + str(frame_index-1) + '-' + str(frame_index)], np.eye(4))
            if traffic != [] and self.prev_frame.traffic_light != []:
                dists = self.dist(current_frame)

        self.visual_images(image_path, current_frame, auxiliary_traffic, candidate, auxiliary, dists)
        self.prev_frame = current_frame

    @staticmethod
    def coord_of_source_lights(image_path):

        x_red, y_red, x_green, y_green = find_tfl_lights(image_path, some_threshold=42)
        x_coord, y_coord = x_red + x_green, y_red + y_green
        candidate = np.array([[x_coord[i], y_coord[i]] for i in range(len(x_coord))])
        auxiliary = ['r' if i < len(x_red) else 'g' for i in range(len(x_coord))]

        return candidate, auxiliary

    @staticmethod
    def get_tfl_coord(candidate, auxiliary, image_path):

        traffic = []
        auxiliary_traffic = []
        model = load_model("model.h5")
        image = np.array(Image.open(image_path))
        image_pad = np.pad(image, 40, pad_with_zeros)[:, :, 40:43]

        for i, coord in enumerate(candidate):
            crop_images = get_crop_images(image_pad, coord)
            result = network(crop_images, model)
            if result:
                traffic.append(coord)
                auxiliary_traffic.append(auxiliary[i])

        return traffic, auxiliary_traffic

    def dist(self, current_frame):

        curr_container = calc_TFL_dist(self.prev_frame, current_frame, self.focal, self.pp)

        return curr_container.traffic_lights_3d_location[:, 2]

    def visual_images(self, image_path, current_frame, auxiliary_traffic, candidate, auxiliary, dist):

        fig, (ax_source_lights, ax_is_tfl, ax_dist) = plt.subplots(3, 1, figsize=(12, 30))

        ax_source_lights.imshow(current_frame.img)
        x_, y_ = candidate[:, 1], candidate[:, 0]
        ax_source_lights.scatter(x_, y_, c=auxiliary, s=1)
        ax_source_lights.set_title('source_light')

        ax_is_tfl.imshow(current_frame.img)
        x_, y_ = np.array(current_frame.traffic_light)[:, 1], np.array(current_frame.traffic_light)[:, 0]
        ax_is_tfl.scatter(x_, y_, c=auxiliary_traffic, s=1)
        ax_is_tfl.set_title('traffic_light')

        if dist != list():
            x_cord, y_cord, image_dist = self.get_cord_tfl(current_frame, image_path)
            ax_dist.imshow(image_dist)
            for i in range(len(x_cord)):
                ax_dist.text(y_cord[i], x_cord[i], r'{0:.1f}'.format(dist[i]), color='r')
        ax_dist.set_title('dist of tfl')

        fig.show()

    @staticmethod
    def get_cord_tfl(current_frame, image_path):

        image = np.array(Image.open(image_path))
        curr_p = current_frame.traffic_light
        x_cord = [p[0] for p in curr_p]
        y_cord = [p[1] for p in curr_p]

        return x_cord, y_cord, image





















