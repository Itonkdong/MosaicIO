import math

import cv2
import os
import pickle

import numpy as np


class ImageProcessor:
    def show_image(self, image, win_name="Default"):
        cv2.imshow(win_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read_image_names(self, folder_path):
        image_names = os.listdir(folder_path)
        image_names = [f"{folder_path}/{name}" for name in image_names]

        return image_names

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.height, width[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    def average_color_slow(self, image):
        image_copy = image.astype(np.float64)
        blue_sum = 0
        green_sum = 0
        red_sum = 0
        pixel_count = 0

        for row in image_copy:
            for pixel in row:
                blue_sum += pixel[0]
                green_sum += pixel[1]
                red_sum += pixel[2]
                pixel_count += 1

        return blue_sum / pixel_count, green_sum / pixel_count, red_sum / pixel_count

    # This approach is more efficient because it uses NumPy's optimized array operations.
    def average_color(self, image):
        return np.mean(image, axis=(0, 1))

    def calculate_distance_slow(self, image_avg1, image_avg2):
        return np.linalg.norm(image_avg1 - image_avg2)

    # This approach is around 3.5 times faster
    def calculate_distance(self, image_avg1, image_avg2):
        return ((image_avg1[0] - image_avg2[0]) ** 2 + (image_avg1[1] - image_avg2[1]) ** 2 + (image_avg1[2] - image_avg2[2]) ** 2) ** 0.5

    def file_exists(self, file_path):
        if os.path.exists(file_path):
            return True
        return False

    def load_from_file(self, file_path):
        file = open(file_path, "rb")
        structure = pickle.loads(file.read())
        file.close()
        return structure

    def save_to_file(self, structure, file_path):
        file = open(file_path, "wb")
        file.write(pickle.dumps(structure))
        file.close()



