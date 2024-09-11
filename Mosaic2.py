import pickle

import cv2
import numpy as np
import os
from scipy.spatial import KDTree


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

    def check_existing_database(self, database_file_path):
        if os.path.exists(database_file_path):
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


class MosaicProcessor:

    def __init__(self, tiles_path, tile_size=(32, 32), possible_override_database=False):
        self.tiles_database_path = f"{tile_size[0]}x{tile_size[1]}-tiles-database.pickle"
        self.average_database_path = f"{tile_size[0]}x{tile_size[1]}-average-database.pickle"
        self.processor = ImageProcessor()
        self.tile_size = tile_size
        self.tiles = self.populate_tiles_database(tiles_path, tile_size)
        self.index_table = self.populate_average_database()

    def populate_tiles_database(self, tiles_path, tile_size):
        print("Populating tiles database...")

        if self.processor.check_existing_database(self.tiles_database_path):
            print("Found existing database")
            return self.processor.load_from_file(self.tiles_database_path)

        tiles_names = self.processor.read_image_names(tiles_path)
        tiles = dict()
        for tile_name in tiles_names:
            tile = cv2.imread(tile_name, 1)
            new_tile_size = tiles_size[1], tile_size[0]
            tile = cv2.resize(tile, new_tile_size)
            tiles[tile_name] = tile

        self.processor.save_to_file(tiles, self.tiles_database_path)
        return tiles

    def __get_regions(self, image):
        height, width, depth = image.shape
        tile_height, tile_width = self.tile_size
        regions = [image[y:y + tile_height, x:x + tile_width] for y in range(0, height, tile_height) for x in
                   range(0, width, tile_width)]
        return regions

    def average_color(self, image):
        return np.mean(image, axis=(0, 1))

    def populate_average_database(self):
        print("Populating averages database...")

        if processor.check_existing_database(self.average_database_path):
            print("Found existing database")
            return self.processor.load_from_file(self.average_database_path)

        index = dict()
        for tile_name, tile in self.tiles.items():
            average_color = self.average_color(tile)
            index[tuple(average_color)] = tile_name

        self.processor.save_to_file(index, self.average_database_path)
        return index

    def find_best_match_for_region(self, region):
        region_average = self.average_color(region)
        closest_tile = None
        min_distance = float("inf")
        for tile_average, tile_name in self.index_table.items():
            distance = np.linalg.norm(region_average - tile_average)
            if distance < min_distance:
                min_distance = distance
                closest_tile = self.tiles[tile_name]

        return closest_tile

    def find_best_match(self, region_average, kdtree):
        distance, index = kdtree.query(region_average)
        return index

    def create_mosaic(self, image_name):
        print("Creating a mosaic...(this action can take couple of seconds)")
        image = cv2.imread(image_name, 1)
        image_regions = self.__get_regions(image)

        image_region_average = [self.average_color(region) for region in image_regions]

        kdtree = KDTree(list(self.index_table.keys()))
        best_match_indices = [self.find_best_match(region_average, kdtree) for region_average in image_region_average]

        mosaic = np.zeros(image.shape, dtype=np.uint8)
        height, width, depth = image.shape
        tile_height, tile_width = self.tile_size
        i = 0
        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                # print(image_regions[i].shape)
                if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                    i += 1
                    continue
                best_tile_index = best_match_indices[i]
                best_tile_name = (list(self.index_table.values()))[best_tile_index]
                mosaic[y:y + tile_height, x:x + tile_width] = self.tiles[best_tile_name]
                i += 1

        return mosaic


if __name__ == '__main__':
    processor = ImageProcessor()
    tiles_size = (4, 4)
    mosaic_processor = MosaicProcessor("tiles", tiles_size)
    input_image = "unreal.jpeg"
    mosaic_image = mosaic_processor.create_mosaic(f"./src/{input_image}")
    cv2.imwrite(f"dest/mosaic-picture-{input_image.split('.')[0]}-{tiles_size[0]}x{tiles_size[1]}.jpg", mosaic_image)
    processor.show_image(mosaic_image, "Mosaic Picture")
