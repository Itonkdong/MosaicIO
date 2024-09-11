import os.path
import time

import cv2
import numpy as np
from scipy.spatial import KDTree
from ImageProcessor import ImageProcessor


class MosaicProcessor:
    __DEFAULT_MOSAIC_TITLE = "My Mosaic Picture"
    __DEFAULT_TILE_SIZE = (32, 32)
    __DEFAULT_TILES_FOLDER = f"{os.path.abspath(os.getcwd())}\\tiles"
    __DEFAULT_LOG_FILE_PATH = f"{os.path.abspath(os.getcwd())}\\logs\\log.txt"
    __DEFAULT_DEST_FOLDER_PATH = f"{os.path.abspath(os.getcwd())}\\dest"

    def __init__(self, image_processor: ImageProcessor, tiles_path=__DEFAULT_TILES_FOLDER,
                 possible_override_database=False):

        self.tiles_path = tiles_path
        self.image_processor = image_processor
        self.last_execution_time = -1

    def __populate_and_get_tiles_database(self, tiles_path, tile_size):
        database_filename = f"{os.path.abspath(os.getcwd())}\\databases\\{tile_size[0]}x{tile_size[1]}-tiles-database.pickle"

        if self.image_processor.file_exists(database_filename):
            print(f"Found existing tiles database for tile size: {tile_size[0]}x{tile_size[1]}")
            return self.image_processor.load_from_file(database_filename)

        print("Calculating and populating tiles database...")
        tiles_names = self.image_processor.read_image_names(tiles_path)
        tiles = dict()
        new_tile_size = tile_size[1], tile_size[0]
        for tile_name in tiles_names:
            tile = cv2.imread(tile_name, 1)
            tile = cv2.resize(tile, new_tile_size)
            tiles[tile_name] = tile

        self.image_processor.save_to_file(tiles, database_filename)
        return tiles

    def __populate_and_get_average_database(self, tiles, tile_size):
        database_filename = f"{os.path.abspath(os.getcwd())}\\databases\\{tile_size[0]}x{tile_size[1]}-average-database.pickle"

        if self.image_processor.file_exists(database_filename):
            print(f"Found existing average database for tile size: {tile_size[0]}x{tile_size[1]}")
            return self.image_processor.load_from_file(database_filename)

        print("Calculating and populating averages database...")
        index_table = dict()
        for tile_name, tile in tiles.items():
            average_color = self.image_processor.average_color(tile)
            index_table[tuple(average_color)] = tile_name

        self.image_processor.save_to_file(index_table, database_filename)
        return index_table

    def __get_regions(self, image, tile_size):
        height, width, depth = image.shape
        tile_height, tile_width = tile_size
        regions = [image[y:y + tile_height, x:x + tile_width] for y in range(0, height, tile_height) for x in
                   range(0, width, tile_width)]
        return regions

    def find_best_match_for_region_improved(self, index_table, tiles, region, kd_tree):
        region_average = self.image_processor.average_color(region)
        distance, i = kd_tree.query(region_average)
        tile_name = list(index_table.values())[i]
        closest_tile = tiles[tile_name]
        return closest_tile

    def find_best_match_for_region_base(self, index_table, tiles, region):
        region_average = self.image_processor.average_color(region)
        closest_tile = None
        min_distance = float("inf")
        for tile_average, tile_name in index_table.items():
            distance = np.linalg.norm(region_average - tile_average)
            if distance < min_distance:
                min_distance = distance
                closest_tile = tiles[tile_name]

        return closest_tile

    def create_mosaic(self, image_filepath, tile_size=__DEFAULT_TILE_SIZE, improved=True,
                      write=True,
                      title=__DEFAULT_MOSAIC_TITLE,
                      dest_folder=__DEFAULT_DEST_FOLDER_PATH,
                      write_log=False,
                      log_filepath=__DEFAULT_LOG_FILE_PATH):

        tiles = self.__populate_and_get_tiles_database(self.tiles_path, tile_size)
        index_table = self.__populate_and_get_average_database(tiles, tile_size)

        print("Creating a mosaic...(this action can take couple of seconds)")
        image = cv2.imread(image_filepath, 1)

        start_time = time.time()
        image_regions = self.__get_regions(image, tile_size)

        kd_tree = None
        if improved:
            kd_tree = KDTree(list(index_table.keys()))

        mosaic = np.zeros(image.shape, dtype=np.uint8)
        height, width, depth = image.shape
        tile_height, tile_width = tile_size

        i = 0
        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                    i += 1
                    continue

                if improved:
                    best_tile = self.find_best_match_for_region_improved(index_table, tiles, image_regions[i], kd_tree)
                else:
                    best_tile = self.find_best_match_for_region_base(index_table, tiles, image_regions[i])

                mosaic[y:y + tile_height, x:x + tile_width] = best_tile
                i += 1
        end_time = time.time()
        self.last_execution_time = end_time - start_time

        if write:
            if title == MosaicProcessor.__DEFAULT_MOSAIC_TITLE:
                cv2.imwrite(
                    f"{dest_folder}\\{title}-{tile_size[0]}x{tile_size[1]}-{"Improved" if improved else "Base"}.jpg",
                    mosaic)
            else:
                cv2.imwrite(f"{MosaicProcessor.__DEFAULT_DEST_FOLDER_PATH}\\{title}.jpg",
                            mosaic)
        if write_log:
            MosaicProcessor.write_log(image_filepath, image, tile_size, self.last_execution_time, improved,
                                      log_filepath)

        return mosaic

    def get_last_execution_time(self):
        return self.last_execution_time

    def get_default_dest_folder_path(self):
        return self.__DEFAULT_DEST_FOLDER_PATH

    @staticmethod
    def write_log(image_filepath, img, tile_size, execution_time, improved_algorithm,
                  log_filepath):

        height, width, depth = img.shape
        img_name = image_filepath.split("\\")[-1]
        tile_height, tile_width = tile_size
        algorithm_type = "Improved" if improved_algorithm else "Base"
        with open(log_filepath, "a") as f:
            f.write(
                f"Img Path: {image_filepath}, Name: {img_name}, Size: {width}x{height}, Tile Size: {tile_width}x{tile_height}, Algorithm: {algorithm_type}, Exe. time: " + str(
                    execution_time) + "\n")


if __name__ == '__main__':
    image_processor = ImageProcessor()
    mosaic_processor = MosaicProcessor(image_processor)

    tiles_size = (9, 9)
    img_name = "4k.jpg"
    img_filepath = f"{os.path.abspath(os.getcwd())}\\test_images\\{img_name}"

    mosaic_image = mosaic_processor.create_mosaic(img_filepath, tiles_size, improved=True, write_log=True)

    print(f"Execution time {mosaic_processor.get_last_execution_time()}")

    image_processor.show_image(mosaic_image, "Mosaic Picture")
