import concurrent.futures
import os.path
import random
import time
import cv2
import numpy as np
from scipy.spatial import KDTree
from ImageProcessor import ImageProcessor


class MosaicProcessor:
    __DEFAULT_TILE_SIZE = (16, 16)
    __DEFAULT_TILES_FOLDER = f"{os.path.abspath(os.getcwd())}\\tiles"

    KEEP_SAME_SIZE = 1
    FILL_BORDERS = 2

    __DEFAULT_RANDOMIZATION_VALUE = 30

    __MAX_WORKERS = None

    __DEFAULT_MOSAIC_TITLE = "My Mosaic Picture"
    __DEFAULT_DEST_FOLDER_PATH = f"{os.path.abspath(os.getcwd())}\\dest"

    __DEFAULT_LOG_FILE_PATH = f"{os.path.abspath(os.getcwd())}\\logs\\log.txt"

    def __init__(self, image_processor: ImageProcessor, tiles_path=__DEFAULT_TILES_FOLDER, override_database=False):
        self.image_processor = image_processor
        self.tiles_path = tiles_path
        self.__tiles = None
        self.__index_table = None

        self.__override_database = override_database
        self.last_execution_time = -1

        self.__save_index_table = None

        self.__kd_tree = None

        self.__save_randomization_value = None

    def create_mosaic(self, image_filepath, tiles_size=__DEFAULT_TILE_SIZE, fill_option=KEEP_SAME_SIZE,
                      randomization_improvement=False,
                      randomization_value=__DEFAULT_RANDOMIZATION_VALUE,
                      improved=False,
                      parallel_processing=False,
                      max_workers=__MAX_WORKERS,
                      write=False,
                      title=__DEFAULT_MOSAIC_TITLE,
                      dest_folder=__DEFAULT_DEST_FOLDER_PATH,
                      write_log=False,
                      log_filepath=__DEFAULT_LOG_FILE_PATH):

        if tiles_size[0] > 128 or tiles_size[1] > 128:
            raise Exception("Tile size cannot be greater than 128")

        self.__tiles = self.__populate_and_get_tiles_database(self.tiles_path, tiles_size)
        self.__index_table = self.__populate_and_get_average_database(self.__tiles, tiles_size)

        if randomization_value > len(self.__index_table.keys()) or randomization_value < 0:
            randomization_improvement = False

        if randomization_improvement:
            self.__save_index_table = self.__populate_and_get_average_database(self.__tiles, tiles_size)

        print("Creating a mosaic...")
        image = cv2.imread(image_filepath, 1)

        start_time = time.time()
        image_regions = self.__get_regions(image, tiles_size)

        if improved:
            self.__kd_tree = KDTree(list(self.__index_table.keys()))

        mosaic_shape = self.get_mosaic_size(image.shape, tiles_size, fill_option)

        best_match_region_times = []

        if parallel_processing:
            mosaic = self.__create_mosaic_parallel(mosaic_shape, tiles_size, image_regions, fill_option, improved,
                                                   randomization_improvement, randomization_value, max_workers)

        else:
            mosaic = self.__create_mosaic_sequential(mosaic_shape, tiles_size, image_regions, fill_option, improved,
                                                     randomization_improvement, randomization_value,
                                                     best_match_region_times)

        self.last_execution_time = time.time() - start_time

        if write:
            if title == MosaicProcessor.__DEFAULT_MOSAIC_TITLE:
                cv2.imwrite(
                    f"{dest_folder}\\{title}-{tiles_size[0]}x{tiles_size[1]}-{"Improved" if improved else "Base"}.jpg",
                    mosaic)
            else:
                cv2.imwrite(f"{dest_folder}\\{title}.jpg",
                            mosaic)

        if write_log:
            region_average_time = format(np.average(best_match_region_times), '.10f') if len(
                best_match_region_times) != 0 else str(0)
            MosaicProcessor.write_log(image_filepath, image, tiles_size, len(image_regions), region_average_time,
                                      self.last_execution_time, improved,
                                      randomization_improvement,
                                      randomization_value,
                                      parallel_processing,
                                      max_workers,
                                      log_filepath)

        return mosaic

    def __create_mosaic_parallel(self, mosaic_shape, tiles_size, image_regions, fill_option, improved,
                                 randomization_improvement, randomization_value, max_workers):
        mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        mosaic_height, mosaic_width, depth = mosaic_shape
        tile_height, tile_width = tiles_size

        if improved:
            if randomization_improvement and randomization_value > 0:
                self.__save_randomization_value = randomization_value
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    matched_tiles = list(executor.map(self.__prepare_index_randomization_improved, image_regions))
            elif not randomization_improvement:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    matched_tiles = list(executor.map(self.find_best_match_for_region_improved, image_regions))

        else:
            if randomization_improvement and randomization_value > 0:
                self.__save_randomization_value = randomization_value
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    matched_tiles = list(executor.map(self.__prepare_index_randomization_base, image_regions))
            elif not randomization_improvement:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    matched_tiles = list(executor.map(self.find_best_match_for_region_base, image_regions))

        i = 0
        for y in range(0, mosaic_height, tile_height):
            for x in range(0, mosaic_width, tile_width):
                if randomization_value == 0:
                    break

                if fill_option == MosaicProcessor.KEEP_SAME_SIZE:
                    if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                        i += 1
                        continue

                mosaic[y:y + tile_height, x:x + tile_width] = matched_tiles[i]
                i += 1

        return mosaic

    def __create_mosaic_sequential(self, mosaic_shape, tiles_size, image_regions, fill_option, improved,
                                   randomization_improvement, randomization_value, best_match_region_times):
        mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        mosaic_height, mosaic_width, depth = mosaic_shape
        tile_height, tile_width = tiles_size

        i = 0
        for y in range(0, mosaic_height, tile_height):
            for x in range(0, mosaic_width, tile_width):

                if randomization_value == 0:
                    break

                if fill_option == MosaicProcessor.KEEP_SAME_SIZE:
                    if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                        i += 1
                        continue

                per_region_start_time = time.time()

                if randomization_improvement:
                    random_index_keys = random.sample(list(self.__save_index_table.keys()), randomization_value)
                    self.__index_table = {key: self.__save_index_table[key] for key in random_index_keys}
                    self.__kd_tree = KDTree(list(self.__index_table.keys()))

                if improved:
                    best_tile = self.find_best_match_for_region_improved(image_regions[i])
                else:
                    best_tile = self.find_best_match_for_region_base(image_regions[i])

                best_match_region_times.append(time.time() - per_region_start_time)

                mosaic[y:y + tile_height, x:x + tile_width] = best_tile
                i += 1

        return mosaic

    def __populate_and_get_tiles_database(self, tiles_path, tiles_size):
        database_filepath = f"{os.path.abspath(os.getcwd())}\\databases\\{tiles_size[0]}x{tiles_size[1]}-tiles-database.pickle"

        if self.image_processor.file_exists(database_filepath) and not self.__override_database:
            return self.image_processor.load_from_file(database_filepath)

        print("Calculating and populating tiles database...")
        tiles_names = self.image_processor.read_image_names(tiles_path)
        tiles = dict()
        new_tile_size = tiles_size[1], tiles_size[0]
        for tile_name in tiles_names:
            tile = cv2.imread(tile_name, 1)
            tile = cv2.resize(tile, new_tile_size)
            tiles[tile_name] = tile

        self.image_processor.save_to_file(tiles, database_filepath)
        return tiles

    def __populate_and_get_average_database(self, tiles, tiles_size):
        database_filepath = f"{os.path.abspath(os.getcwd())}\\databases\\{tiles_size[0]}x{tiles_size[1]}-average-database.pickle"

        if self.image_processor.file_exists(database_filepath) and not self.__override_database:
            return self.image_processor.load_from_file(database_filepath)

        print("Calculating and populating average database...")
        index_table = dict()
        for tile_filepath, tile in tiles.items():
            tile_average_color = self.image_processor.average_color(tile)
            index_table[tuple(tile_average_color)] = tile_filepath

        self.image_processor.save_to_file(index_table, database_filepath)
        return index_table

    def __get_regions(self, image, tiles_size):
        height, width, depth = image.shape
        tile_height, tile_width = tiles_size
        regions = [image[y:y + tile_height, x:x + tile_width] for y in range(0, height, tile_height) for x in
                   range(0, width, tile_width)]
        return regions

    def get_mosaic_size(self, image_size, tiles_size, option):
        height, width, depth = image_size

        match option:
            case MosaicProcessor.KEEP_SAME_SIZE:
                return height, width, depth
            case MosaicProcessor.FILL_BORDERS:
                tile_height, tile_width = tiles_size
                if height % tile_height != 0:
                    mosaic_height = ((height // tile_height) * tile_height) + tile_height
                else:
                    mosaic_height = ((height // tile_height) * tile_height)

                if width % tile_width != 0:
                    mosaic_width = ((width // tile_width) * tile_width) + tile_width
                else:
                    mosaic_width = ((width // tile_width) * tile_width)

                return mosaic_height, mosaic_width, depth

    def find_best_match_for_region_base(self, region):
        region_average = self.image_processor.average_color(region)
        closest_tile = None
        min_distance = float("inf")
        for tile_average, tile_name in self.__index_table.items():
            distance = self.image_processor.calculate_distance(region_average, tile_average)
            if distance < min_distance:
                min_distance = distance
                closest_tile = self.__tiles[tile_name]

        return closest_tile

    def find_best_match_for_region_improved(self, region):
        region_average = self.image_processor.average_color(region)
        distance, i = self.__kd_tree.query(region_average)
        tile_name = list(self.__index_table.values())[i]
        closest_tile = self.__tiles[tile_name]
        return closest_tile

    def find_best_match_for_region_improved_parallel(self, region, kd_tree, index_table):
        region_average = self.image_processor.average_color(region)
        distance, i = kd_tree.query(region_average)
        tile_name = list(index_table.values())[i]
        closest_tile = self.__tiles[tile_name]
        return closest_tile

    def __prepare_index_randomization_improved(self, image_region):
        random_index_keys = random.sample(list(self.__save_index_table.keys()), self.__save_randomization_value)
        index_table = {key: self.__save_index_table[key] for key in random_index_keys}
        kd_tree = KDTree(list(index_table.keys()))
        return self.find_best_match_for_region_improved_parallel(image_region, kd_tree, index_table)

    def __prepare_index_randomization_base(self, image_region):
        random_index_keys = random.sample(list(self.__save_index_table.keys()), self.__save_randomization_value)
        self.__index_table = {key: self.__save_index_table[key] for key in random_index_keys}
        return self.find_best_match_for_region_base(image_region)

    def get_last_execution_time(self):
        return self.last_execution_time

    @staticmethod
    def get_formatted_tile_size(tile_size):
        return f"{tile_size[0]}x{tile_size[1]}"

    @staticmethod
    def write_log(image_filepath, img, tile_size, number_regions, region_average_time, execution_time,
                  improved_algorithm,
                  randomization,
                  randomization_value,
                  parallel_processing,
                  workers,
                  log_filepath):

        split = image_filepath.split("/")
        if len(split) > 1:
            img_name = split[-1]
        else:
            img_name = image_filepath.split("\\")[-1]

        workers = os.cpu_count() if workers is None else workers

        algorithm_type = "Improved" if improved_algorithm else "Base"
        with open(log_filepath, "a") as f:
            f.write(
                f"NAME: {img_name:<20} SIZE: {MosaicProcessor.get_formatted_tile_size((img.shape[0], img.shape[1])):<10} T_SIZE: {MosaicProcessor.get_formatted_tile_size(tile_size):<9} "
                f"REG: {number_regions:<8} REG_AVG: {region_average_time + " s":<15}  "
                f"ALG: {algorithm_type:<10} RAND: {str(randomization):<8} RAND_V: {-1 if not randomization else randomization_value:<5} PARA: {str(parallel_processing):<8} THREADS: {-1 if not parallel_processing else workers:<5}  EXEC_TIME: {execution_time} s\n"
            )


if __name__ == '__main__':
    image_processor = ImageProcessor()
    mosaic_processor = MosaicProcessor(image_processor)

    tiles_size = (32, 32)
    img_filepath = "./test_images/both/girl.jpg"

    mosaic_image = mosaic_processor.create_mosaic(img_filepath, tiles_size,
                                                  randomization_improvement=False,
                                                  randomization_value=30,
                                                  improved=True,
                                                  parallel_processing=True
                                                  )
    print(f"Execution time {mosaic_processor.get_last_execution_time()}")

    image_processor.show_image(mosaic_image, "Mosaic Picture")
