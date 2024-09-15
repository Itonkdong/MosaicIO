import concurrent.futures
import os.path
import random
import time

import cv2
import numpy as np
from scipy.spatial import KDTree
from ImageProcessor import ImageProcessor


class MosaicProcessor:
    __DEFAULT_MOSAIC_TITLE = "My Mosaic Picture"
    __DEFAULT_TILE_SIZE = (16, 16)
    __DEFAULT_TILES_FOLDER = f"{os.path.abspath(os.getcwd())}\\tiles"
    __DEFAULT_LOG_FILE_PATH = f"{os.path.abspath(os.getcwd())}\\logs\\log.txt"
    __DEFAULT_DEST_FOLDER_PATH = f"{os.path.abspath(os.getcwd())}\\dest"
    __MAX_WORKERS = None
    __DEFAULT_RANDOMIZATION_VALUE = 30

    KEEP_SAME_SIZE = 1
    FILL_BORDERS = 2

    def __init__(self, image_processor: ImageProcessor, tiles_path=__DEFAULT_TILES_FOLDER, override_database=False):
        self.__override_database = override_database
        self.tiles_path = tiles_path
        self.image_processor = image_processor
        self.last_execution_time = -1

        self.__index_table = None
        self.__tiles = None
        self.__kd_tree = None
        self.__save_index_table = None
        self.__save_randomization_value = None

    def __populate_and_get_tiles_database(self, tiles_path, tile_size):
        database_filename = f"{os.path.abspath(os.getcwd())}\\databases\\{tile_size[0]}x{tile_size[1]}-tiles-database.pickle"

        if self.image_processor.file_exists(database_filename) and not self.__override_database:
            # print(f"Found existing tiles database for tile size: {tile_size[0]}x{tile_size[1]}")
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

        if self.image_processor.file_exists(database_filename) and not self.__override_database:
            # print(f"Found existing average database for tile size: {tile_size[0]}x{tile_size[1]}")
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

    def __prepare_index_randomization_improved(self, image_region):
        random_index_keys = random.sample(list(self.__save_index_table.keys()), self.__save_randomization_value)
        index_table = {key: self.__save_index_table[key] for key in random_index_keys}
        kd_tree = KDTree(list(index_table.keys()))
        return self.find_best_match_for_region_improved_parallel(image_region, kd_tree, index_table)

    def __prepare_index_randomization_base(self, image_region):
        random_index_keys = random.sample(list(self.__save_index_table.keys()), self.__save_randomization_value)
        self.__index_table = {key: self.__save_index_table[key] for key in random_index_keys}
        return self.find_best_match_for_region_base(image_region)

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

    def __create_mosaic_parallel(self, mosaic_shape, tile_size, image_regions, fill_option, improved,
                                 randomization_improvement, randomization_value, max_workers):
        mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        mosaic_height, mosaic_width, depth = mosaic_shape
        tile_height, tile_width = tile_size

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

        # Place the matched tiles in the correct position to form the mosaic
        i = 0
        for y in range(0, mosaic_height, tile_height):
            for x in range(0, mosaic_width, tile_width):
                if randomization_value == 0:
                    break

                # Determines whether to fill the image or to keep te original size
                if fill_option == MosaicProcessor.KEEP_SAME_SIZE:
                    if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                        i += 1
                        continue

                mosaic[y:y + tile_height, x:x + tile_width] = matched_tiles[i]
                i += 1

        return mosaic

    def __create_mosaic_sequential(self, mosaic_shape, tile_size, image_regions, fill_option, improved,
                                   randomization_improvement, randomization_value, best_match_region_times):
        mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        mosaic_height, mosaic_width, depth = mosaic_shape
        tile_height, tile_width = tile_size

        i = 0
        for y in range(0, mosaic_height, tile_height):
            for x in range(0, mosaic_width, tile_width):

                if randomization_value == 0:
                    break

                # Determines whether to fill the image or to keep te original size
                if fill_option == MosaicProcessor.KEEP_SAME_SIZE:
                    if image_regions[i].shape[1] != tile_width or image_regions[i].shape[0] != tile_height:
                        i += 1
                        continue

                per_region_start_time = time.time()

                # Weather to use randomization improvement
                if randomization_improvement:
                    random_index_keys = random.sample(list(self.__save_index_table.keys()), randomization_value)
                    self.__index_table = {key: self.__save_index_table[key] for key in random_index_keys}

                if randomization_improvement and improved:
                    self.__kd_tree = KDTree(list(self.__index_table.keys()))

                # What algorithm to use
                if improved:
                    best_tile = self.find_best_match_for_region_improved(image_regions[i])
                else:
                    best_tile = self.find_best_match_for_region_base(image_regions[i])
                    # best_tile = self.find_best_match_for_region_base_old(index_table, tiles, image_regions[i])

                best_match_region_times.append(time.time() - per_region_start_time)

                mosaic[y:y + tile_height, x:x + tile_width] = best_tile
                i += 1

        return mosaic

    def create_mosaic(self, image_filepath, tile_size=__DEFAULT_TILE_SIZE, fill_option=KEEP_SAME_SIZE,
                      improved=False,
                      randomization_improvement=False,
                      randomization_value=__DEFAULT_RANDOMIZATION_VALUE,
                      parallel_processing=False,
                      max_workers=__MAX_WORKERS,
                      write=False,
                      title=__DEFAULT_MOSAIC_TITLE,
                      dest_folder=__DEFAULT_DEST_FOLDER_PATH,
                      write_log=False,
                      log_filepath=__DEFAULT_LOG_FILE_PATH):

        if tile_size[0] > 128 or tile_size[1] > 128:
            raise Exception("Tile size cannot be greater than 128")

        self.__tiles = self.__populate_and_get_tiles_database(self.tiles_path, tile_size)
        self.__index_table = self.__populate_and_get_average_database(self.__tiles, tile_size)

        if randomization_value > len(self.__index_table.keys()) or randomization_value < 0:
            randomization_improvement = False

        if randomization_improvement:
            self.__save_index_table = self.__populate_and_get_average_database(self.__tiles, tile_size)

        print("Creating a mosaic...")
        image = cv2.imread(image_filepath, 1)

        start_time = time.time()
        image_regions = self.__get_regions(image, tile_size)

        if improved:
            self.__kd_tree = KDTree(list(self.__index_table.keys()))

        mosaic_shape = self.get_mosaic_size(image.shape, tile_size, fill_option)

        best_match_region_times = []

        if parallel_processing:
            mosaic = self.__create_mosaic_parallel(mosaic_shape, tile_size, image_regions, fill_option, improved,
                                                   randomization_improvement, randomization_value, max_workers)

        else:
            mosaic = self.__create_mosaic_sequential(mosaic_shape, tile_size, image_regions, fill_option, improved,
                                                     randomization_improvement, randomization_value,
                                                     best_match_region_times)

        self.last_execution_time = time.time() - start_time

        if write:
            if title == MosaicProcessor.__DEFAULT_MOSAIC_TITLE:
                cv2.imwrite(
                    f"{dest_folder}\\{title}-{tile_size[0]}x{tile_size[1]}-{"Improved" if improved else "Base"}.jpg",
                    mosaic)
            else:
                cv2.imwrite(f"{dest_folder}\\{title}.jpg",
                            mosaic)
        if write_log:
            region_average_time = format(np.average(best_match_region_times), '.10f') if len(
                best_match_region_times) != 0 else str(0)
            MosaicProcessor.write_log(image_filepath, image, tile_size, len(image_regions), region_average_time,
                                      self.last_execution_time, improved,
                                      randomization_improvement,
                                      randomization_value,
                                      parallel_processing,
                                      max_workers,
                                      log_filepath)

        return mosaic

    def get_last_execution_time(self):
        return self.last_execution_time

    def get_default_dest_folder_path(self):
        return self.__DEFAULT_DEST_FOLDER_PATH

    @staticmethod
    def get_formatted_tile_size(tile_size):
        return f"{tile_size[0]}x{tile_size[1]}"

    def get_mosaic_size(self, image_size, tile_size, option):
        height, width, depth = image_size

        match option:
            case MosaicProcessor.KEEP_SAME_SIZE:
                return height, width, depth
            case MosaicProcessor.FILL_BORDERS:
                tile_height, tile_width = tile_size
                if height % tile_height != 0:
                    mosaic_height = ((height // tile_height) * tile_height) + tile_height
                else:
                    mosaic_height = ((height // tile_height) * tile_height)

                if width % tile_width != 0:
                    mosaic_width = ((width // tile_width) * tile_width) + tile_width
                else:
                    mosaic_width = ((width // tile_width) * tile_width)

                return mosaic_height, mosaic_width, depth

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


def do_experiment_1(improved=True, base=False):
    if not improved and not base:
        return

    image_processor = ImageProcessor()
    mosaic_processor = MosaicProcessor(image_processor)

    with open("./test_results/test_run.txt", "r") as f:
        test_run = int(f.readline())

    with open("./test_results/test_run.txt", "w") as f:
        f.write(str(test_run + 1))

    test_folder_path = "./test_images/single"
    test_results_folder_path = f"./test_results/{test_run}"
    log_filepath = f"./test_results/{test_run}/test_log.txt"
    all_tile_sizes = [(4,4),(8,8), (16,16), (32,32), (64,64)]

    if not os.path.isdir(test_results_folder_path):
        os.mkdir(test_results_folder_path)

    image_paths = image_processor.read_image_names(test_folder_path)

    for img_filepath in image_paths:
        img_name = img_filepath.split("/")[-1].split(".")[0]

        for tile_size in all_tile_sizes:
            title_base = f"test-{img_name}-{MosaicProcessor.get_formatted_tile_size(tile_size)}-base"
            title_improved = f"test-{img_name}-{MosaicProcessor.get_formatted_tile_size(tile_size)}-improved"
            print(f"Running experiment on: {img_name}, tile_size: {tile_size}")
            if improved:
                mosaic_processor.create_mosaic(img_filepath, tile_size, title=title_improved,
                                               dest_folder=test_results_folder_path,
                                               improved=True,write=True ,write_log=True, log_filepath=log_filepath)
                print(f"Execution time: {mosaic_processor.get_last_execution_time()}")

            if base:
                mosaic_processor.create_mosaic(img_filepath, tile_size, title=title_base,
                                               dest_folder=test_results_folder_path,
                                               improved=False, write_log=True, log_filepath=log_filepath)
                print(f"Execution time: {mosaic_processor.get_last_execution_time()}")


def do_experiment_2():
    image_processor = ImageProcessor()
    mosaic_processor = MosaicProcessor(image_processor)

    with open("./experiments/experiment_run.txt", "r") as f:
        run_number = int(f.readline())

    with open("./experiments/experiment_run.txt", "w") as f:
        f.write(str(run_number + 1))

    img_filepath = "./test_images/both/girl.jpg"
    dest_folder = f"./experiments/{run_number}"
    log_filepath = f"./experiments/{run_number}/log.txt"
    tile_size = (16, 16)
    randomization_value = 60

    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    combinations = (True, False)

    for improved in combinations:
        for randomization in combinations:
            for parallelization in combinations:
                title = f"{"improved" if improved else "base"}-{"rand" if randomization else "no_rand"}-{"para" if parallelization else "no_para"}"
                mosaic_processor.create_mosaic(img_filepath, tile_size,
                                               improved=improved,
                                               randomization_improvement=randomization,
                                               randomization_value=randomization_value,
                                               parallel_processing=parallelization,
                                               max_workers=None,
                                               write_log=True,
                                               write=True,
                                               title=title,
                                               fill_option=MosaicProcessor.KEEP_SAME_SIZE,
                                               dest_folder=dest_folder,
                                               log_filepath=log_filepath)


def normal_run():
    image_processor = ImageProcessor()
    mosaic_processor = MosaicProcessor(image_processor, "./tiles", False)

    tiles_size = (32, 32)
    img_filepath = "./test_images/both/girl.jpg"

    mosaic_image = mosaic_processor.create_mosaic(img_filepath, tiles_size,
                                                  improved=False,
                                                  randomization_improvement=True,
                                                  randomization_value=2,
                                                  parallel_processing=False,
                                                  max_workers=None,
                                                  write_log=True,
                                                  write=True,
                                                  title="Test-28",
                                                  fill_option=MosaicProcessor.FILL_BORDERS)

    print(f"Execution time {mosaic_processor.get_last_execution_time()}")

    image_processor.show_image(mosaic_image, "Mosaic Picture")


if __name__ == '__main__':
    # do_experiment_1()
    normal_run()
    # print(os.cpu_count())
    # do_experiment_2()
