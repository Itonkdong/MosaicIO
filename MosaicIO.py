import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from Mosaic4 import *
from ImageProcessor import ImageProcessor


class MosaicIO:
    __APP_NAME = "Mosaic.IO"
    __WINDOWS_SIZE = "450x650"
    __AVAILABLE_TILE_SIZES = ["2x2", "4x4", "8x8", "16x16", "32x32", "32x24"]
    __MOSAIC_DEFAULT_TITLE = "MyMosaicFromMosaicIO"

    def __init__(self, mosaic_processor: MosaicProcessor, image_processor: ImageProcessor, root: tk.Tk) -> None:
        self.__mosaic_processor = mosaic_processor
        self.__image_processor = image_processor

        self.__root = root
        self.__combobox = None
        self.__selected_img_name = None
        self.__img_name_label = None
        self.__mosaic_title_entry = None
        self.__mosaic_title = ""
        self.__full_save_path_label = None
        self.__info_label = None
        self.__tile_size = None
        self.__algorithm_choice = None
        self.__mosaic = None
        self.__creation_time = 0
        self.__show_mosaic_button = None

        self.__setup_window()

    def __get_and_show_mosaic_title(self):
        self.__mosaic_title = self.__mosaic_title_entry.get()
        self.__full_save_path_label.config(
            text=f"You mosaic will be saved in:\n {self.__mosaic_processor.get_default_dest_folder_path()}\\{self.__mosaic_title}.jpg")
        self.__full_save_path_label.pack(pady=(10, 20))

    def __prepare_do_mosaic(self):

        try:
            self.__tile_size = tuple(map(int, self.__combobox.get().split("x")))
        except Exception as e:
            self.show_info("Invalid tile size.")
            return

        if self.__tile_size[0] > 32 or self.__tile_size[1] > 32:
            self.show_info("Tile size cannot be greater than 32x32.")
            return

        # The user has forgotten to click "Confirm"
        if self.__mosaic_title == "":
            self.__get_and_show_mosaic_title()

        if not self.__selected_img_name:
            self.show_info("You must select an image first.")
            return

        self.show_info("Creating a mosaic...(this action can take couple of seconds)")

        self.__root.after(50, self.__do_mosaic)

    def __do_mosaic(self, ):
        start_time = time.time()
        self.__mosaic = self.__mosaic_processor.create_mosaic(self.__selected_img_name, self.__tile_size,
                                                              title=self.__mosaic_title)

        end_time = time.time()

        self.__show_mosaic_button.pack(pady=(10, 10))
        self.show_info(
            f"Successfully created and saved your mosaic. \nExecution time: {end_time - start_time} seconds.")
        # self.__image_processor.show_image(self.__mosaic, self.__mosaic_title)

    def hide_info(self):
        self.__info_label.config(text="")

    def show_info(self, message: str):
        self.__info_label.config(text=message)

    def show_mosaic(self):
        if self.__mosaic is None:
            return

        self.__image_processor.show_image(self.__mosaic, self.__mosaic_title)

    # Function to handle file selection
    def __on_file_select(self):
        self.__selected_img_name = filedialog.askopenfilename()
        self.__img_name_label.config(text=f"Selected: {self.__selected_img_name}")
        self.__img_name_label.pack(pady=10)  # Make the label visible

    def __setup_window(self):
        self.__root.title(MosaicIO.__APP_NAME)
        self.__root.geometry(MosaicIO.__WINDOWS_SIZE)

        # Add a title label
        title_label = tk.Label(self.__root, text=MosaicIO.__APP_NAME, font=("Arial", 16))
        title_label.pack(pady=5)

        sub_title = tk.Label(self.__root, text="Create a mosaic from your picture!", font=("Arial", 12))
        sub_title.pack(pady=10)

        # Add a button to open the file selector
        file_button = tk.Button(self.__root, text="Select your picture", command=self.__on_file_select)
        file_button.pack(pady=10)

        # Add an input field
        self.__img_name_label = tk.Label(self.__root, text="")
        self.__img_name_label.pack(pady=5)

        # Add an input field
        entry_label = tk.Label(self.__root, text="Select or input tile's size")
        entry_label.pack(pady=5)
        self.__combobox = ttk.Combobox(self.__root, values=MosaicIO.__AVAILABLE_TILE_SIZES)
        self.__combobox.pack(pady=5)
        self.__combobox.current(2)

        saved_mosaic_title_label = tk.Label(self.__root, text="Give your mosaic a title:")
        saved_mosaic_title_label.pack(pady=(20, 5))
        self.__mosaic_title_entry = ttk.Entry(self.__root, width=40)
        self.__mosaic_title_entry.pack(pady=(5, 20))
        self.__mosaic_title_entry.insert(0, self.__MOSAIC_DEFAULT_TITLE)

        confirm_mosaic_title_button = tk.Button(self.__mosaic_title_entry, text="Confirm",
                                                command=self.__get_and_show_mosaic_title)
        confirm_mosaic_title_button.pack(side="right", padx=(250, 0))

        self.__full_save_path_label = tk.Label(self.__root, text="")
        self.__full_save_path_label.pack()

        # Create a label
        label = tk.Label(self.__root, text="Choose an Algorithm:")
        label.pack(pady=5)

        # Create a Tkinter variable to hold the selected value of the radio buttons
        self.algorithm_choice = tk.StringVar(value="Improved")  # Set a default value

        # Create two radio buttons for "Base Algorithm" and "Improved Algorithm"
        self.base_algorithm_rb = tk.Radiobutton(self.__root, text="Base Algorithm", variable=self.algorithm_choice,
                                                value="Base")
        self.base_algorithm_rb.pack()

        self.improved_algorithm_rb = tk.Radiobutton(self.__root, text="Improved Algorithm",
                                                    variable=self.algorithm_choice, value="Improved")
        self.improved_algorithm_rb.pack()

        # Add a button to trigger the button click function
        button = tk.Button(self.__root, text="Create Mosaic", command=self.__prepare_do_mosaic)
        button.pack(pady=20)

        self.__show_mosaic_button = tk.Button(self.__root, text="Show Mosaic", command=self.show_mosaic)

        self.__info_label = tk.Label(self.__root, text="", fg="red")
        self.__info_label.pack(pady=(10, 5))

    def start(self):
        # Start the Tkinter event loop
        self.__root.mainloop()


if __name__ == '__main__':
    mosaic_processor = MosaicProcessor()
    image_processor = ImageProcessor()
    root = tk.Tk()
    app = MosaicIO(mosaic_processor, image_processor, root)
    app.start()
