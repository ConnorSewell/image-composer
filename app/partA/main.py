import app.utils.source_indexer as si
import app.utils.mozaic_builder as mb


def start(imgPath, output_height, output_width, window_size):
    si.start_indexing(window_size)
    mb.main(inputImagePath=imgPath, tileSize= window_size)