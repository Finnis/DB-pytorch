import os
import shutil
import errno
import logging
import numpy as np
	
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Warning: {e}')


def _create_dir(path):
    """
    Create a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(paths: list, empty=True):
    """Create a directory if it does not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return:
    """
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            _create_dir(path)
        if empty:
            _empty_dir(path)


def clean_ckpts(path, num_max=7):
    filenames = sorted(os.listdir(path),  key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if len(filenames) >= num_max:
        to_delete = filenames[:len(filenames)-num_max]
        for x in to_delete:
            os.remove(os.path.join(path, x))