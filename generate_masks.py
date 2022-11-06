import argparse
from collections import namedtuple
from pathlib import Path
from scipy.io import loadmat
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--egohands_dir", type=str, default="egohands_data")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


Sample = namedtuple("Sample", ["image", "polygons"])


def main(args):
    samples = getSamples(args.egohands_dir)
    # print(samples)
    show_sample(samples[0])
    pass


def getSamples(egohands_dir):
    egohands_path = Path(egohands_dir)
    egohands_path.glob("_LABELLED_SAMPLES/*/")
    samples = []
    folders = list(egohands_path.glob("_LABELLED_SAMPLES/*"))
    for folder in folders:
        # print("loading {} directory".format(folder))
        polygons_path = folder / "polygons.mat"
        if not polygons_path.exists():
            continue
        polygons = loadmat(polygons_path, squeeze_me=True)["polygons"]
        image_paths = list(folder.glob("*.jpg"))
        image_paths.sort(key=lambda x: x.stem.lower())
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            samples.append(Sample(image, polygons[i]))
            break

        # print("loaded {} directory".format(folder))

    return samples


def polygons_to_binary(polygon, width, height):
    img = Image.new("L", (width, height), 0)
    polygon_list = polygon.flatten().tolist()
    # polygon_list = [(polygon[i][0], polygon[i][1]) for i in range(polygon.shape[0])]
    # polygon_list = [(200, 200), (200, 300), (300, 300), (300, 200)]
    if len(polygon_list) == 0:
        return img
    ImageDraw.Draw(img).polygon(polygon_list, outline=1, fill=1)
    mask = np.array(img)
    return mask


def show_sample(sample):
    my_left, my_right, your_left, your_right = sample.polygons
    my_left_mask = polygons_to_binary(my_left, sample.image.width, sample.image.height)
    my_right_mask = polygons_to_binary(
        my_right, sample.image.width, sample.image.height
    )
    your_left_mask = polygons_to_binary(
        your_left, sample.image.width, sample.image.height
    )
    your_right_mask = polygons_to_binary(
        your_right, sample.image.width, sample.image.height
    )

    all_masks = np.bitwise_or(my_left_mask, my_right_mask)
    all_masks = np.bitwise_or(all_masks, your_left_mask)
    all_masks = np.bitwise_or(all_masks, your_right_mask)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(sample.image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(all_masks)
    plt.show()
    # print(np.array(your_right_mask).sum())
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
