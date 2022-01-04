from argparse import ArgumentParser
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


def main():
    parser = ArgumentParser(description="Download script for the unsplash dataset")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Path to the output folder")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    filename = data_path / "photos.tsv000"
    df = pd.read_csv(filename, sep='\t', header=0)

    for i in range(5):
        print(df["photo_description"][i])

        img_url = df["photo_image_url"][i] + "?fm=jpg&w=1080&q=85&fit=max"
        req = Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        img = urlopen(req).read()
        with open(output_path / f"./out_{i}.jpg", "wb") as file:
            file.write(img)


if __name__ == "__main__":
    main()
