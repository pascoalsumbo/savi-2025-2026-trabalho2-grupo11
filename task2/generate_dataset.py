import os
from tqdm import tqdm

from synth_dataset import mnist_source, generate_one_scene, save_sample
from utils import ensure_dir, set_seed


def generate_split(mnist_ds, out_split_dir: str, n_images: int,
                   canvas_size: int, n_digits_range: tuple,
                   scale_range, avoid_overlap: bool):
    ensure_dir(out_split_dir)
    for i in tqdm(range(n_images), desc=os.path.basename(out_split_dir)):
        n_digits = n_digits_range[0] if n_digits_range[0] == n_digits_range[1] else \
                   __import__("random").randint(n_digits_range[0], n_digits_range[1])

        img, objs = generate_one_scene(
            mnist_ds=mnist_ds,
            canvas_size=canvas_size,
            n_digits=n_digits,
            scale_range=scale_range,
            avoid_overlap=avoid_overlap,
            min_gap=2
        )
        save_sample(out_split_dir, i, img, objs)


def main():
    set_seed(42)

    base_dir = os.path.dirname(__file__)
    data_root = os.path.join(base_dir, "data")
    raw_root = os.path.join(data_root, "raw_mnist")

    mnist_train, mnist_test = mnist_source(raw_root)

    canvas_size = 128

    # tamanhos (podes ajustar se o PC não aguentar)
    n_train = 60000
    n_test = 10000

    versions = {
        # A: 1 dígito, sem scale
        "synth_vA": {"n_digits_range": (1, 1), "scale_range": None},
        # B: 1 dígito, com scale
        "synth_vB": {"n_digits_range": (1, 1), "scale_range": (22, 36)},
        # C: 3-5 dígitos, sem scale
        "synth_vC": {"n_digits_range": (3, 5), "scale_range": None},
        # D: 3-5 dígitos, com scale
        "synth_vD": {"n_digits_range": (3, 5), "scale_range": (22, 36)},
    }

    for name, cfg in versions.items():
        out_dir = os.path.join(data_root, name)
        train_dir = os.path.join(out_dir, "train")
        test_dir  = os.path.join(out_dir, "test")

        print(f"\n=== Generating {name} ===")
        generate_split(mnist_train, train_dir, n_train, canvas_size, cfg["n_digits_range"], cfg["scale_range"], True)
        generate_split(mnist_test,  test_dir,  n_test,  canvas_size, cfg["n_digits_range"], cfg["scale_range"], True)

    print("\nDONE.")


if __name__ == "__main__":
    main()
