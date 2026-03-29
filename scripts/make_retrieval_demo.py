from pathlib import Path
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def person_id_from_name(name: str) -> str:
    return name.split('_')[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "my_dataset" / "images"
    out_path = root / "docs" / "images" / "retrieval_demo.png"

    files = sorted([p for p in image_dir.glob("*.jpg")])
    if len(files) < 10:
        raise RuntimeError("Not enough images in my_dataset/images to build a demo figure.")

    grouped = defaultdict(list)
    for file in files:
        grouped[person_id_from_name(file.name)].append(file)

    valid_ids = [pid for pid, imgs in grouped.items() if len(imgs) >= 3]
    if not valid_ids:
        raise RuntimeError("No person ID has at least 3 images.")

    random.seed(42)
    query_pid = valid_ids[0]
    same_person = grouped[query_pid]

    query_img = same_person[0]
    gt_img = same_person[1]

    other_ids = [pid for pid in grouped.keys() if pid != query_pid and len(grouped[pid]) > 0]
    random.shuffle(other_ids)
    wrong_imgs = [grouped[pid][0] for pid in other_ids[:4]]
    if len(wrong_imgs) < 4:
        raise RuntimeError("Not enough negative samples to construct Top-K examples.")

    topk = [wrong_imgs[0], gt_img, wrong_imgs[1], wrong_imgs[2], wrong_imgs[3]]

    fig = plt.figure(figsize=(17, 4.8), dpi=180)
    gs = fig.add_gridspec(1, 7, width_ratios=[1.05, 1.05, 1, 1, 1, 1, 1], wspace=0.08)

    panels = [
        ("Query", query_img, "#4C78A8"),
        ("Ground Truth", gt_img, "#2CA02C"),
    ]

    for idx, (title, path, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_edgecolor(color)

    for i, img_path in enumerate(topk):
        ax = fig.add_subplot(gs[0, i + 2])
        ax.imshow(mpimg.imread(img_path))
        is_hit = person_id_from_name(img_path.name) == query_pid
        color = "#2CA02C" if is_hit else "#D62728"
        label = f"Top-{i + 1}" + (" ✓" if is_hit else " ✗")
        ax.set_title(label, fontsize=10, pad=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_edgecolor(color)

    fig.suptitle(
        "Query / Ground Truth / Top-K Retrieval Example",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        -0.02,
        "Green border: correct identity match  |  Red border: incorrect retrieval",
        ha="center",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
