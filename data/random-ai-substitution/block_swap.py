import argparse
import math
import random

import numpy as np
from PIL import Image


def _center_weight(row: int, col: int, n: int) -> float:
    """Return a weight inversely proportional to distance from image center.

    Center blocks get weight ~1.0, edge blocks get weight ~0.0.
    """
    center = (n - 1) / 2.0
    max_dist = math.sqrt(center**2 + center**2) or 1.0
    dist = math.sqrt((row - center)**2 + (col - center)**2)
    return max(0.0, 1.0 - dist / max_dist)


def _select_blocks_scattered(total_blocks: int, n: int, m: int) -> list[int]:
    """Select m blocks uniformly at random."""
    return random.sample(range(total_blocks), m)


def _select_blocks_clustered(total_blocks: int, n: int, m: int,
                              num_clusters: int) -> list[int]:
    """Select m blocks grouped into num_clusters irregular clusters biased toward center."""
    if num_clusters > m:
        num_clusters = m

    selected = set()

    # Distribute blocks unevenly across clusters for organic feel
    base = m // num_clusters
    remainder = m % num_clusters
    cluster_sizes = [base + (1 if i < remainder else 0) for i in range(num_clusters)]
    # Shuffle so cluster sizes aren't uniform
    random.shuffle(cluster_sizes)

    # Compute center weights for all blocks
    weights = {}
    for idx in range(total_blocks):
        r, c = divmod(idx, n)
        weights[idx] = _center_weight(r, c, n) ** 2  # squared to exaggerate center bias

    for cluster_size in cluster_sizes:
        # Pick seed weighted toward center, excluding already-selected blocks
        available = [i for i in range(total_blocks) if i not in selected]
        avail_weights = [weights[i] for i in available]
        # Avoid all-zero weights (shouldn't happen but safety)
        if sum(avail_weights) == 0:
            seed = random.choice(available)
        else:
            seed = random.choices(available, weights=avail_weights, k=1)[0]

        sr, sc = divmod(seed, n)
        cluster = {(sr, sc)}
        # Build frontier: (row, col) of neighbors not yet in cluster
        frontier: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < n and 0 <= nc < n:
                frontier.append((nr, nc))

        while len(cluster) < cluster_size and frontier:
            # Randomly pick a frontier block to grow into (creates irregular shapes)
            pick_idx = random.randrange(len(frontier))
            r, c = frontier[pick_idx]
            # Remove from frontier
            frontier[pick_idx] = frontier[-1]
            frontier.pop()

            idx = r * n + c
            if idx in selected:
                continue

            # Probabilistic acceptance — adds jaggedness instead of smooth growth
            # Accept with probability proportional to how many neighbors
            # are already in the cluster (encourages cohesion) + some randomness
            neighbor_count = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) in cluster
            )
            accept_prob = 0.3 + 0.15 * neighbor_count
            if random.random() > accept_prob:
                # Put it back for later (gives chance for irregular gaps)
                frontier.append((r, c))
                continue

            cluster.add((r, c))

            # Expand frontier with new block's neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    nidx = nr * n + nc
                    if nidx not in selected and (nr, nc) not in cluster:
                        frontier.append((nr, nc))

        # Convert selected block coordinates back to indices
        selected.update(r * n + c for r, c in cluster)

    return list(selected)


def block_swap(img_path1: str, img_path2: str, n: int, m: int,
               seed: int | None = None,
               out_prefix: str | None = None,
               clustered: bool = False,
               num_clusters: int | None = None) -> tuple:
    """
    Segment two images into N×N blocks, randomly swap m blocks,
    and save the results with a swap mask.

    Args:
        img_path1: Path to the first input image.
        img_path2: Path to the second input image.
        n: Number of blocks along each dimension (N×N grid).
        m: Number of blocks to randomly swap.
        seed: Optional random seed for reproducibility.
        out_prefix: Prefix for output filenames.
        clustered: If True, selected blocks form spatial clusters.
        num_clusters: Number of clusters when clustered mode is on.
    """
    if seed is not None:
        random.seed(seed)

    size = 512
    img1 = Image.open(img_path1).resize((size, size)).convert("RGB")
    img2 = Image.open(img_path2).resize((size, size)).convert("RGB")

    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)

    block_h = size // n
    block_w = size // n

    total_blocks = n * n
    if m > total_blocks:
        raise ValueError(f"m ({m}) cannot exceed total blocks ({total_blocks})")

    if clustered:
        nc = num_clusters if num_clusters is not None else max(1, m // 3)
        selected = _select_blocks_clustered(total_blocks, n, m, nc)
    else:
        selected = _select_blocks_scattered(total_blocks, n, m)

    # Mask: 0 = unchanged, 255 = swapped
    mask = np.zeros((size, size), dtype=np.uint8)

    for idx in selected:
        row, col = divmod(idx, n)
        y1, y2 = row * block_h, (row + 1) * block_h
        x1, x2 = col * block_w, (col + 1) * block_w

        # Swap blocks between the two images
        arr1[y1:y2, x1:x2], arr2[y1:y2, x1:x2] = (
            arr2[y1:y2, x1:x2].copy(),
            arr1[y1:y2, x1:x2].copy(),
        )

        # Mark swapped region in mask
        mask[y1:y2, x1:x2] = 255

    if out_prefix is not None:
        Image.fromarray(arr1).save(f"{out_prefix}_swapped1.png")
        Image.fromarray(arr2).save(f"{out_prefix}_swapped2.png")
        Image.fromarray(mask).save(f"{out_prefix}_mask.png")
        print(f"Saved: {out_prefix}_swapped1.png, {out_prefix}_swapped2.png, {out_prefix}_mask.png")

    print(f"Swapped {m} of {total_blocks} blocks.")
    return arr1, arr2, mask


def main():
    parser = argparse.ArgumentParser(
        description="Randomly swap N×N sub-blocks between two images."
    )
    parser.add_argument("img1", help="Path to the first image")
    parser.add_argument("img2", help="Path to the second image")
    parser.add_argument("-n", type=int, default=4,
                        help="Grid size N (default: 4, produces 4×4=16 blocks)")
    parser.add_argument("-m", type=int, default=3,
                        help="Number of blocks to swap (default: 3)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("-c", "--clusters", type=int, default=None,
                        help="Enable clustered mode with specified number of clusters")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Output filename prefix (default: output)")
    args = parser.parse_args()

    block_swap(args.img1, args.img2, args.n, args.m, args.seed,
               args.output, clustered=args.clusters is not None,
               num_clusters=args.clusters)


if __name__ == "__main__":
    main()
