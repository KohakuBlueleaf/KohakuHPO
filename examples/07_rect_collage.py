"""Rectangle collage: replicate an image with a fixed number of solid rectangles.

A fully discrete cousin of gaussian splatting: one configuration is ALL k rectangles at once
(7 numbers each: x, y, w, h, r, g, b), painted fully opaque in index order onto a white canvas,
and the objective is plain MSE against the target. The optimizer adjusts the whole set jointly;
index order is the z-order, so layering and occlusion are part of the search space.

Hard edges and full opacity make the loss piecewise constant in the geometry parameters, so
gradients are zero almost everywhere and gradient descent is useless here; this is exactly the
landscape black-box optimization exists for. The renderer is ~20 lines of pure torch, and a
whole batch of q configs renders in one vectorized call (GPU friendly, no extra dependency).

The target is either your own image (``--image photo.jpg``) or, with no image given, a hidden
random rectangle config (so a near-zero MSE is achievable). Optimization always runs on a square
``--size x --size`` resize in a fixed ``[-1, 1]^2`` frame; the final result is re-rendered by
stretching that frame over the ORIGINAL resolution and aspect ratio (or a 4x upscale in
hidden-target mode).

Usage:
    python examples/07_rect_collage.py --rects 12 --budget 1200 --q 16 --size 64
    python examples/07_rect_collage.py --image photo.jpg --rects 24 --budget 2400
"""

import argparse

import numpy as np
import torch

import kohakuhpo as khpo

PARAM_KINDS = ("x", "y", "w", "h", "r", "g", "b")


def make_space(n_rects: int) -> khpo.SearchSpace:
    """One SearchSpace entry per rectangle parameter, with its natural range."""
    params = {}
    for i in range(n_rects):
        params[f"rect{i}.x"] = ("float", -1.0, 1.0)
        params[f"rect{i}.y"] = ("float", -1.0, 1.0)
        params[f"rect{i}.w"] = ("log", 0.05, 2.0)
        params[f"rect{i}.h"] = ("log", 0.05, 2.0)
        params[f"rect{i}.r"] = ("float", 0.0, 1.0)
        params[f"rect{i}.g"] = ("float", 0.0, 1.0)
        params[f"rect{i}.b"] = ("float", 0.0, 1.0)
    return khpo.SearchSpace(params)


class Renderer:
    """Batched solid-rectangle renderer: opaque painter's-order compositing on a white canvas.

    Coordinates live in a fixed ``[-1, 1]^2`` frame at any pixel count, so the same rectangles
    render at any resolution or aspect ratio (the final stretch render reuses this).
    """

    def __init__(self, n_rects: int, device: str) -> None:
        self.n = n_rects
        self.device = device
        self.keys = [f"rect{i}.{k}" for i in range(n_rects) for k in PARAM_KINDS]
        self._grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (h, w) not in self._grids:
            xg = torch.linspace(-1.0, 1.0, w, device=self.device).view(1, 1, w)
            yg = torch.linspace(-1.0, 1.0, h, device=self.device).view(1, h, 1)
            self._grids[(h, w)] = (xg, yg)
        return self._grids[(h, w)]

    def pack(self, configs: list[dict]) -> torch.Tensor:
        """Config dicts -> one ``[B, N, 7]`` tensor in a single host-to-device copy."""
        arr = np.asarray([[cfg[k] for k in self.keys] for cfg in configs], dtype=np.float32)
        return torch.from_numpy(arr.reshape(len(configs), self.n, 7)).to(self.device)

    @torch.no_grad()
    def __call__(self, configs: list[dict], h: int, w: int) -> torch.Tensor:
        t = self.pack(configs)
        xg, yg = self._grid(h, w)
        x, y, rw, rh, color = t[..., 0], t[..., 1], t[..., 2], t[..., 3], t[..., 4:7]
        # inside[b, n, h, w]: hard pixel-in-box test (the "solid" in solid rectangle)
        inside = (
            (xg[:, None] >= (x - rw / 2)[..., None, None])
            & (xg[:, None] <= (x + rw / 2)[..., None, None])
            & (yg[:, None] >= (y - rh / 2)[..., None, None])
            & (yg[:, None] <= (y + rh / 2)[..., None, None])
        ).float()
        canvas = torch.ones(len(configs), 3, h, w, device=self.device)  # white canvas
        for i in range(self.n):  # index = z-order; each rect fully replaces what it covers
            m = inside[:, i][:, None]
            canvas = canvas * (1.0 - m) + color[:, i, :, None, None] * m
        return canvas


def load_image(path: str, size: int, max_final_dim: int, device):
    """Load an image; return (square target [1,3,size,size], original-size target [1,3,H0,W0]).

    The original is capped to ``max_final_dim`` on its longer side (aspect preserved) to keep the
    final full-resolution render cheap.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    if max(w0, h0) > max_final_dim:
        scale = max_final_dim / max(w0, h0)
        w0, h0 = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    to_tensor = lambda im: (  # noqa: E731
        torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    )
    square = to_tensor(img.resize((size, size), Image.LANCZOS))
    original = to_tensor(img.resize((w0, h0), Image.LANCZOS))
    return square, original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="target image; omit for a hidden rect target")
    ap.add_argument("--size", type=int, default=64, help="square resolution used during search")
    ap.add_argument("--max-final-dim", type=int, default=1024, help="cap for the final render")
    ap.add_argument("--rects", type=int, default=12)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="heterogeneous")
    ap.add_argument("--save", default="rect_collage.png", help="side-by-side output image")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        khpo.use_device(device)  # GP surrogate math on the GPU as well
    render = Renderer(args.rects, device)
    print(f"device={device}, {args.rects} opaque rectangles adjusted jointly -> d={args.rects * 7}")

    space = make_space(args.rects)

    if args.image:
        target, target_full = load_image(args.image, args.size, args.max_final_dim, device)
        final_h, final_w = target_full.shape[2], target_full.shape[3]
        print(f"target: {args.image} (search at {args.size}^2, final {final_w}x{final_h})")
    else:
        hidden_rng = np.random.default_rng(args.seed + 777)
        hidden = space.to_config(hidden_rng.random(space.dim))
        target = render([hidden], args.size, args.size)
        target_full = None
        final_h = final_w = args.size * 4
        print(f"target: hidden rect config (search at {args.size}^2, final {final_w}x{final_h})")

    def objective(configs: list[dict]) -> np.ndarray:
        imgs = render(configs, args.size, args.size)
        return ((imgs - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

    study = khpo.Study(
        space,
        {"name": "s3turbo", "preset": args.preset, "budget": args.budget},
        seed=args.seed,
    )
    for batch in study.loop(budget=args.budget, q=args.q, progress=True, desc="rect collage"):
        batch.report(objective(batch.configs))

    print(f"final best MSE = {study.best_value:.6f}")

    # Final render: the same [-1,1]^2 frame stretched over the original resolution/aspect.
    best_full = render([study.best_config], final_h, final_w)
    if target_full is None:
        target_full = render([hidden], final_h, final_w)
    side_by_side = torch.cat([target_full[0], best_full[0]], dim=2).cpu()
    try:
        from torchvision.utils import save_image

        save_image(side_by_side, args.save)
        print(f"[wrote {args.save}]  (left: target, right: best found, {final_w}x{final_h} each)")
    except ImportError:
        np.save(args.save.replace(".png", ".npy"), side_by_side.numpy())
        print(f"[torchvision not installed; wrote {args.save.replace('.png', '.npy')}]")


if __name__ == "__main__":
    main()
