# GPU, vectorized objectives, and parallel evaluation

Three independent levers; combine as needed.

## 1. The optimizer's own math on GPU

```python
khpo.use_device("cuda:0")     # all GP work (Cholesky, L-BFGS, posterior draws) moves to the GPU
khpo.use_device("cpu")        # back to CPU float64
```

Default is CPU float64 (exact, stable). On CUDA the dtype switches to float32 (consumer GPUs
throttle float64); the GP's escalating-jitter Cholesky keeps float32 numerically safe at BO
design sizes. Pass `dtype=` to override. This pays off when the surrogate work dominates: large
candidate pools, large `max_data`, or many GP fits per ask (gpbo/hebo batching).

## 2. Vectorized objectives (batched evaluation)

If your objective is itself batched code (a simulator step, a network forward), take the whole
batch in one call:

```python
def f_batch(configs: list[dict]) -> np.ndarray:
    U = torch.tensor(space.to_units(configs), device="cuda")
    return simulate(U).cpu().numpy()

khpo.minimize(f_batch, space, "s3turbo", budget=400, q=32, vectorized=True)
```

## 3. Raw cube mode (no dicts anywhere)

For fully tensor-native workflows, drive the optimizer directly with arrays:

```python
opt = khpo.build("s3turbo", khpo.OPTIMIZER, space=khpo.SearchSpace.from_dim(25), seed=0)
for _ in range(50):
    U = opt.ask(32)                                   # np.ndarray (32, 25)
    y = simulate(torch.tensor(U, device="cuda"))      # your batched evaluation
    opt.tell(U, y.cpu().numpy())
```

## 4. Process-pool evaluation (independent expensive configs)

For objectives that are single-config and expensive (training runs, EDA jobs), spread a batch
across processes:

```python
khpo.minimize(train_and_eval, space, "s3turbo", budget=300, q=8, workers=8)
```

The pool uses the ``forkserver`` context (torch state does not survive a plain fork); the
objective must be picklable (a top-level function, not a lambda or closure).

A worked end-to-end case of all three levers: `examples/07_rect_collage.py` (pure torch, no extra deps) and
`examples/08_gs2d_image_match.py` (IGS's fused Triton renderer) both match a target image by
evaluating each batch of configs as one vectorized render call.
