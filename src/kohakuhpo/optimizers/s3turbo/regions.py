"""Trust-region state shared by S3-TuRBO's main and candidate regions."""


class Region:
    """One trust region: the ``main`` region on the incumbent, or a scouted ``candidate`` basin."""

    __slots__ = ("center", "radius", "kind", "best_y", "best_u", "visits", "succ", "fail", "warmup")

    def __init__(self, center, radius, kind, warmup=0):
        self.center = center.copy()
        self.radius = float(radius)
        self.kind = kind  # "main" | "candidate"
        self.best_y = float("inf")
        self.best_u = center.copy()
        self.visits = 0
        self.succ = 0
        self.fail = 0
        self.warmup = int(warmup)
