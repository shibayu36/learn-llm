class Config:
    def __init__(self) -> None:
        self.d_model: int = 128
        self.context_length: int = 128
        self.batch_size: int = 16
        self.steps: int = 10000
        self.learning_rate: float = 0.0003
        self.eval_every: int = 1000
        self.eval_batches: int = 8
        self.seed: int = 42
        self.cpu_threads: int = 4
        self.device: str = "auto"
        self.run_name: str = "part1"
