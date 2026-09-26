class Config:
    def __init__(self) -> None:
        # 1tokenを表すベクトルの次元数（shapeの D）。パラメータ数を最も大きく左右する
        self.d_model: int = 64
        # 一度にモデルへ入れる系列の最大token数（shapeの T）
        self.context_length: int = 256
        # 1stepの更新で同時に学習する系列の数（shapeの B）
        self.batch_size: int = 16
        # パラメータを更新する回数
        self.steps: int = 2000
        # AdamWに渡す学習率
        self.learning_rate: float = 0.0003
        # AdamWの重み減衰。更新のたびに全パラメータを少しだけ0に近づけ、一部のパラメータ
        # だけ極端に大きくなるのを防ぐ。本と同じ0.1
        self.weight_decay: float = 0.1
        # 何stepごとにvalidation lossを測るか
        self.eval_every: int = 200
        # 評価1回で使うバッチ数
        self.eval_batches: int = 8
        # 乱数の種
        self.seed: int = 42

    # 学習時の設定を config.json に保存・復元するため。読み込んだモデルは学習時と
    # 同じ設定で組み立てる必要がある
    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eval_every": self.eval_every,
            "eval_batches": self.eval_batches,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        config = cls()
        config.d_model = data["d_model"]
        config.context_length = data["context_length"]
        config.batch_size = data["batch_size"]
        config.steps = data["steps"]
        config.learning_rate = data["learning_rate"]
        config.weight_decay = data["weight_decay"]
        config.eval_every = data["eval_every"]
        config.eval_batches = data["eval_batches"]
        config.seed = data["seed"]
        return config
