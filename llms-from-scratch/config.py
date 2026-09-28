class Config:
    def __init__(self) -> None:
        # 1tokenを表すベクトルの次元数（shapeの D）。パラメータ数を最も大きく左右する
        self.d_model: int = 128
        # 積むTransformerブロックの数。ブロックごとに別のパラメータを持つ
        self.n_layers: int = 4
        # Attentionのヘッド数。d_model をこの数で等分して各ヘッドに割り当てる
        self.n_heads: int = 4
        # 一度にモデルへ入れる系列の最大token数（shapeの T）
        self.context_length: int = 256
        # 1stepの更新で同時に学習する系列の数（shapeの B）
        self.batch_size: int = 16
        # パラメータを更新する回数
        self.steps: int = 5000
        # AdamWに渡す学習率
        self.learning_rate: float = 0.001
        # AdamWの重み減衰。更新のたびに全パラメータを少しだけ0に近づけ、一部のパラメータ
        # だけ極端に大きくなるのを防ぐ。本と同じ0.1
        self.weight_decay: float = 0.1
        # 何stepごとにvalidation lossを測るか
        self.eval_every: int = 200
        # 評価1回で使うバッチ数
        self.eval_batches: int = 8
        # 乱数の種
        self.seed: int = 42
        # 学習に使う計算装置。生成・評価はCPUで行う（1文字ずつの小さな計算はGPUの方が遅い）
        self.device: str = "mps"

    # 学習時の設定を config.json に保存・復元するため。読み込んだモデルは学習時と
    # 同じ設定で組み立てる必要がある
    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
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
        config.n_layers = data["n_layers"]
        config.n_heads = data["n_heads"]
        config.context_length = data["context_length"]
        config.batch_size = data["batch_size"]
        config.steps = data["steps"]
        config.learning_rate = data["learning_rate"]
        config.weight_decay = data["weight_decay"]
        config.eval_every = data["eval_every"]
        config.eval_batches = data["eval_batches"]
        config.seed = data["seed"]
        return config
