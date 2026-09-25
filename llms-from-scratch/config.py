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
        # 何stepごとにvalidation lossを測るか
        self.eval_every: int = 200
        # 評価1回で使うバッチ数
        self.eval_batches: int = 8
        # 乱数の種
        self.seed: int = 42
