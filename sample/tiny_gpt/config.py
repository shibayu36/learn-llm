class Config:
    def __init__(self) -> None:
        # 1tokenを表すベクトルの次元数（shapeの D）。1tokenの性質を
        # D個の数で表すので、Dが大きいほど1tokenに持たせられる性質の数が増える
        self.d_model: int = 128
        # 1系列に入れるtoken数の上限。shapeの T はこの値以下。
        # Attentionが参照できる範囲であり、生成時に見返せる長さでもある
        self.context_length: int = 128
        # 1回の更新に使う系列の数（shapeの B）。
        # B本ぶんのlossを平均して、パラメータを1回だけ更新する
        self.batch_size: int = 16
        # パラメータを更新する回数。
        # 1stepで batch_size × context_length 個のtokenから学習する
        self.steps: int = 10000
        # 1回の更新でパラメータを動かす幅。lossが下がる向きへ、
        # この幅に応じて動かす（1.11で扱う）。AdamWに渡す
        self.learning_rate: float = 0.0003
        # 何stepごとにvalidationのlossを測るか
        self.eval_every: int = 1000
        # 評価1回で使うバッチ数。複数バッチの平均を取り、lossのぶれを抑える
        self.eval_batches: int = 8
        # 乱数の種。初期化するパラメータと、学習に使うバッチの切り出し位置を
        # 実行ごとに同じにする
        self.seed: int = 42
        # 結果の保存先 runs/<run_name>。設定を変えて比較するときは名前も変える
        self.run_name: str = "part1"
