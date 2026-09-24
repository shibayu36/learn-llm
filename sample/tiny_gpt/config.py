class Config:
    def __init__(self) -> None:
        # 1tokenを表すベクトルの次元数（shapeの D）。Embeddingから
        # Attention・MLP・LM Headまで、すべての層がこの幅でつながる
        self.d_model: int = 128
        # 一度にモデルへ入れる系列の最大token数（shapeの T）。
        # Attentionが参照できる範囲であり、生成時に見返せる長さでもある
        self.context_length: int = 128
        # 1stepの更新で同時に学習する系列の数（shapeの B）
        self.batch_size: int = 16
        # パラメータを更新する回数。
        # 1stepで batch_size × context_length 個のtokenから学習する
        self.steps: int = 10000
        # 1回の更新で、勾配の方向へパラメータを動かす幅。AdamWに渡す
        self.learning_rate: float = 0.0003
        # 何stepごとにvalidationデータのlossを測るか
        self.eval_every: int = 1000
        # 評価1回で使うバッチ数。複数バッチの平均を取り、lossのぶれを抑える
        self.eval_batches: int = 8
        # 乱数の種。初期化する重みと、学習に使うバッチの切り出し位置を
        # 実行ごとに同じにする
        self.seed: int = 42
        # CPUで計算するときに使うスレッド数
        self.cpu_threads: int = 4
        # 結果の保存先 runs/<run_name>。設定を変えて比較するときは名前も変える
        self.run_name: str = "part1"
