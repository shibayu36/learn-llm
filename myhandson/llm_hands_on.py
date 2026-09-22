import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
print("実行環境:", torch.__version__, "CPU")


# ── 部品: Tokenizer とデータ ──────────────────────────────────

class Tokenizer:
    """文字とtoken IDを相互に変換する。この教材では1文字＝1token。"""

    def __init__(self, chars: list[str]):
        self.chars = chars     # ID順に並んだ文字。例: ['。', 'え', 'は', '合', '図', '答', '赤', '青']
        # 文字 → ID と ID → 文字 の対応表を、0番から順に作る
        self.stoi = {}
        self.itos = {}
        for i in range(len(chars)):
            ch = chars[i]
            self.stoi[ch] = i
            self.itos[i] = ch

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Tokenizer":
        """全文章をつなげ、重複を除いて、文字コード順に並べた語彙を作る"""
        return cls(sorted(set("".join(texts))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """文字列 → ID列。 例: "合図は" → [3, 4, 2]"""
        ids = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"語彙にない文字: {ch}")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        """ID列 → 文字列。 例: [3, 4, 2] → "合図は" """
        text = ""
        for i in ids:
            text += self.itos[i]
        return text


def make_dataset(texts: list[str], tok: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """文章のリストから、学習用の入力xと正解yを作る。"""
    # 各文章をID列にしてから、まとめてTensorにする
    encoded = []
    for s in texts:
        encoded.append(tok.encode(s)) # encoded[0] = [3, 4, 2, 6, 0, 5, 1, 2, 6, 0]
    data = torch.tensor(encoded, dtype=torch.long) # shape [2, 10]

    # 解きたいのは「ここまでの文字列を見て、次の1文字を当てる」問題。1文から次の9問が作れる。
    #   合                → 図
    #   合図              → は
    #   合図は            → 赤
    #   ...
    #   合図は赤。答えは  → 赤   ← 前半の色を参照しないと解けない本命の問題
    #   合図は赤。答えは赤 → 。
    # この9問を1つずつ作る代わりに、文全体を1文字ずらして並べる。
    #   x = 合 図 は 赤 。 答 え は 赤     （末尾1文字を落とす）
    #   y = 図 は 赤 。 答 え は 赤 。     （先頭1文字を落とす）
    # 位置iの問題は「x[0..i]を見てy[i]を当てる」。モデルは位置iで先の文字を見ないので、
    # 1回モデルに通すだけで9問がまとめて解ける。
    # yは9文字の文章ではなく、9個の「次の1文字」が並んだもの。
    x = data[:, :-1]
    y = data[:, 1:]
    return x, y


# ── 部品: モデル ──────────────────────────────────────────────

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # 8行×8列の学習可能な表。行＝直前の文字のID、列＝次の文字候補のスコア。
        # この表の数値が、学習で更新される「パラメータ」のすべて。
        # 学習後の中身はこうなる（値はlogit、3行を抜粋）:
        #          。    え    は    合    図    答    赤    青
        #   合   -1.9  -1.9  -2.5  -1.8  +2.9  -3.1  -3.4  -2.3   ← 「合」の次は「図」だけ高い
        #   答   -4.2  +3.6  -3.5  -3.2  -3.9  -0.5  -3.8  -3.1   ← 「答」の次は「え」
        #   は   -0.9  -2.7  -3.0  -2.1  -3.3  -1.4  +3.5  +3.5   ← 赤と青が同点。区別する情報がない
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # IDに対応する行を取り出すだけ。各位置を独立に引くので、他の位置の文字は一切見ていない。
        #   ids [2, 9] の各IDが表の1行（8個のスコア）に置き換わり、[2, 9, 8] になる。
        #   例: ids[0] = 合 図 は 赤 … → 行3, 行4, 行2, 行6, … を並べた9行
        return self.table(ids)  # [B, T, 語彙数]


# ── 部品: 推論 ────────────────────────────────────────────────

@torch.no_grad() # 観察するだけで、パラメータは更新しない
def next_probs(model, tok: Tokenizer, prompt: str, temperature=1.0) -> torch.Tensor:
    """promptの次に来る文字の確率分布を返す。"""
    if not prompt or temperature <= 0:
        raise ValueError("promptとtemperatureを指定してください")
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    # モデルは全位置ぶんのスコアを返すが、使うのは最後の位置だけ。続きを生成したいのはそこだから
    logits = model(ids)[0, -1]   # 8候補のスコア
    # スコアをtemperatureで割ってから、softmaxで「非負で合計1」の確率に変換する。
    #   「は」の行: logit   青 +3.49  赤 +3.48  。 -0.87  答 -1.43  …
    #            → exp      32.9     32.4     0.42     0.24    …  合計 66.3
    #            → ÷合計    0.497    0.489    0.006    0.004   …  ← 実行結果の値
    #   青と赤のlogit差は0.016だけ。青と「。」の差4.4は、expを通すと約80倍の差になる
    return F.softmax(logits / temperature, dim=-1)


def show_next(model, tok: Tokenizer, prompt: str, temperature=1.0):
    """次の文字の確率を、高い順に並べて表示する"""
    p = next_probs(model, tok, prompt, temperature)
    ranked = sorted(tok.chars, key=lambda ch: p[tok.stoi[ch]].item(), reverse=True)
    table = {}
    for ch in ranked:
        table[ch] = round(p[tok.stoi[ch]].item(), 4)
    print(prompt, table)


@torch.no_grad()
def generate(model, tok: Tokenizer, prompt: str, new_tokens=2, temperature=1.0) -> str:
    """promptの続きをnew_tokens文字ぶん生成し、prompt込みの文字列を返す。

    temperature=None のときは確率が最大の文字を選ぶ（greedy）。
    """
    if not prompt or new_tokens < 0 or (temperature is not None and temperature <= 0):
        raise ValueError("引数の範囲を確認する")
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(new_tokens):
        logits = model(ids)[:, -1, :] # 最後の位置のスコアだけ使う
        if temperature is None:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            # 確率に従って1文字選ぶ(sampling)。同じ入力でも呼ぶたびに結果が変わりうるのはここが原因
            # 確率に従ってサイコロ振っているイメージ
            next_id = torch.multinomial(probs, num_samples=1)
        # 選んだ文字を入力の末尾につなげ、次の周回ではその伸びた入力をモデルに渡す。
        # 生成した出力は後続の予測の条件にもなる
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())


# ── 部品: 学習 ────────────────────────────────────────────────

@torch.no_grad()
def loss_of(model, x, y) -> float:
    """全位置の予測と正解yのずれを、loss（誤差）という1つの数値にして返す。"""
    model.eval()
    logits = model(x)   # [2, 9, 8]。全位置の予測をまとめて計算する
    # cross_entropyは位置ごとに「正解の文字に与えた確率p」を見て、-log(p)を平均する。
    # 正解に0.9を与えれば0.105、0.1なら2.30。正解の確率が低いほど大きくなる。
    # 「予測1つ × 正解1つ」の組を並べた形で受け取るので、2文章×9位置 = 18組に平らにして渡す
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()


def fit_small(model, x, y, steps=800, lr=0.01):
    """lossが小さくなる向きにパラメータを少し動かす、をsteps回繰り返す。"""
    model.train()
    # optimizerは「勾配をもとにパラメータをどれだけ動かすか」を決める部品
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)   # 前回の勾配を消す
        loss.backward()                         # 各パラメータを動かすとlossがどう変わるか（勾配）を求める
        optimizer.step()                        # lossが減る向きにパラメータを少し動かす
    model.eval()


# ── 実験 ──────────────────────────────────────────────────────

def main():
    # 学習データ。どちらも末尾が「答えは」で、次に来る文字は前半の色で決まる。
    texts = ["合図は赤。答えは赤。", "合図は青。答えは青。"]
    tok = Tokenizer.from_texts(texts)   # 語彙は8文字
    x, y = make_dataset(texts, tok)
    print(tok.stoi)
    print("入力:", tok.decode(x[0].tolist()))
    print("正解:", tok.decode(y[0].tolist()))
    print("x / y:", tuple(x.shape), tuple(y.shape))

    torch.manual_seed(42)
    bigram = BigramLM(tok.vocab_size) # 表の中身はまだランダム。学習で意味のある値になる
    # 出力は [2文章, 9位置, 8候補]。位置ごとに「ここまでの文字列を見て、次の1文字を当てる」問題の答えが並んでいる。
    print("logits:", tuple(bigram(x).shape))

    # 学習前の表で予測する
    red_prompt = "合図は赤。答えは"
    blue_prompt = "合図は青。答えは"
    show_next(bigram, tok, red_prompt)
    show_next(bigram, tok, blue_prompt)
    # 前半が赤でも青でも分布が完全に一致する。最後の「は」の行しか見ていないため
    assert torch.equal(next_probs(bigram, tok, red_prompt), next_probs(bigram, tok, blue_prompt))

    # 学習する
    print("学習前のloss:", loss_of(bigram, x, y))
    fit_small(bigram, x, y)
    print("学習後のloss:", loss_of(bigram, x, y))
    # 学習で変わったのは8×8の表の値だけ。forwardは同じなので、前半を見ないことも変わらない
    show_next(bigram, tok, red_prompt)
    show_next(bigram, tok, blue_prompt)

    # 学習後の表で生成する
    torch.manual_seed(10)
    for _ in range(3):
        print(generate(bigram, tok, red_prompt))
    # 同じ入力でtemperatureだけを変える。変わるのは選択に使う分布であって、学習済みのパラメータではない。
    #   logit÷T で差が縮む/広がるだけなので、順位は変わらない:
    #   T=0.5  青 0.508  赤 0.492  。 0.0001 …   青と「。」のlogit差 4.4→8.7。「。」はほぼ0
    #   T=2.0  青 0.422  赤 0.419  。 0.048  …   差 4.4→2.2。「。」にも5%配られる
    for temperature in (0.5, 1.0, 2.0):
        print("temperature", temperature)
        show_next(bigram, tok, red_prompt, temperature)


if __name__ == "__main__":
    main()
