import torch

from tiny_gpt.config import Config
from tiny_gpt.generate import generate
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import Tokenizer
from tiny_gpt.train import select_device


def main() -> None:
    config = Config()
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    tokenizer = Tokenizer()
    model = TinyGPT(tokenizer.vocab_size, config).to(device)
    parameter_count = 0
    for parameter in model.parameters():
        parameter_count += parameter.numel()

    print("学習前のGPT / device:", device, "/ parameter数:", parameter_count)
    print(generate(model, tokenizer, "プログラミングを学ぶには、", 80, device))


if __name__ == "__main__":
    main()
