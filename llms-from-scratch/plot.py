from pathlib import Path

from matplotlib.figure import Figure


def save_loss_plot(training: dict, path: Path) -> None:
    """学習時に記録した先頭バッチのtrain・validation lossを曲線にする。"""
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    axis.plot(training["steps"], training["train_losses"], label="train", color="#397a9b")
    axis.plot(
        training["steps"], training["validation_losses"], label="validation", color="#d27832"
    )
    axis.set_xlabel("Training steps")
    axis.set_ylabel("Cross-entropy loss (lower is better)")
    config = training["config"]
    axis.set_title(
        f"{training['run_name']}\n"
        f"First {config['eval_batches']} batches | lr={config['learning_rate']:g}"
    )
    axis.legend()
    axis.grid(alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.savefig(path, dpi=160)
