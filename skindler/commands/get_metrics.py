import json
from pathlib import Path

import pandas as pd
from typer import Typer

from skindler.modules.utils import AttackOutput, load_attacks, count_metrics
app = Typer()


@app.command()
def main(
        attack_path: Path = Path('experiment/threshold_0.75/attack_output')
):
    attacks = load_attacks(attack_path)
    save_metrics_path = Path(attack_path).parent / "metrics.csv"
    x = [i.x for i in attacks]
    y = [i.y for i in attacks]
    x_attacked = [i.x_attacked for i in attacks]
    y_trans = [i.y_trans for i in attacks]
    y_trans_attacked = [i.y_trans_attacked for i in attacks]

    # count metrics

    orig_translate_metrics = count_metrics(y, y_trans)
    attack_translate_metrics = count_metrics(y, y_trans_attacked)
    x_metrics = count_metrics(x, x_attacked)

    table = {'x': x, 'y': y,
             'x_attacked': x_attacked, 'y_trans': y_trans, 'y_trans_attacked': y_trans_attacked}

    for metric_name, metric_value in orig_translate_metrics.items():
        table[f"orig.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in attack_translate_metrics.items():
        table[f"attacked.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in x_metrics.items():
        table[f"x_{metric_name}"] = metric_value

    table = pd.DataFrame(table)
    table.to_csv(str(save_metrics_path))


if __name__ == "__main__":
    app()
