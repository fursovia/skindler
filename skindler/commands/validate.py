from pathlib import Path
import json

from typer import Typer
import pandas as pd

from skindler.modules.metrics import ALL_METRICS
from skindler.utils import count_metrics

app = Typer()


@app.command()
def validate(results_path: Path):

    df = []
    with results_path.open() as f:
        for line in f.readlines():
            df.append(json.loads(line.strip()))
    df = pd.DataFrame(df)

    x = df['x'].tolist()
    y = df['y'].tolist()
    x_attacked = [i.replace("[[", "").replace("]]", "")
                  for i in df['x_attacked'].tolist()]
    y_trans = df['y_trans'].tolist()
    y_trans_attacked = df['y_trans_attacked'].tolist()

    orig_translate_metrics = count_metrics(y, y_trans)
    attack_translate_metrics = count_metrics(y, y_trans_attacked)
    translation_diff_metrics = count_metrics(y_trans, y_trans_attacked)
    x_metrics = count_metrics(x, x_attacked)

    for metric_name, metric_value in orig_translate_metrics.items():
        df[f"orig.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in attack_translate_metrics.items():
        df[f"attacked.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in translation_diff_metrics.items():
        df[f"diff_translation_{metric_name}"] = metric_value

    for metric_name, metric_value in x_metrics.items():
        df[f"diff_x_{metric_name}"] = metric_value

    save_to = results_path.parent / f"{results_path.stem}_metrics.csv"
    df.to_csv(save_to, index=False)


if __name__ == "__main__":
    app()
