from pathlib import Path
import json

from typer import Typer
import pandas as pd

from skindler.modules.metrics import ALL_METRICS


MODELS_FOLDER = (Path(__file__).parent / ".." / "..").resolve() / "models"


app = Typer()


@app.command()
def valiate(results_path: Path):

    df = []
    with results_path.open() as f:
        for line in f.readlines():
            df.append(json.loads(line.strip()))
    df = pd.DataFrame(df)

    en = df["en"].tolist()
    ru_trans = df["ru_trans"].tolist()
    en_attacked = df["en_attacked"].tolist()
    ru_trans_attacked = df["ru_trans_attacked"].tolist()

    for metric_name, metric in ALL_METRICS.items():
        df[f"{metric_name}_en"] = metric(en, en_attacked)
        df[f"{metric_name}_ru"] = metric(ru_trans, ru_trans_attacked)

    save_to = results_path.parent / f"{results_path.stem}_metrics.csv"
    df.to_csv(save_to, index=False)


if __name__ == "__main__":
    app()
