from pathlib import Path

import jsonlines
import typer
from allenai_common import Params
from datasets import load_dataset

from skindler import DATASET_NAME, SENTENCES_TO_ATTACK, SRC_LNG, TGT_LNG
from skindler.attackers import AttackerInput, AttackerOutput, Attacker

app = typer.Typer()


@app.command()
def attack(
        config_path: str,
        data_path: str = None,
        out_dir: str = None,
        samples: int = typer.Option(
            SENTENCES_TO_ATTACK,
            help="Number of samples")
):
    params = Params.from_file(config_path)
    attacker = Attacker.from_params(params["attacker"])
    typer.echo("loaded attack module")
    try:
        data = load_dataset(*DATASET_NAME)['test'][:samples]
        x = [ex[SRC_LNG] for ex in data["translation"]]
        y = [ex[TGT_LNG] for ex in data["translation"]]
        data = [(x_, y_) for (x_, y_) in zip(x, y)]
    except BaseException:
        data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                data.append(obj)
        data = data[:samples]
        x = [i['x'] for i in data]
        y = [i['y'] for i in data]
        data = [(x_, y_) for (x_, y_) in zip(x, y)]

    typer.echo("loaded data")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    params["out_dir"] = str(out_dir)
    config_path = out_dir / "config.json"
    params.to_file(str(config_path))
    output_path = out_dir / "data.json"

    typer.secho(f"Saving results to {output_path}", fg="green")
    with jsonlines.open(output_path, "w") as writer:
        for i, sample in enumerate(data):
            inputs = AttackerInput(*sample)

            try:
                adversarial_output = attacker.attack(inputs)
            except Exception as e:
                error_message = typer.style(
                    f">>> Failed to attack because {e}",
                    fg=typer.colors.RED,
                    bold=True)
                typer.echo(error_message)
                adversarial_output = AttackerOutput(
                    x=inputs.x,
                    y=inputs.y,
                    x_attacked=inputs.x,
                    y_trans=inputs.y,
                    y_trans_attacked=inputs.y
                )

            initial_text = getattr(adversarial_output, 'x')
            adv_text = getattr(adversarial_output, 'x_attacked')

            if str(initial_text) != adv_text:
                adv_text = typer.style(
                    adv_text, fg=typer.colors.GREEN, bold=True)
            else:
                adv_text = typer.style(
                    adv_text, fg=typer.colors.RED, bold=True)

            message = f"[{i} / {len(data)}] \n{initial_text}\n{adv_text}\n"
            typer.echo(message)
            writer.write(adversarial_output.to_dict())


if __name__ == "__main__":
    app()
