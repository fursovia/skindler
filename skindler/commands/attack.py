from pathlib import Path
from datetime import datetime

from allenai_common import Params
import typer
import jsonlines
from datasets import load_dataset

from skindler.attackers import AttackerInput, AttackerOutput, Attacker
from skindler import SENTENCES_TO_ATTACK, DATASET_NAME, SENTENCES_TO_ATTACK

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
        source_lang = 'en'
        target_lang = 'ru'
        x = [ex[source_lang] for ex in data["translation"]]
        y = [ex[target_lang] for ex in data["translation"]]
        data = [(x_, y_) for (x_, y_) in zip(x, y)]
    except BaseException:
        data = load_jsonlines(data_path)[:samples]
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
