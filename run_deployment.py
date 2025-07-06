import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
@click.option(
    "--skip-train",
    is_flag=True,
    default=False,
    help="Skip training and only run inference",
)
def run_main(skip_train: bool):
    """Run the deployment pipeline with manual MLflow serving."""

    if not skip_train:
        print("[bold cyan]Running training pipeline...")
        continuous_deployment_pipeline()

        print(
            "\n[bold green]Now manually run this in a separate terminal:[/bold green]\n"
            "mlflow models serve -m 'runs:/<your_run_id>/model' -p 1234 --no-conda\n"
            "[yellow]Replace <your_run_id> with the actual run ID from MLflow UI[/yellow]"
        )

    print("\n[bold cyan]Running inference pipeline...[/bold cyan]")
    inference_pipeline()

    print(
        "\n[bold green]You can inspect your experiments with:[/bold green]\n"
        f"mlflow ui --backend-store-uri {get_tracking_uri()}"
    )


if __name__ == "__main__":
    run_main()