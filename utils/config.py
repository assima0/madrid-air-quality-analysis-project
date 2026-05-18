from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILES_DIR = BASE_DIR / "files"
SAMPLE_PATH = BASE_DIR / "sample.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_task_dirs(task_name: str):
    task_dir = OUTPUT_DIR / task_name
    tables_dir = task_dir / "tables"
    figures_dir = task_dir / "figures"
    graphs_dir = task_dir / "graphs"
    models_dir = task_dir / "models"

    for directory in [tables_dir, figures_dir, graphs_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return tables_dir, figures_dir, graphs_dir, models_dir