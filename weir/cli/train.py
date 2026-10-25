from __future__ import annotations

import hydra
from omegaconf import DictConfig

from weir.cli.utils import setup_logging
from weir.core.application.train import run
from weir.core.utils import CONFIG_DIR


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
