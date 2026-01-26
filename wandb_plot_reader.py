"""
W&B Plot Reader - Fetch data from Weights & Biases and create matplotlib plots.
"""

import wandb
import pandas as pd
from typing import Optional, List, Union, Dict, Any



class WandbPlotReader:
    """Reader class to fetch W&B run data and create matplotlib plots."""

    def __init__(self, entity: Optional[str] = None, project: Optional[str] = None):
        """
        Initialize the W&B Plot Reader.

        Args:
            entity: W&B entity (username or team name). If None, uses default.
            project: W&B project name. If None, must be specified in methods.
        """
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def get_runs(
        self,
        project: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "-created_at",
    ) -> List[wandb.apis.public.Run]:
        """
        Fetch runs from a W&B project.

        Args:
            project: Project name (uses default if not specified).
            filters: MongoDB-style filters for runs.
            order: Sort order for runs.

        Returns:
            List of W&B Run objects.
        """
        project = project or self.project
        if not project:
            raise ValueError("Project must be specified")

        path = f"{self.entity}/{project}" if self.entity else project
        return list(self.api.runs(path, filters=filters, order=order))

    def get_run(
        self,
        run_id: str,
        project: Optional[str] = None,
    ) -> wandb.apis.public.Run:
        """
        Get a specific run by ID.

        Args:
            run_id: Run ID string.
            project: Project name (uses default if not specified).

        Returns:
            W&B Run object.
        """
        project = project or self.project
        if not project:
            raise ValueError("Project must be specified")

        path = f"{self.entity}/{project}/{run_id}" if self.entity else f"{project}/{run_id}"
        return self.api.run(path)

    def get_run_history(
        self,
        run: Union[str, wandb.apis.public.Run],
        keys: Optional[List[str]] = None,
        samples: int = 10000,
        project: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get history DataFrame for a specific run.

        Args:
            run: Run ID string or Run object.
            keys: Specific keys to fetch. If None, fetches all.
            samples: Number of samples to fetch.
            project: Project name (required if run is a string).

        Returns:
            DataFrame with run history.
        """
        if isinstance(run, str):
            run = self.get_run(run, project=project)

        history = run.history(samples=samples, keys=keys)
        return history

    def get_run_summary(
        self, run: Union[str, wandb.apis.public.Run], project: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary metrics for a run.

        Args:
            run: Run ID string or Run object.
            project: Project name (required if run is a string).

        Returns:
            Dictionary of summary metrics.
        """
        if isinstance(run, str):
            run = self.get_run(run, project=project)

        return dict(run.summary)

