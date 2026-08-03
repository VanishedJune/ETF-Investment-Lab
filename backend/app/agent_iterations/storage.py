from __future__ import annotations

from datetime import date
from pathlib import Path
import json
from typing import Any


class ArtifactConsistencyError(RuntimeError):
    """Raised when manifest and iteration artifacts do not describe one snapshot."""


class AgentIterationStorage:
    """Atomic, project-local storage for agent-produced model artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _directory(self, instrument_code: str) -> Path:
        if instrument_code not in {"399006", "NDX"}:
            raise ValueError(f"Unsupported iteration instrument: {instrument_code}")
        return self.root / instrument_code

    def read_manifest(self, instrument_code: str) -> dict[str, Any] | None:
        path = self._directory(instrument_code) / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArtifactConsistencyError("manifest.json contains invalid JSON") from error

    def read_iterations(self, instrument_code: str) -> list[dict[str, Any]]:
        path = self._directory(instrument_code) / "iterations.jsonl"
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ArtifactConsistencyError(
                    f"iterations.jsonl line {line_number} contains invalid JSON"
                ) from error
        return records

    @staticmethod
    def _artifact_date(value: Any, *, field: str, iteration: int) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ArtifactConsistencyError(
                f"iteration {iteration} has invalid {field}={value!r}"
            ) from error

    @staticmethod
    def _artifact_integer(value: Any, *, field: str, iteration: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactConsistencyError(
                f"iteration {iteration} {field} must be an integer; got {value!r}"
            )
        return value

    @staticmethod
    def _validate_hash_chain(
        iterations: list[dict[str, Any]],
        *,
        value_field: str,
        parent_field: str,
    ) -> None:
        if not any(
            value_field in record or parent_field in record
            for record in iterations
        ):
            return
        previous_hash: Any = None
        for sequence, record in enumerate(iterations, start=1):
            value = record.get(value_field)
            parent = record.get(parent_field)
            if not isinstance(value, str) or not value:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} has invalid {value_field}={value!r}"
                )
            if parent != previous_hash:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} {parent_field}={parent!r} does not "
                    f"match previous {value_field}={previous_hash!r}"
                )
            previous_hash = value

    @staticmethod
    def _validate_snapshot(
        manifest: dict[str, Any] | None,
        iterations: list[dict[str, Any]],
        *,
        expected_instrument_code: str,
    ) -> None:
        if manifest is None:
            if iterations:
                raise ArtifactConsistencyError("Iteration records exist without a manifest")
            return
        if manifest.get("instrument_code") != expected_instrument_code:
            raise ArtifactConsistencyError(
                f"manifest.instrument_code={manifest.get('instrument_code')!r} does not "
                f"match artifact directory instrument_code={expected_instrument_code!r}"
            )
        completed_value = manifest.get("completed_iterations", 0)
        if isinstance(completed_value, bool):
            raise ArtifactConsistencyError(
                f"manifest.completed_iterations={completed_value!r} is not an integer"
            )
        try:
            completed = int(completed_value)
        except (TypeError, ValueError) as error:
            raise ArtifactConsistencyError(
                f"manifest.completed_iterations={completed_value!r} is not an integer"
            ) from error
        if completed != len(iterations):
            raise ArtifactConsistencyError(
                f"manifest.completed_iterations={completed} does not match curve length={len(iterations)}"
            )
        latest_version = manifest.get("latest_model_version")
        expected_version = f"M{completed:03d}" if completed else None
        if latest_version != expected_version:
            raise ArtifactConsistencyError(
                f"manifest.latest_model_version={latest_version} does not match "
                f"completed_iterations={completed} ({expected_version})"
            )
        if not iterations:
            return
        last_model = iterations[-1].get("model")
        last_version = (
            last_model.get("version")
            if isinstance(last_model, dict)
            else None
        )
        if last_version is None:
            raise ArtifactConsistencyError("Last iteration record has no model version")
        if latest_version != last_version:
            raise ArtifactConsistencyError(
                f"manifest.latest_model_version={latest_version} does not match last model={last_version}"
            )
        latest_model = manifest.get("latest_model")
        if not isinstance(latest_model, dict):
            raise ArtifactConsistencyError(
                "manifest.latest_model is missing or is not an object"
            )
        if latest_model != last_model:
            raise ArtifactConsistencyError(
                "manifest.latest_model does not exactly match the last iteration model"
            )

        previous_cutoff: date | None = None
        previous_future_label: date | None = None
        for sequence, record in enumerate(iterations, start=1):
            if record.get("iteration") != sequence:
                raise ArtifactConsistencyError(
                    f"iteration record {sequence} has iteration={record.get('iteration')!r}; "
                    f"expected {sequence}"
                )
            model = record.get("model")
            if not isinstance(model, dict):
                raise ArtifactConsistencyError(
                    f"iteration {sequence} has no valid model"
                )
            if record.get("instrument_code") != expected_instrument_code:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} instrument_code="
                    f"{record.get('instrument_code')!r} does not match "
                    f"{expected_instrument_code!r}"
                )
            if model.get("instrument_code") != expected_instrument_code:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} model.instrument_code="
                    f"{model.get('instrument_code')!r} does not match "
                    f"{expected_instrument_code!r}"
                )
            expected_model_version = f"M{sequence:03d}"
            if model.get("version") != expected_model_version:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} model.version={model.get('version')!r}; "
                    f"expected {expected_model_version}"
                )
            if "iteration" in model and model.get("iteration") != sequence:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} model.iteration={model.get('iteration')!r}; "
                    f"expected {sequence}"
                )
            if (
                "model_version" in record
                and record.get("model_version") != expected_model_version
            ):
                raise ArtifactConsistencyError(
                    f"iteration {sequence} model_version={record.get('model_version')!r}; "
                    f"expected {expected_model_version}"
                )
            expected_parent = None if sequence == 1 else f"M{sequence - 1:03d}"
            if model.get("parent_version") != expected_parent:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} model.parent_version={model.get('parent_version')!r}; "
                    f"expected {expected_parent!r}"
                )
            if (
                "parent_model_version" in record
                and record.get("parent_model_version") != expected_parent
            ):
                raise ArtifactConsistencyError(
                    f"iteration {sequence} parent_model_version="
                    f"{record.get('parent_model_version')!r}; expected {expected_parent!r}"
                )

            cutoff = AgentIterationStorage._artifact_date(
                record.get("cutoff_date"),
                field="cutoff_date",
                iteration=sequence,
            )
            if previous_cutoff is not None and cutoff <= previous_cutoff:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} cutoff_date={cutoff.isoformat()} must be "
                    f"strictly later than {previous_cutoff.isoformat()}"
                )
            if previous_future_label is not None and cutoff <= previous_future_label:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} cutoff_date={cutoff.isoformat()} must be later "
                    f"than the previous future_label_max_date="
                    f"{previous_future_label.isoformat()}"
                )
            source_max = AgentIterationStorage._artifact_date(
                record.get("source_data_max_date"),
                field="source_data_max_date",
                iteration=sequence,
            )
            if source_max > cutoff:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} source_data_max_date={source_max.isoformat()} "
                    f"exceeds cutoff_date={cutoff.isoformat()}"
                )
            future_label = AgentIterationStorage._artifact_date(
                record.get("future_label_max_date"),
                field="future_label_max_date",
                iteration=sequence,
            )
            if future_label <= cutoff:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} future_label_max_date="
                    f"{future_label.isoformat()} must be later than cutoff_date="
                    f"{cutoff.isoformat()}"
                )
            previous_cutoff = cutoff
            previous_future_label = future_label

            ratio = record.get("fund_etf_ratio")
            if ratio not in {"7:3", "6:4", "5:5"}:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} has invalid fund_etf_ratio={ratio!r}"
                )
            fund = AgentIterationStorage._artifact_integer(
                record.get("fund_allocation"),
                field="fund_allocation",
                iteration=sequence,
            )
            etf = AgentIterationStorage._artifact_integer(
                record.get("etf_allocation"),
                field="etf_allocation",
                iteration=sequence,
            )
            target = AgentIterationStorage._artifact_integer(
                record.get("target_position"),
                field="target_position",
                iteration=sequence,
            )
            if fund + etf != target:
                raise ArtifactConsistencyError(
                    f"iteration {sequence} allocation sum {fund}+{etf} does not "
                    f"match target_position={target}"
                )

        AgentIterationStorage._validate_hash_chain(
            iterations,
            value_field="model_hash",
            parent_field="parent_model_hash",
        )
        AgentIterationStorage._validate_hash_chain(
            iterations,
            value_field="data_hash",
            parent_field="parent_data_hash",
        )

    def read_consistent_snapshot(
        self,
        instrument_code: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        last_error: ArtifactConsistencyError | None = None
        for _attempt in range(2):
            try:
                manifest = self.read_manifest(instrument_code)
                iterations = self.read_iterations(instrument_code)
                self._validate_snapshot(
                    manifest,
                    iterations,
                    expected_instrument_code=instrument_code,
                )
            except ArtifactConsistencyError as error:
                last_error = error
                continue
            return manifest, iterations
        if last_error is None:
            raise ArtifactConsistencyError("Unable to read a consistent artifact snapshot")
        raise last_error

    def write(
        self,
        instrument_code: str,
        *,
        manifest: dict[str, Any],
        iterations: list[dict[str, Any]],
    ) -> None:
        directory = self._directory(instrument_code)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        iterations_path = directory / "iterations.jsonl"
        manifest_temp = directory / "manifest.json.tmp"
        iterations_temp = directory / "iterations.jsonl.tmp"
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        iterations_temp.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in iterations
            ),
            encoding="utf-8",
        )
        iterations_temp.replace(iterations_path)
        manifest_temp.replace(manifest_path)

    def status(
        self,
        instrument_code: str,
        *,
        required_baseline: int,
        monthly_increment: int,
    ) -> dict[str, Any]:
        manifest, _iterations = self.read_consistent_snapshot(instrument_code)
        manifest = manifest or {}
        completed = int(manifest.get("completed_iterations", 0))
        return {
            "instrument_code": instrument_code,
            "completed_iterations": completed,
            "latest_model_version": manifest.get("latest_model_version"),
            "required_baseline": required_baseline,
            "baseline_remaining": max(0, required_baseline - completed),
            "monthly_increment": monthly_increment,
            "execution_mode": "agent_offline",
            "baseline_status": (
                "unchanged" if completed >= required_baseline else "incomplete"
            ),
        }
