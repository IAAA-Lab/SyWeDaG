"""Machine-learning hourly precipitation generation strategy."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import List, Sequence

import numpy as np
from xgboost import XGBRegressor

from data_sources.base_source import HourlyPrecipitationRecord
from utils.system_utils import safe_print


class HistoricalHourlyPrecipitationModel:
    """Learn daily 24-hour precipitation profiles from hourly observations."""

    def __init__(self, records: Sequence[HourlyPrecipitationRecord], random_state: int | None = None):
        self.records = list(records)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._model: XGBRegressor | None = None
        self._fallback_profiles: dict[int, list[np.ndarray]] = defaultdict(list)
        self._generated_days = 0

    @staticmethod
    def _features(day: datetime, daily_total: float) -> list[float]:
        return [
            float(day.month),
            float(day.timetuple().tm_yday),
            float(day.weekday()),
            float(daily_total),
        ]

    def fit(self) -> "HistoricalHourlyPrecipitationModel":
        """Fit one regressor per hour to historical normalized profiles."""
        by_date: dict[str, dict[int, float]] = defaultdict(dict)
        for record in self.records:
            if 0 <= int(record.hour) < 24 and record.precipitation >= 0:
                by_date[record.date][int(record.hour)] = float(record.precipitation)

        features: list[list[float]] = []
        targets: list[np.ndarray] = []
        incomplete_days = 0
        dry_days = 0
        for date_text, hourly_values in sorted(by_date.items()):
            if len(hourly_values) != 24:
                incomplete_days += 1
                continue
            day = datetime.strptime(date_text, "%Y-%m-%d")
            profile = np.array([hourly_values[hour] for hour in range(24)], dtype=float)
            total = float(profile.sum())
            if total <= 0:
                dry_days += 1
                continue
            normalized = profile / total
            features.append(self._features(day, total))
            targets.append(normalized)
            self._fallback_profiles[day.month].append(normalized)

        if not features:
            raise ValueError("Hourly precipitation history contains no complete rainy days")

        safe_print(
            f"🤖 Hourly precipitation model training: {len(features)} rainy days, "
            f"{dry_days} dry days ignored, {incomplete_days} incomplete days ignored"
        )

        self._model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(np.asarray(features), np.asarray(targets))
        safe_print("✅ Hourly precipitation model trained with 24-hour profiles")
        return self

    def generate(self, date_text: str, daily_total: float) -> list[float]:
        """Generate a stochastic 24-hour profile preserving the daily total."""
        total = max(float(daily_total or 0.0), 0.0)
        self._generated_days += 1
        if total == 0:
            if self._should_report_day():
                safe_print(f"🌤️ {date_text}: daily precipitation=0.00 mm -> 24 dry hours")
            return [0.0] * 24
        if self._model is None:
            raise RuntimeError("The hourly precipitation model must be fitted first")

        day = datetime.strptime(date_text, "%Y-%m-%d")
        profile = np.asarray(
            self._model.predict(np.asarray([self._features(day, total)]))[0],
            dtype=float,
        )
        if np.any(profile < 0) or float(profile.sum()) <= 0:
            candidates = self._fallback_profiles.get(day.month, [])
            profile = candidates[0].copy() if candidates else np.full(24, 1 / 24)
        profile = np.clip(profile, 0.0, None)

        # Keep the learned shape while preventing identical profiles for equal totals.
        profile *= self._rng.lognormal(mean=0.0, sigma=0.12, size=24)
        profile_sum = float(profile.sum())
        if profile_sum <= 0:
            profile = np.full(24, 1 / 24)
        else:
            profile /= profile_sum
        generated = (profile * total).tolist()
        wet_hours = sum(value > 0.0 for value in generated)
        if self._should_report_day():
            safe_print(
                f"🌧️ {date_text}: target={total:.2f} mm, generated="
                f"{sum(generated):.2f} mm, wet hours={wet_hours}, "
                f"peak={max(generated):.2f} mm/h"
            )
        return generated

    def _should_report_day(self) -> bool:
        """Report a short sample plus periodic progress during long runs."""
        return self._generated_days <= 3 or self._generated_days % 30 == 0

    def generate_for_days(self, daily_data: List[dict]) -> list[float]:
        """Generate concatenated hourly precipitation for synthetic daily records."""
        return [
            value
            for record in daily_data
            for value in self.generate(record["date"], record.get("precipitation", 0.0))
        ]
