"""Dosing regimen types — DosingEvent and DosingRegimen.

Frozen dataclasses representing dosing schedules. Factory methods
produce common clinical regimens (oral repeated, IV infusion, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ORAL_NODE: str = "stomach_lumen"
DEFAULT_IV_NODE: str = "venous_blood"

# Infusion discretization: boluses every INFUSION_STEP_H during infusion window
INFUSION_STEP_H: float = 0.1


# ---------------------------------------------------------------------------
# DosingEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DosingEvent:
    """A single dosing event.

    Attributes:
        time_h: Time of dose administration (hours from t=0).
        dose_mg: Dose amount (mg).
        node: Target node in the body graph (e.g. ``"stomach_lumen"``).
        duration_h: Infusion duration (hours). 0.0 = instantaneous bolus.
    """

    time_h: float
    dose_mg: float
    node: str
    duration_h: float = 0.0

    def __post_init__(self) -> None:
        if self.dose_mg < 0:
            raise ValueError(f"dose_mg must be non-negative, got {self.dose_mg}")
        if self.time_h < 0:
            raise ValueError(f"time_h must be non-negative, got {self.time_h}")
        if self.duration_h < 0:
            raise ValueError(f"duration_h must be non-negative, got {self.duration_h}")

    @property
    def is_bolus(self) -> bool:
        """True if this is an instantaneous bolus (duration_h == 0)."""
        return self.duration_h == 0.0

    @property
    def end_time_h(self) -> float:
        """Time at which the infusion ends (same as time_h for bolus)."""
        return self.time_h + self.duration_h


# ---------------------------------------------------------------------------
# DosingRegimen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DosingRegimen:
    """Ordered collection of dosing events.

    Events are stored as a tuple (immutable) sorted by time.

    Attributes:
        events: Tuple of DosingEvent instances, sorted by time_h.
    """

    events: tuple[DosingEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("DosingRegimen must have at least one event")
        # Validate sort order
        for i in range(1, len(self.events)):
            if self.events[i].time_h < self.events[i - 1].time_h:
                raise ValueError(
                    f"Events must be sorted by time_h: "
                    f"event[{i}].time_h={self.events[i].time_h} < "
                    f"event[{i-1}].time_h={self.events[i-1].time_h}"
                )

    @property
    def n_doses(self) -> int:
        """Number of dosing events."""
        return len(self.events)

    @property
    def total_dose_mg(self) -> float:
        """Sum of all dose amounts."""
        return sum(e.dose_mg for e in self.events)

    @property
    def last_dose_time_h(self) -> float:
        """Time of the last dosing event."""
        return self.events[-1].time_h

    # -- Factory methods ---------------------------------------------------

    @staticmethod
    def single_oral(dose_mg: float) -> DosingRegimen:
        """Single oral dose at t=0.

        Args:
            dose_mg: Dose amount (mg).

        Returns:
            A DosingRegimen with one oral bolus event.
        """
        return DosingRegimen(
            events=(
                DosingEvent(time_h=0.0, dose_mg=dose_mg, node=DEFAULT_ORAL_NODE),
            )
        )

    @staticmethod
    def single_iv(dose_mg: float, duration_h: float = 0.0) -> DosingRegimen:
        """Single IV dose (bolus or infusion) at t=0.

        Args:
            dose_mg: Dose amount (mg).
            duration_h: Infusion duration (hours). 0.0 = bolus.

        Returns:
            A DosingRegimen with one IV event.
        """
        return DosingRegimen(
            events=(
                DosingEvent(
                    time_h=0.0,
                    dose_mg=dose_mg,
                    node=DEFAULT_IV_NODE,
                    duration_h=duration_h,
                ),
            )
        )

    @staticmethod
    def oral_repeated(
        dose_mg: float,
        interval_h: float,
        n_doses: int,
    ) -> DosingRegimen:
        """Repeated oral dosing at fixed intervals.

        First dose at t=0, subsequent doses every ``interval_h``.

        Args:
            dose_mg: Dose amount per administration (mg).
            interval_h: Dosing interval (hours).
            n_doses: Number of doses.

        Returns:
            A DosingRegimen with ``n_doses`` oral bolus events.
        """
        if n_doses < 1:
            raise ValueError(f"n_doses must be >= 1, got {n_doses}")
        if interval_h <= 0:
            raise ValueError(f"interval_h must be > 0, got {interval_h}")
        events = tuple(
            DosingEvent(
                time_h=i * interval_h,
                dose_mg=dose_mg,
                node=DEFAULT_ORAL_NODE,
            )
            for i in range(n_doses)
        )
        return DosingRegimen(events=events)

    @staticmethod
    def iv_infusion(
        dose_mg: float,
        duration_h: float,
        interval_h: float,
        n_doses: int,
    ) -> DosingRegimen:
        """Repeated IV infusion at fixed intervals.

        First infusion starts at t=0, subsequent infusions every
        ``interval_h``.

        Args:
            dose_mg: Dose amount per infusion (mg).
            duration_h: Infusion duration per dose (hours).
            interval_h: Dosing interval (hours).
            n_doses: Number of infusions.

        Returns:
            A DosingRegimen with ``n_doses`` IV infusion events.
        """
        if n_doses < 1:
            raise ValueError(f"n_doses must be >= 1, got {n_doses}")
        if interval_h <= 0:
            raise ValueError(f"interval_h must be > 0, got {interval_h}")
        if duration_h < 0:
            raise ValueError(f"duration_h must be >= 0, got {duration_h}")
        events = tuple(
            DosingEvent(
                time_h=i * interval_h,
                dose_mg=dose_mg,
                node=DEFAULT_IV_NODE,
                duration_h=duration_h,
            )
            for i in range(n_doses)
        )
        return DosingRegimen(events=events)
