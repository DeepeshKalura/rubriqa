"""Rubric and RubricCriterion (used by entry-evaluation scoring).

The exam entity (Exam + Question) and the org hierarchy that used to live in
this file were removed in ADR-0001. The Program/Entry model carries the
free-text equivalents (`school_name`, `class_name`, `full_name`,
`student_external_id`) that the entry flow actually uses.
"""
from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Rubric(Base, TimestampMixin):
    __tablename__ = "rubrics"
    __table_args__ = (UniqueConstraint("name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    criteria: Mapped[list["RubricCriterion"]] = relationship(
        back_populates="rubric", cascade="all, delete-orphan", order_by="RubricCriterion.id"
    )


class RubricCriterion(Base, TimestampMixin):
    __tablename__ = "rubric_criteria"
    __table_args__ = (
        UniqueConstraint("rubric_id", "name"),
        CheckConstraint("weight >= 0", name="weight_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    rubric_id: Mapped[int] = mapped_column(
        ForeignKey("rubrics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    description: Mapped[str | None] = mapped_column(Text)
    knockout: Mapped[bool] = mapped_column(default=False)
    knockout_below_fraction: Mapped[float] = mapped_column(Float, default=0.0)

    rubric: Mapped[Rubric] = relationship(back_populates="criteria")
