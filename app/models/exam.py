"""Exams, questions and configurable rubrics."""
from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Exam(Base, TimestampMixin):
    __tablename__ = "exams"
    __table_args__ = (UniqueConstraint("class_id", "name", "exam_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    class_id: Mapped[int] = mapped_column(
        ForeignKey("classes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    exam_date: Mapped[str | None] = mapped_column(String(20))
    subject: Mapped[str | None] = mapped_column(String(120))
    language: Mapped[str] = mapped_column(String(20), default="mixed")
    total_marks: Mapped[float | None] = mapped_column(Float)
    scoring_strategy: Mapped[str | None] = mapped_column(String(40))  # override global flag
    enable_feedback: Mapped[bool] = mapped_column(Boolean, default=False)

    klass: Mapped["Klass"] = relationship(back_populates="exams")  # noqa: F821
    questions: Mapped[list["Question"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", order_by="Question.q_no"
    )


class Question(Base, TimestampMixin):
    __tablename__ = "questions"
    __table_args__ = (UniqueConstraint("exam_id", "q_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(
        ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    q_no: Mapped[str] = mapped_column(String(20), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hindi: Mapped[str | None] = mapped_column(Text)
    max_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # per-question strategy override: exact | numeric | rubric_llm | hybrid (NULL = inherit)
    strategy: Mapped[str | None] = mapped_column(String(40))
    # answer key used by exact/numeric strategies and as ground truth for LLM
    answer_key: Mapped[str | None] = mapped_column(Text)
    answer_key_alternatives: Mapped[str | None] = mapped_column(Text)  # '|' separated
    rubric_id: Mapped[int | None] = mapped_column(
        ForeignKey("rubrics.id", ondelete="SET NULL")
    )

    exam: Mapped[Exam] = relationship(back_populates="questions")
    rubric: Mapped["Rubric | None"] = relationship(back_populates="questions")


class Rubric(Base, TimestampMixin):
    __tablename__ = "rubrics"
    __table_args__ = (UniqueConstraint("name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    questions: Mapped[list[Question]] = relationship(back_populates="rubric")
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
    name: Mapped[str] = mapped_column(String(80), nullable=False)  # correctness, ...
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    description: Mapped[str | None] = mapped_column(Text)
    # knockout: if awarded == 0 on this criterion, total score becomes 0.
    knockout: Mapped[bool] = mapped_column(Boolean, default=False)
    # minimum fraction of this criterion required for the knockout to trigger
    knockout_below_fraction: Mapped[float] = mapped_column(Float, default=0.0)

    rubric: Mapped[Rubric] = relationship(back_populates="criteria")
