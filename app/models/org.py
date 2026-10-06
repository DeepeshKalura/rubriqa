"""Organization hierarchy: Organization -> Class -> Batch -> Student."""
from __future__ import annotations

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    code: Mapped[str | None] = mapped_column(String(50))

    classes: Mapped[list["Klass"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class Klass(Base, TimestampMixin):
    """`Class` is a Python keyword; the domain entity is a school class."""

    __tablename__ = "classes"
    __table_args__ = (UniqueConstraint("organization_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    section: Mapped[str | None] = mapped_column(String(40))

    organization: Mapped[Organization] = relationship(back_populates="classes")
    batches: Mapped[list["Batch"]] = relationship(
        back_populates="klass", cascade="all, delete-orphan"
    )
    exams: Mapped[list["Exam"]] = relationship(back_populates="klass")  # noqa: F821


class Batch(Base, TimestampMixin):
    __tablename__ = "batches"
    __table_args__ = (UniqueConstraint("class_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    class_id: Mapped[int] = mapped_column(
        ForeignKey("classes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    year: Mapped[int | None] = mapped_column()

    klass: Mapped[Klass] = relationship(back_populates="batches")
    enrollments: Mapped[list["Enrollment"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class Student(Base, TimestampMixin):
    __tablename__ = "students"
    __table_args__ = (UniqueConstraint("organization_id", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(80), nullable=False)  # roll number
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str | None] = mapped_column(String(200))

    enrollments: Mapped[list["Enrollment"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    submissions: Mapped[list["Submission"]] = relationship(back_populates="student")  # noqa: F821


class Enrollment(Base, TimestampMixin):
    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("batch_id", "student_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    roll_no_in_batch: Mapped[str | None] = mapped_column(String(40))

    batch: Mapped[Batch] = relationship(back_populates="enrollments")
    student: Mapped[Student] = relationship(back_populates="enrollments")
