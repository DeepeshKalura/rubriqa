"""Google Forms OAuth credentials (encrypted refresh tokens)."""
from __future__ import annotations

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class GoogleCredential(Base, TimestampMixin):
    __tablename__ = "google_credentials"
    __table_args__ = (UniqueConstraint("account_email"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account_email: Mapped[str] = mapped_column(String(200), nullable=False)
    # Fernet-encrypted token JSON. The Fernet key comes from APP_SECRET_KEY.
    encrypted_token: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[str | None] = mapped_column(Text)
    revoked: Mapped[bool] = mapped_column(default=False)
