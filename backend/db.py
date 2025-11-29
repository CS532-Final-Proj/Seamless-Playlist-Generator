import os
from typing import List

from sqlalchemy import UUID, ForeignKey, Integer, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    selectinload,
)

# Database URL from environment
DATABASE_URL = os.getenv("POSTGRES_URL")
if DATABASE_URL:
    # Convert to asyncpg URL
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create session factory
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Base class for models
Base = declarative_base()


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    title: Mapped[str] = mapped_column()
    # artist: Mapped[str] = mapped_column()
    # album: Mapped[str] = mapped_column()
    duration: Mapped[str] = mapped_column()

    def __repr__(self):
        return f"<Track(id={self.id}, title={self.title})>"


class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[str] = mapped_column(primary_key=True, type_=UUID)

    tracks: Mapped[List["PlaylistTrack"]] = relationship(order_by="PlaylistTrack.order")

    def __repr__(self):
        return f"<Playlist(id={self.id})>"


class PlaylistTrack(Base):
    __tablename__ = "playlist_tracks"

    id: Mapped[str] = mapped_column(primary_key=True, type_=UUID)
    playlist_id: Mapped[str] = mapped_column(
        ForeignKey("playlists.id"), primary_key=True
    )
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"))
    order: Mapped[int] = mapped_column()

    track: Mapped[Track] = relationship()

    UniqueConstraint("playlist_id", "order", name="uix_playlist_order")

    def __repr__(self):
        return f"<PlaylistTrack(id={self.id}, playlist_id={self.playlist_id}, track_id={self.track_id}, order={self.order})>"


async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Dependency to get DB session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_playlist_by_id(
    session: AsyncSession, playlist_id: str
) -> Playlist | None:
    stmt = (
        select(Playlist)
        .where(Playlist.id == playlist_id)
        .options(selectinload(Playlist.tracks).joinedload(PlaylistTrack.track))
    )
    result = await session.execute(stmt)
    return result.scalars().first()
