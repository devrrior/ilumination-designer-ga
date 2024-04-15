from .base import Base
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Mapped
from sqlalchemy import String, Integer, Float
from sqlalchemy.dialects.mysql import LONGTEXT



class Bulb(Base):
    __tablename__ = 'bulbs'

    id: Mapped[int] = mapped_column(primary_key=True)
    brand: Mapped[str] = mapped_column(String(255))
    technology_type: Mapped[str] = mapped_column(String(255))
    wattage: Mapped[float] = mapped_column(Float())
    lumens: Mapped[int] = mapped_column(Integer())
    light_type: Mapped[str] = mapped_column(String(255))
    units_per_package: Mapped[int] = mapped_column(Integer())
    price_per_package: Mapped[float] = mapped_column(Float())
    url: Mapped[str] = mapped_column(LONGTEXT)