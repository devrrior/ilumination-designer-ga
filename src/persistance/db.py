from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models.bulb import Bulb
from .models.base import Base
import csv


engine = create_engine("mysql+pymysql://root:@localhost:3306/lighting_ga")

Session = sessionmaker(bind=engine)


def load_data_from_csv(path):
    Session = sessionmaker(bind=engine)
    session = Session()
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data = Bulb(
                brand=row['brand'],
                technology_type=row['technology_type'],
                wattage=row['wattage'],
                lumens=row['lumens'],
                light_type=row['light_type'],
                units_per_package=row['units_per_package'],
                price_per_package=row['price_per_package'],
                url=row['url']
            )
            session.add(data)
    session.commit()

def create_tables():
    Base.metadata.create_all(engine)