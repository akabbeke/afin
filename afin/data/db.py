from re import I
from sqlalchemy import create_engine
engine = create_engine('sqlite:///afin.db')

def create_db():
    from models import Base
    Base.metadata.create_all(engine)
