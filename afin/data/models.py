from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stock'

    id = Column(Integer, primary_key = True, autoincrement=True)

    symbol = Column(String)
    name = Column(String)
    security_name = Column(String)
    category = Column(String)
    test_issue = Column(Boolean)
    status = Column(Boolean)
    round_lot_size = Column(Integer)

    prices = relationship("StockPrice", back_populates="stock")


class StockPrice(Base):
    __tablename__ = 'stock_price'
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stock.id'))
    stock = relationship("Stock", back_populates="prices")

    date = Column(Date)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

