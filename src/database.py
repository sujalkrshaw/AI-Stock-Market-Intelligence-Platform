from sqlalchemy import (

    create_engine,

    Column,

    Integer,

    String,

    Float
)

from sqlalchemy.orm import (

    declarative_base,

    sessionmaker
)

# ==========================================
# DATABASE URL
# ==========================================

DATABASE_URL = (

    "sqlite:///database/portfolio.db"
)

# ==========================================
# ENGINE
# ==========================================

engine = create_engine(

    DATABASE_URL,

    connect_args={

        "check_same_thread": False
    }
)

# ==========================================
# SESSION
# ==========================================

SessionLocal = sessionmaker(

    autocommit=False,

    autoflush=False,

    bind=engine
)

# ==========================================
# BASE
# ==========================================

Base = declarative_base()

# ==========================================
# PORTFOLIO TABLE
# ==========================================

class Portfolio(Base):

    __tablename__ = "portfolio"

    id = Column(

        Integer,

        primary_key=True,

        index=True
    )

    stock = Column(String)

    quantity = Column(Integer)

    buy_price = Column(Float)

    current_price = Column(Float)

    investment = Column(Float)

    current_value = Column(Float)

    profit_loss = Column(Float)

    return_percentage = Column(Float)

# ==========================================
# CREATE TABLES
# ==========================================

Base.metadata.create_all(
    bind=engine
)

print("\nDATABASE CONNECTED SUCCESSFULLY\n")