from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Replace with your actual connection string
connection_string = "mysql+mysqlconnector://mlflow:tuh4ujr6nhc2DQM!cgx@35.246.172.208:3306/mlflow"

try:
    # Create an engine
    engine = create_engine(connection_string)

    # Test the connection
    connection = engine.connect()
    connection.close()

    print("Connection successful!")
except SQLAlchemyError as e:
    print("Connection failed:", e)
