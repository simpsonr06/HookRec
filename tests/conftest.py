import pandas as pd
import pytest
@pytest.fixture
def toy_data():
    interactions=pd.DataFrame([(1,10,10),(1,11,8),(1,12,2),(2,10,9),(2,11,7),(2,13,4),(3,12,9),(3,13,8),(3,14,3),(4,10,4),(4,14,8),(4,15,7)],columns=['user_id','artist_id','weight'])
    artists=pd.DataFrame([(10,'Artist A'),(11,'Artist B'),(12,'Artist C'),(13,'Artist D'),(14,'Artist E'),(15,'Artist F')],columns=['artist_id','artist_name'])
    tags=pd.DataFrame([(10,'rock'),(10,'alternative'),(11,'rock'),(12,'jazz'),(13,'jazz'),(14,'electronic'),(15,'electronic')],columns=['artist_id','tag'])
    return interactions,artists,tags
