
import sys

import project2
from project2 import project2

def test_eda():
    send = [{"id":100,"cuisine":"greek","ingredients":["rice","pepper","tomatoes","salt"]},
            {"id":200,"cuisine":"indian","ingredients":["sea salt","milk","water"]}]
    y, ids, rec_ings = project2.eda(send)
    print(y,ids,rec_ings)
    assert y == ['greek', 'indian']
    assert ids == [100,200]
    assert rec_ings == [['rice','pepper','tomatoes','salt'],['sea salt','milk','water']]
