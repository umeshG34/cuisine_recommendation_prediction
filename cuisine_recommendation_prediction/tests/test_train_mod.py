
import sys

import project2
from project2 import project2

def test_train_mod():
    send = [{"id":100,"cuisine":"greek","ingredients":["rice","pepper","tomatoes","salt"]},
            {"id":200,"cuisine":"indian","ingredients":["sea salt","milk","water"]}]
    clf , vectorize, ids, X = project2.train_mod(send)
    print(len(ids), X.shape)
    assert len(ids) == 2
    assert X.shape[0] == 2
    assert X.shape[1] == len(set(['rice','pepper','tomatoes','salt','sea salt','milk','water']))
