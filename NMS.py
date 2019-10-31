#!/usr/bin/python3
"""Soft-NMS.

Usage:
    nms ejemplo
    nms stdin

Options:
    -h --help       Show this screen.
"""
from docopt import docopt
import math
import json
import sys

def area(x1, y1, x2, y2):
    # area del rectangulo tal que (x1, y1) es la esq sup izq 
    # y (x2, y2) es la esq inf derecha
    if x2 > x1 and y2 > y1:
        return (x2 - x1) * (y2 - y1)
    return 0

def iou(b1, b2):
    b1x1, b1y1, b1w, b1h = b1
    b1x2, b1y2 = b1x1 + b1w, b1y1 + b1h

    b2x1, b2y1, b2w, b2h = b2
    b2x2, b2y2 = b2x1 + b2w, b2y1 + b2h

    # Esquina superior izq de la region que se superpone
    over_x1, over_y1 = max(b1x1, b2x1), max(b1y1, b2y1)

    # Esquina inf der de la region que se superpone
    over_x2, over_y2 = min(b1x2, b2x2), min(b1y2, b2y2)

    over_area = area(over_x1, over_y1, over_x2, over_y2)

    combined_area = area(b1x1, b1y1, b1x2, b1y2) + area(b2x1, b2y1, b2x2, b2y2) - over_area

    _iou = over_area / combined_area
    # return ((over_x1, over_y1, over_x2, over_y2), over_area, combined_area, _iou)
    return _iou

sigma = .5
# sigma es un hiper parametro de la funcion, segun el paper suelen utilizar 0.5
f = lambda iou: math.exp(- (iou ** 2)/sigma)

def nms(B, S, Nt=0.5, softmax=True):
    D = set()
    # conjunto de salida
    BS = list(zip(B, S))    # [(b1, s1), ...]
    BS = sorted(BS, key=lambda x: x[1]) # ordenamos por score
    while BS != []:
        M = BS.pop() # tomamos el bounding box con score maximo y lo agregamos al set de salida
        D.add(M)
        if softmax:
            # en caso de utlizar softmax, recomputamos el score de todos los bounding boxes
            # escalando cada valor por f(iou), esta es la funcion exponencial especificada en el paper.
            BS = [(b, s * f(iou(M[0], b))) for (b, s) in BS]
        else:
            # en caso de no utilizar softmax, simplemente quitamos a los BBs que no pasen la condicion de Nt.
            BS = [(b, s) for b, s in BS if iou(M[0], b) < Nt]
    return D

def convert(B):
    """
    Convertir una lista de BBs cuyas coordenadas son de la esq sup derecha a una lista
    cuyas coordenadas son de la esq sup idz. Los Width y Height no se modifican
    """
    return [(x - w, y, w, h) for x, y, w, h in B]

def convert_inv(B):
    return [(x + w, y, w, h) for x, y, w, h in B]

def process_json(jsondata):
    """
    Ejemplo entrada:
    [{
        "B": [[1,1,2,2], [2,2,2,2], [2.2,2.2,2.2,2.2], [2.4,2.2,2.2,2.3], [10,10,1,1], [1,1,2,2]],
        "S": [0.3, 0.6, 0.5, 0.9, 0.8, 0.9]
    },
    {
        "B": [[1,1,2,2], [2,2,2,2], [2.2,2.2,2.2,2.2], [2.4,2.2,2.2,2.3], [10,10,1,1], [1,1,2,2]],
        "S": [0.3, 0.6, 0.5, 0.9, 0.8, 0.9]
    }]

    Ejemplo salida:
    [{
         "B": [[2.4, 2.2, 2.2, 2.3], [10, 10, 1, 1], [2.2, 2.2, 2.2, 2.2], [1, 1, 2, 2], [2, 2, 2, 2], [1, 1, 2, 2]], 
         "S": [0.8896604515182397, 0.8, 0.05302723640637362, 0.9, 0.309909789343868, 0.037773574449889234]
    },
    {
         "B": [[2.4, 2.2, 2.2, 2.3], [10, 10, 1, 1], [2.2, 2.2, 2.2, 2.2], [1, 1, 2, 2], [2, 2, 2, 2], [1, 1, 2, 2]], 
         "S": [0.8896604515182397, 0.8, 0.05302723640637362, 0.9, 0.309909789343868, 0.037773574449889234]
    }]
    """
    output = []
    # import IPython; IPython.embed()
    for params in jsondata:
        nms_out = nms(convert(params["B"]), params["S"])
        Bout = []
        Sout = []
        for b, s in nms_out:
            Bout.append(b)
            Sout.append(s)
        output.append({
            "B": convert_inv(Bout),
            "S": Sout
        })
    return json.dumps(output)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    # print(arguments)
    if arguments["ejemplo"]:
        B = [(1,1,2,2), (2,2,2,2), (2.2,2.2,2.2,2.2), (2.4,2.2,2.2,2.3), (10,10,1,1), (1,1,2,2)]
        S = [0.3, 0.6, 0.5, 0.9, 0.8, 0.9]

        print(nms(B, S, Nt=0.7, softmax=False))
        print(nms(B, S))
    elif arguments["stdin"]:
        jsondata = json.loads(sys.stdin.read())
        print(process_json(jsondata))
  
    # import numpy as np
    # Bt = []
    # for i in range(100):
    #     box = tuple(np.random.rand(4) * 20)
    #     Bt.append(box)
    # St = list(np.random.rand(100))
    # print(nms(Bt, St))

    # import IPython; IPython.embed()