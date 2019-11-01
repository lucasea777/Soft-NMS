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
import numpy as np

def area(x1, y1, x2, y2):
    # area del rectangulo tal que (x1, y1) es la esq sup izq 
    # y (x2, y2) es la esq inf derecha
    if x2 > x1 and y2 > y1:
        return (x2 - x1) * (y2 - y1)
    return 0

def area_numpy(x1, y1, x2, y2):
    # area del rectangulo tal que (x1, y1) es la esq sup izq 
    # y (x2, y2) es la esq inf derecha
    return np.where((x2 > x1) & (y2 > y1), (x2 - x1) * (y2 - y1), np.zeros_like(x1))

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

def iou_numpy(b1, b2):
    b1x1, b1y1, b1w, b1h = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
    b1x2, b1y2 = b1x1 + b1w, b1y1 + b1h

    b2x1, b2y1, b2w, b2h = b2
    b2x2, b2y2 = b2x1 + b2w, b2y1 + b2h

    # Esquina superior izq de la region que se superpone
    over_x1, over_y1 = np.maximum(b1x1, b2x1), np.maximum(b1y1, b2y1)

    # Esquina inf der de la region que se superpone
    over_x2, over_y2 = np.minimum(b1x2, b2x2), np.minimum(b1y2, b2y2)

    over_area = area_numpy(over_x1, over_y1, over_x2, over_y2)

    combined_area = area_numpy(b1x1, b1y1, b1x2, b1y2) + area_numpy(b2x1, b2y1, b2x2, b2y2) - over_area

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

def nms_numpy(B, S):
    # conjunto de salida
    S = np.array(S)
    B = np.array(B)
    idx = np.argsort(S)[::-1]
    for idxidx, i in enumerate(idx):
        # generamos una mascara para ocultar los indices ya utlizados
        mask = np.ones(len(S), np.bool)
        mask[idx[:idxidx+1]] = 0
        S[mask] *= np.exp(-(iou_numpy(B[mask], B[i]) ** 2)/sigma)
    return list(zip(B.tolist(), S.tolist()))

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
        nms_out = nms_numpy(convert(params["B"]), params["S"])
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

# B = [(1,1,2,2), (2,2,2,2), (2.2,2.2,2.2,2.2), (2.4,2.2,2.2,2.3), (10,10,1,1), (1,1,2,2)]
# S = [0.3, 0.6, 0.5, 0.9, 0.8, 0.9]

# print(nms_numpy(B, S))

if __name__ == "__main__":
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    # sys.stdin = open("/home/luks/Desktop/trabajo/deepvision/test2.json")
    # arguments = {
    #     "ejemplo": False,
    #     "stdin": True
    # }
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
    # N = 1000
    # for i in range(N):
    #     box = tuple(np.random.rand(4) * 20)
    #     Bt.append(box)
    # St = list(np.random.rand(N))
    # # nms(Bt, St)
    # o1 = []
    # for b, s in nms(Bt, St):
    #     o1.append(np.array([b[0], b[1], b[2], b[3], s]))
    # o1 = np.sort(np.array(o1))
    # for i in [0,1,2,3,5]:
    #     o1 = o1[o1[:,0].argsort()]
    # o2 = []
    # for b, s in nms_numpy(Bt, St):
    #     o2.append(np.array([b[0], b[1], b[2], b[3], s]))
    # o2 = np.sort(np.array(o2))
    # for i in [0,1,2,3,5]:
    #     o2 = o2[o2[:,0].argsort()]
    # print(np.allclose(o1, o2))

    

    # import IPython; IPython.embed()