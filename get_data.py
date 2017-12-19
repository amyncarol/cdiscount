import numpy as np
import pandas as pd
import io
import bson
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp

def process(q, iolock, prod_to_category, prod_to_images):
    while True:
        d=q.get()
        if d is None:
            break
        product_id = d['_id']
        category_id = d['category_id']
        prod_to_category[product_id] = category_id
        l = []
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            l.append(picture)
        prod_to_images[product_id] = l

def get_data(NCORE, bsonfile):
    """
    given number of cores, and bsonfile location,
    returns:
    dataframe1: _id, category_id
    dataframe2: _id, list of images
    """
    q = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    manager =  mp.Manager()
    prod_to_category = manager.dict()
    prod_to_images = manager.dict()
    pool = mp.Pool(NCORE, initializer = process, initargs=(q, iolock, prod_to_category, prod_to_images))

    data = bson.decode_file_iter(open(bsonfile, 'rb'))
    for c, d in enumerate(data):
        q.put(d)   # blocks until q below its max size

    # tell workers we're done
    for _ in range(NCORE):
        q.put(None)
    pool.close()
    pool.join()

    prod_to_category = dict(prod_to_category)
    prod_to_images = dict(prod_to_images)
    
    prod_to_category = pd.DataFrame(list(prod_to_category.items()), columns=['_id', 'category_id'])
    prod_to_images = pd.DataFrame(list(prod_to_images.items()), columns=['_id', 'images'])

    return prod_to_category, prod_to_images
