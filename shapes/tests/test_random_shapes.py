#!/usr/bin/env python

import unittest

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.draw as draw

from shapes import random_shapes, overlay_texture

class TestRandomShapes(unittest.TestCase):
    def test_overlay_texture(self):
        texture = io.imread('data/brodatz/colored/D001.tif')
        image = io.imread('data/Luke_Skywalker.png')
        start = (0, 0)
        extent = (30, 30)

        rr, cc = draw.rectangle(start, extent=extent, shape=image.shape[:2])
        mask = np.zeros_like(image)
        mask[rr, cc] = 1 
        mask = mask.astype(bool)

        overlay = overlay_texture(image, texture, mask)
        plt.imshow(overlay)
        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()
