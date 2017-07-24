#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:30:01 2017

@author: mittelberger2

This module finds the local maxima in an image. If there is noise present in the image, make sure to blur it before
passing it to 'local_maxima', otherwise almost every pixel will be marked as a maximum. You can also use a high value
for 'noise_tolerance', but usually using a blurred image and a lower value for 'noise_tolerance' gives better results.

The algorithm is a python implementation of the ImageJ plugin "MaximumFinder".
(https://github.com/imagej/imagej1/blob/master/ij/plugin/filter/MaximumFinder.java)
"""

import numpy as np
from . import analyze_maxima

def local_maxima(blurred_image, noise_tolerance=0):
    """
    This is a wrapper function for the two steps of finding the local maxima of an image. It is recommended to always
    use this function for finding local maxima.
    It returns a list of (y, x) tuples, where each tuple gives the coordinates of one local maxium in pixel
    coordinates. The list is sorted after intensity, with the highest maximum first.
    """
    raw_maxima = _raw_local_maxima(blurred_image)
    return _analyze_and_mark_maxima(blurred_image, raw_maxima[1], noise_tolerance=noise_tolerance)

def _raw_local_maxima(blurred_image):
    shape = blurred_image.shape
    local_maxima = np.zeros(shape)

    # This is the fast version of the commented for-loops below
    extended_blurred_image = np.empty(shape + (9,))
    extended_blurred_image[..., 0] = blurred_image
    y_positions = [(1, None), (0, -1), (0, None), (1, None), (0, -1), (0, None), (1, None), (0, -1)]
    x_positions = [(0, None), (0, None), (1, None), (1, None), (1, None), (0, -1), (0, -1), (0, -1)]
    mappings = {(0, None): (0, None), (1, None): (0, -1), (0, -1): (1, None)}
    for i in range(8):
        target_y = y_positions[i]
        source_y = mappings[target_y]
        target_x = x_positions[i]
        source_x = mappings[target_x]
        extended_blurred_image[..., i+1][target_y[0]:target_y[1], target_x[0]:target_x[1]] = (
                                                blurred_image[source_y[0]:source_y[1], source_x[0]:source_x[1]])

    max_positions = np.argmax(extended_blurred_image, axis=-1)
    local_maxima[max_positions == 0] = blurred_image[max_positions == 0]
    local_maxima[0, :] = 0
    local_maxima[-1, :] = 0
    local_maxima[:, 0] = 0
    local_maxima[:, -1] = 0

#        y_positions = [1, -1, 0, 1, -1,  0,  1, -1]
#        x_positions = [0,  0, 1, 1,  1, -1, -1, -1]
#        for y in range(shape[0]):
#            if y < 1 or y > shape[0] - 2:
#                continue
#            for x in range(shape[1]):
#                if x < 1 or x > shape[1] - 2:
#                    continue
#                is_max = True
#                for k in range(8):
#                    if blurred_image[y, x] <= blurred_image[y + y_positions[k], x + x_positions[k]] + noise_tolerance:
#                        is_max = False
#                        break
#                if is_max:
#                    local_maxima[y,x] = blurred_image[y,x]
    return local_maxima, list(zip(*np.where(local_maxima)))

def _analyze_and_mark_maxima(blurred_image, maxima, noise_tolerance=0):
    flattened_blurred_image = blurred_image.ravel().astype(np.float32)
    shape = blurred_image.shape
    sorted_maxima = maxima.copy()
    sorted_maxima.sort(key=lambda entry: blurred_image[entry], reverse=True)
    resulting_maxima = []
    array_sorted_maxima = np.array(sorted_maxima, dtype=np.uintc)
    flattened_array_sorted_maxima = array_sorted_maxima[:, 0] * shape[1] + array_sorted_maxima[:, 1]
    analyze_maxima.analyze_maxima(flattened_blurred_image, shape, flattened_array_sorted_maxima, resulting_maxima, noise_tolerance)
    converted_resulting_maxima = []
    for maximum in resulting_maxima:
        converted_resulting_maxima.append((maximum//shape[1], maximum%shape[1]))
    return converted_resulting_maxima