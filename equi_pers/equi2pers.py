import os
import sys
import cv2
import numpy as np


def equi2pers(wFOV, hFOV, THETA, PHI, h, w, img):

    [height, width, _] = img.shape
    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    equ_h = height
    equ_w = width
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    x_map = np.ones([h, w], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, w), [h, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, h), [w, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(
        D[:, :, np.newaxis], 3, axis=2)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([h * w, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([h, w]) / np.pi * 180
    lat = -lat.reshape([h, w]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    persp = cv2.remap(img,
                      lon.astype(np.float32),
                      lat.astype(np.float32),
                      cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_WRAP)
    return persp