import os
import numpy as np
from _azint import generate_matrix, spmv


class Poni():
    def __init__(self, filename):
        config = {}
        with open(filename) as opened_file:
            for line in opened_file:
                if line.startswith("#") or (":" not in line):
                    continue
                words = line.split(":", 1)
                key = words[0].strip().lower()
                value = words[1].strip()
                config[key] = value
                
        self.dist = float(config['distance'])
        self.poni1 = float(config['poni1'])
        self.poni2 = float(config['poni2'])
        self.rot1 = float(config['rot1'])
        self.rot2 = float(config['rot2'])
        self.rot3 = float(config['rot3'])
        self.wavelength = float(config['wavelength'])
        
def rotation_matrix(poni: Poni):
    cos_rot1 = np.cos(poni.rot1)
    cos_rot2 = np.cos(poni.rot2)
    cos_rot3 = np.cos(poni.rot3)
    sin_rot1 = np.sin(poni.rot1)
    sin_rot2 = np.sin(poni.rot2)
    sin_rot3 = np.sin(poni.rot3)

    # Rotation about axis 1: Note this rotation is left-handed
    rot1 = np.array([[1.0, 0.0, 0.0],
                     [0.0, cos_rot1, sin_rot1],
                     [0.0, -sin_rot1, cos_rot1]])
    
    # Rotation about axis 2. Note this rotation is left-handed
    rot2 = np.array([[cos_rot2, 0.0, -sin_rot2],
                     [0.0, 1.0, 0.0],
                     [sin_rot2, 0.0, cos_rot2]])
    
    # Rotation about axis 3: Note this rotation is right-handed
    rot3 = np.array([[cos_rot3, -sin_rot3, 0.0],
                     [sin_rot3, cos_rot3, 0.0],
                     [0.0, 0.0, 1.0]])
    
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)
    return rotation_matrix

# calculate max q value from the 4 corners of the detector
def calculate_maxq(shape, poni: Poni, pixel_size: float):
    rot = rotation_matrix(poni)
    maxq = 0.0
    n, m = shape[0] - 1, shape[1] -1
    i1 = [0, 0, n, n]
    i2 = [0, m, 0, m]
    for i in range(4):
        p = [(i1[i] + 0.5) * pixel_size - poni.poni1,
             (i2[i] + 0.5) * pixel_size - poni.poni2,
             poni.dist
        ]
        pos = np.dot(rot, p)
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        tth = np.arctan2(r, pos[2])
        # q = 4pi/lambda sin( 2theta / 2 ) in nm-1
        q = 4.0e-9 * np.pi / poni.wavelength * np.sin(0.5*tth)
        if q > maxq:
            maxq = q
    return maxq

class AzimuthalIntegrator():
    def __init__(self, poni_file, shape, pixel_size, n_splitting, mask, bins, solid_angle=True):
        poni = Poni(poni_file)
        qbins = bins[0]
        if not any([isinstance(qbins, np.ndarray), isinstance(qbins, list)]):
            maxq = calculate_maxq(shape, poni, pixel_size)
            # assume beam center is inside the detector
            bins[0] = np.linspace(0.0, maxq, qbins+1)
            
        self.q = 0.5*(bins[0][1:] + bins[0][:-1])
        if len(bins) == 2:
            self.phi = 0.5*(bins[1][1:] + bins[1][:-1])
        else:
            self.phi = None
            
        self.output_shape = [len(axis)-1 for axis in bins[::-1]]
        self.sparse_matrix = generate_matrix(poni, shape, pixel_size, n_splitting, mask, bins)
            
        if solid_angle:
            d1 = (np.arange(shape[0], dtype=np.float32) + 0.5) * pixel_size - poni.poni1
            d2 = (np.arange(shape[1], dtype=np.float32) + 0.5) * pixel_size - poni.poni2
            p1, p2 = np.meshgrid(d2, d1)
            solid_angle = poni.dist / np.sqrt(poni.dist**2 + p1*p1 + p2*p2)
            self.norm = spmv(*self.sparse_matrix, solid_angle**3)
        else:
            self.norm = spmv(*self.sparse_matrix, np.ones(shape[0]*shape[1], dtype=np.float32))
            
    def integrate(self, img):
        signal = spmv(*self.sparse_matrix, img)
        result = np.divide(signal, self.norm, out=np.zeros_like(signal), where=self.norm!=0.0)
        return result.reshape(self.output_shape)
