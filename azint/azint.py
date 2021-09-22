import os
import numpy as np
from sparse import Sparse
from typing import Optional, Union
from collections.abc import Sequence

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

def calc_coordinates(shape, pixel_size, poni):
    d1 = (np.arange(shape[0], dtype=np.float32) + 0.5) * pixel_size - poni.poni1
    d2 = (np.arange(shape[1], dtype=np.float32) + 0.5) * pixel_size - poni.poni2
    p2, p1 = np.meshgrid(d2, d1)
    return p1, p2

class AzimuthalIntegrator():
    """
    This class is an azimuthal integrator 
    """
    def __init__(self, 
                 poni_file: str, 
                 shape: tuple[int, int], 
                 pixel_size: float, 
                 n_splitting: int, 
                 bins: list[Union[int, Sequence], Optional[Sequence]], 
                 mask: np.ndarray = None, 
                 solid_angle: bool = True,
                 polarization_factor: Optional[float] = None):
        """
        Args:
            poni_file: Name of Poni file that sets up the geometry
                of the integrator
            shape: Shape of the images to be integrated
            pixel_size: Pixel size of detector
            n_splitting: Each pixel in the image gets split into (n, n) subpixels that get binned individually
            bins: list of q and optionally phi bins. q bins can either be number of bins or a sequence defining the bin edges.
                Phi bins is a sequence
            mask: Pixel mask to exclude bad pixels. Pixels marked with 1 will be excluded
            solid_angle: Perform solid angle correction
            polarization_factor: Polarization factor for the polarization correction
                1 (linear horizontal polarization)
                -1 (linear vertical polarization)
            
        Attributes:
            q (ndarray): q bins defined as q = 4pi/lambda sin(theta) in nm-1
            phi (ndarray, optional): phi bins is case of 2D integration
        """
        self.poni = Poni(poni_file)
        
        p1, p2 = calc_coordinates(shape, pixel_size, self.poni)
        p3 = np.ones(np.prod(shape), dtype=np.float32)*self.poni.dist
        pos = np.dot(rotation_matrix(self.poni), 
                         np.vstack((p1.reshape(-1), p2.reshape(-1), p3)))
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        tth = np.arctan2(r, pos[2])
        
        qbins = bins[0]
        # calculate auto range min/max q bins
        if not any([isinstance(qbins, np.ndarray), isinstance(qbins, list)]):
            # q = 4pi/lambda sin( 2theta / 2 ) in nm-1
            q = 4.0e-9 * np.pi / self.poni.wavelength * np.sin(0.5*tth)
            bins[0] = np.linspace(np.amin(q), np.amax(q), qbins+1)
            
        # bin centers
        self.q = 0.5*(bins[0][1:] + bins[0][:-1])
        if len(bins) == 2:
            self.phi = 0.5*(bins[1][1:] + bins[1][:-1])
        else:
            self.phi = None
            
        if mask is None:
            mask = np.zeros(shape, dtype=np.uint8)
            
        self.input_size = np.prod(shape)
        self.output_shape = [len(axis)-1 for axis in bins[::-1]]
        self.sparse_matrix = Sparse(self.poni, shape, pixel_size, n_splitting, mask, bins)
        self.corrections = np.ones(shape[0]*shape[1], dtype=np.float32)
        if solid_angle:
            solid_angle = self.poni.dist / np.sqrt(self.poni.dist**2 + p1*p1 + p2*p2)
            self.corrections *= (solid_angle**3).reshape(-1)
            
        if polarization_factor:
            phi = np.arctan2(pos[0], pos[1])
            cos2_tth = np.cos(tth) ** 2
            polarization = 0.5 * (1.0 + cos2_tth - polarization_factor * np.cos(2.0 * (phi)) * (1.0 - cos2_tth))
            self.corrections *= polarization.reshape(-1)
            
        self.norm = self.sparse_matrix.spmv(self.corrections)
            
    def integrate(self, 
                  img: np.ndarray, 
                  mask: np.ndarray = None) -> np.ndarray:
        """
        Calculate the azimuthal integrated profile
        Args:
            img: Input image to be integrated
            mask: Optional pixel mask to exclude bad pixels. Note if mask is constant using the mask argument in
                the constructor is more efficient
        Returns:
            azimuthal integrated image
        """
        if img.size != self.input_size:
            raise RuntimeError('Size of image is wrong!\nExpected %d\nActual size %d' %(self.input_size, img.size))
        if mask is None:
            norm = self.norm
        else:
            inverted_mask = 1 - mask
            img = img*inverted_mask
            norm = self.sparse_matrix.spmv(inverted_mask.reshape(-1)*self.corrections)
                
        signal = self.sparse_matrix.spmv(img)
        result = np.divide(signal, norm, out=np.zeros_like(signal), where=norm!=0.0)
        return result.reshape(self.output_shape)
