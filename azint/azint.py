import os
import numpy as np
from sparse import Sparse
from typing import Optional, Union
from collections.abc import Sequence, Iterable

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
                 radial_bins: Union[int, Sequence],
                 azimuth_bins: Optional[Union[int, Sequence]] = None,
                 unit: str = 'q',
                 mask: np.ndarray = None, 
                 solid_angle: bool = True,
                 polarization_factor: Optional[float] = None,
                 error_model: Optional[str] = None):
        """
        Args:
            poni_file: Name of Poni file that sets up the geometry
                of the integrator
            shape: Shape of the images to be integrated
            pixel_size: Pixel size of detector
            n_splitting: Each pixel in the image gets split into (n, n) subpixels that get binned individually
            radial_bins: radial bins can either be number of bins or a sequence defining the bin edges in Angstrom^-1.
            azimuth_bins: azimthual bins can either be number of bins or a sequence defining the bin edges between [0, 360] degrees.
            unit: Ouput units for the radial coordindate
            mask: Pixel mask to exclude bad pixels. Pixels marked with 1 will be excluded
            solid_angle: Perform solid angle correction
            polarization_factor: Polarization factor for the polarization correction
                1 (linear horizontal polarization)
                -1 (linear vertical polarization)
            
        Attributes:
            radial_axis (ndarray): radial axis depeding on units in q or 2theta
            azimuth_axis (ndarray, optional): azimuth axis in degrees is case of 2D integration
        """
        
        if error_model and error_model != 'poisson':
            raise RuntimeError('Only poisson error model is supported')
        
        if error_model and n_splitting > 1:
            raise RuntimeError('Cannot estimate errors with pixel splitting.\n Set n_splitting to 1 for error estimation')
        
        if unit not in ('q', '2th'):
            raise RuntimeError('Wrong output unit. Allowed units: q, 2th')
        
        self.error_model = error_model
        self.poni = Poni(poni_file)
        
        p1, p2 = calc_coordinates(shape, pixel_size, self.poni)
        p3 = np.ones(np.prod(shape), dtype=np.float32)*self.poni.dist
        pos = np.dot(rotation_matrix(self.poni), 
                         np.vstack((p1.reshape(-1), p2.reshape(-1), p3)))
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        tth = np.arctan2(r, pos[2])
        
        # calculate auto range min/max radial bins
        if not isinstance(radial_bins, Iterable):
            if unit == 'q':
                # q = 4pi/lambda sin( 2theta / 2 ) in A-1
                q = 4.0e-10 * np.pi / self.poni.wavelength * np.sin(0.5*tth)
                radial_bins = np.linspace(np.amin(q), np.amax(q), radial_bins+1)
                self.radial_axis = 0.5*(radial_bins[1:] + radial_bins[:-1])
            elif unit == '2th':
                radial_bins = np.linspace(np.amin(tth), np.amax(tth), radial_bins+1)
                self.radial_axis = np.rad2deg(0.5*(radial_bins[1:] + radial_bins[:-1]))
        bins = [radial_bins]
        
        self.azimuth_axis = None
        if azimuthal_bins is not None:
            if not isinstance(azimuthal_bins, Iterable):
                azimuthal_bins = np.linspace(0, 360, azimuthal_bins+1)
            self.azimuth_axis = 0.5*(azimuthal_bins[1:] + azimuthal_bins[:-1])
            bins.append(azimuthal_bins)
            
        if mask is None:
            mask = np.zeros(shape, dtype=np.uint8)
            
        self.input_size = np.prod(shape)
        self.output_shape = [len(axis)-1 for axis in bins[::-1]]
        self.sparse_matrix = Sparse(self.poni, shape, pixel_size, n_splitting, mask, bins, unit)
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
                  mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calculate the azimuthal integrated profile
        Args:
            img: Input image to be integrated
            mask: Optional pixel mask to exclude bad pixels. Note if mask is constant using the mask argument in
                the constructor is more efficient
        Returns:
            azimuthal integrated image and optionally error estimate sigma when error_model is specified
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
        result = result.reshape(self.output_shape)
        
        if self.error_model:
            # poisson error model
            sigma = np.sqrt(signal)
            sigma = np.divide(sigma, norm, out=np.zeros_like(sigma), where=norm!=0.0)
            sigma = sigma.reshape(self.output_shape)
            return result, sigma
        else:
            return result, None
