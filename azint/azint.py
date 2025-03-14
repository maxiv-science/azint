import os
import json
from dataclasses import dataclass
from typing import Optional, Union
from collections.abc import Sequence, Iterable
import numpy as np
from .detector import Detector
from _azint import Sparse


__all__ = ['Poni', 'AzimuthalIntegrator']


@dataclass
class Poni:
    det: Detector
    dist: float
    poni1: float
    poni2: float
    rot1: float
    rot2: float
    rot3: float
    wavelength: float
    
    @classmethod
    def from_file(cls, filename):
        config = {}
        with open(filename) as fh:
            for line in fh:
                if line.startswith("#") or (":" not in line):
                    continue
                words = line.split(":", 1)
                key = words[0].strip().lower()
                value = words[1].strip()
                config[key] = value
                
        det_name = config['detector']
        det_config = json.loads(config['detector_config'])
        if "orientation" in config['detector_config']:
            det_config.pop("orientation", None)
        det = Detector.factory(det_name, det_config)
        return cls(det, 
                   float(config['distance']), 
                   float(config['poni1']), 
                   float(config['poni2']), 
                   float(config['rot1']), 
                   float(config['rot2']),
                   float(config['rot3']),
                   float(config['wavelength']))
    
    @classmethod
    def from_dict(cls, config):
        det_name = config['detector']
        det_config = {}
        if 'detector_config' in config:
            det_config = json.loads(config['detector_config'])
            if "orientation" in config['detector_config']:
                det_config.pop("orientation", None)
        det = Detector.factory(det_name, det_config)
        return cls(det, 
                   float(config['distance']), 
                   float(config['poni1']), 
                   float(config['poni2']), 
                   float(config['rot1']), 
                   float(config['rot2']),
                   float(config['rot3']),
                   float(config['wavelength']))

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


def transform(poni, p1, p2, p3):
    rot = rotation_matrix(poni)
    pos = np.dot(rot, np.vstack((p1.reshape(-1), p2.reshape(-1), p3.reshape(-1))))
    r = np.sqrt(pos[0]**2 + pos[1]**2)
    tth = np.arctan2(r, pos[2])
    phi = np.arctan2(pos[0], pos[1])
    return tth, phi


def setup_radial_bins(poni, radial_bins, unit, tth):
    if unit == 'q':
        # calculate auto range min/max radial bins
        if not isinstance(radial_bins, Iterable):
            # q = 4pi/lambda sin( 2theta / 2 ) in A-1
            q = 4.0e-10 * np.pi / poni.wavelength * np.sin(0.5*tth)
            radial_bins = np.linspace(np.amin(q), np.amax(q), radial_bins+1)
        else:
            radial_bins = np.asarray(radial_bins)
                
    elif unit == '2th':
        # radial bins in 2th are in deg
        if not isinstance(radial_bins, Iterable):
            radial_bins = np.rad2deg(np.linspace(np.amin(tth), np.amax(tth), radial_bins+1))
        else:
            radial_bins = np.asarray(radial_bins)
                
    return radial_bins
    
    
def setup_azimuth_bins(azimuth_bins):
    if azimuth_bins is None:
        return None
    
    if not isinstance(azimuth_bins, Iterable):
        azimuth_bins = np.linspace(0, 360, azimuth_bins+1)
    else:
        azimuth_bins = np.asarray(azimuth_bins)
    return azimuth_bins

    
def setup_corrections(poni, solid_angle, polarization_factor, p1, p2, tth, phi):
    corrections = np.ones(p1.size, dtype=np.float32)
    if solid_angle:
        solid_angle = poni.dist / np.sqrt(poni.dist**2 + p1*p1 + p2*p2)
        corrections *= (solid_angle**3).reshape(-1)
            
    if not polarization_factor is None:
        cos2_tth = np.cos(tth) ** 2
        polarization = 0.5 * (1.0 + cos2_tth - polarization_factor * np.cos(2.0 * phi) * (1.0 - cos2_tth))
        corrections *= polarization.reshape(-1)
        
    return corrections


class AzimuthalIntegrator():
    """
    This class is an azimuthal integrator 
    """
    def __init__(self,
                 poni: Union[str, Poni],
                 n_splitting: int, 
                 radial_bins: Union[int, Sequence],
                 azimuth_bins: Optional[Union[int, Sequence]] = None,
                 unit: str = 'q',
                 mask: Optional[Union[np.ndarray, str]] = None,
                 solid_angle: bool = True,
                 polarization_factor: Optional[float] = None,
                 error_model: Optional[str] = None):
        """
        Args:
            poni: Name of Poni file or instance of Poni
            n_splitting: Each pixel in the image gets split into (n, n) subpixels that get binned individually
            radial_bins: radial bins can either be number of bins or a sequence defining the bin edges in Angstrom^-1.
            azimuth_bins: azimthual bins can either be number of bins or a sequence defining the bin edges between [0, 360] degrees.
            unit: Ouput units for the radial coordindate
            mask: Pixel mask to exclude bad pixels. Pixels marked with 1 will be excluded
            solid_angle: Perform solid angle correction
            polarization_factor: Polarization factor for the polarization correction
                1 (linear horizontal polarization)
                -1 (linear vertical polarization)
            error_model: Error model used to calculate errors in the transformation. Only options is 'poisson' error model
            
        Attributes:
            radial_axis (ndarray): radial axis depeding on units in q or 2theta
            azimuth_axis (ndarray, optional): azimuth axis in degrees is case of 2D integration
        """
        self.poni = poni
        self.n_splitting = n_splitting
        self.radial_bins = radial_bins
        self.azimuth_bins = azimuth_bins
        self.solid_angle = solid_angle
        self.polarization_factor = polarization_factor
        self.mask_path = ''
        if not isinstance(mask, np.ndarray):
            if mask == '':
                mask = None
            else:
                fname = mask
                self.mask_path = fname
                ending = os.path.splitext(fname)[1]
                if ending == '.npy':
                    mask = np.load(fname)
                else:
                    mask = fabio.open(fname).data
        self.mask = mask
        if error_model and error_model != 'poisson':
            raise RuntimeError('Only poisson error model is supported')
        
        if unit not in ('q', '2th'):
            raise RuntimeError('Wrong radial unit. Allowed units: q, 2th')
        
        if isinstance(poni, str):
            poni = Poni.from_file(poni)
        
        if isinstance(poni, dict):
            poni = Poni.from_dict(poni)
        
        self.unit = unit
        self.error_model = error_model
        
        pixel_centers = np.mean(poni.det.pixel_corners, axis=2)
        p1 = pixel_centers[..., 1] - poni.poni1
        p2 = pixel_centers[..., 2] - poni.poni2
        p3 = pixel_centers[..., 0] + poni.dist
        tth, phi = transform(poni, p1, p2, p3)
        
        radial_bins = setup_radial_bins(poni, radial_bins, unit, tth)
        self.radial_axis = 0.5*(radial_bins[1:] + radial_bins[:-1])
        azimuth_bins = setup_azimuth_bins(azimuth_bins)
        self.azimuth_axis = 0.5*(azimuth_bins[1:] + azimuth_bins[:-1]) if azimuth_bins is not None else None
        
        shape = pixel_centers.shape[:2]
        self.input_size = np.prod(shape)
        if mask is None:
            mask = np.zeros(shape, dtype=np.int8)
        
        if azimuth_bins is None:
            self.output_shape = [len(radial_bins) - 1]
        else:
            self.output_shape = [len(azimuth_bins) - 1, len(radial_bins) - 1]
        
        self.sparse_matrix = Sparse(poni, poni.det.pixel_corners, n_splitting, 
                                    mask, unit, radial_bins, azimuth_bins)
        self.norm = self.sparse_matrix.spmv(np.ones(shape[0]*shape[1], dtype=np.float32))
        corrections = setup_corrections(poni, solid_angle, polarization_factor, p1, p2, tth, phi)
        self.sparse_matrix.set_correction(corrections)
        
        
    def integrate(self, 
                  img: np.ndarray, 
                  normalized: Optional[bool] = True) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Calculates the azimuthal integration of the input image
        
        Args:
            img: Input image to be integrated
            mask: Optional pixel mask to exclude bad pixels. Note if mask is constant using the mask argument in
                the constructor is more efficient
            normalized: Whether to return the normalized result or the integrated signal and norm separately
        
        Returns:
            azimuthal integrated image or just integrated signal if normalized is True
            standard error of the mean (SEM) when error_model is specified else None
            the norm if normalized is False
        """
        img = np.ascontiguousarray(img)
        
        if img.size != self.input_size:
            raise RuntimeError('Size of image is wrong!\nExpected %d\nActual size %d' %(self.input_size, img.size))
        if self.mask is None:
            norm = self.norm
        else:
            inverted_mask = 1 - self.mask
            img = img*inverted_mask
            norm = self.sparse_matrix.spmv(inverted_mask.reshape(-1))
                
        signal = self.sparse_matrix.spmv_corrected(img).reshape(self.output_shape)
        norm = norm.reshape(self.output_shape)
        
        errors = None
        if self.error_model:
            # poisson error model
            errors = np.sqrt(self.sparse_matrix.spmv_corrected2(img)).reshape(self.output_shape)
            if normalized:
                errors = np.divide(errors, norm, out=np.zeros_like(errors), where=norm!=0.0)
        
        if normalized:
            result = np.divide(signal, norm, out=np.zeros_like(signal), where=norm!=0.0)
            return result, errors
        else:
            return signal, errors, norm
        
