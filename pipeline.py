import os
import zmq
import json
import time
import h5py
import numpy as np
from multiprocessing import Process
from azint import AzimuthalIntegrator
from bitshuffle import decompress_lz4
import fabio 

def zmq_worker(poni_file, pixel_size, n_splitting, mask, bins):
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.connect('tcp://p-daq-cn-2:20001')
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:5550')
    
    ai = AzimuthalIntegrator(poni_file, mask.shape, pixel_size, n_splitting, mask, bins, create=False)
    while True:
        parts = pull_sock.recv_multipart(copy=False)
        header = json.loads(parts[0].bytes)
        #print(header)
        if header['htype'] == 'image':
            img = decompress_lz4(parts[1].buffer, header['shape'], np.dtype(header['type']))
            res = ai.integrate(img)
            header['type'] = 'float32'
            header['shape'] = res.shape
            header['compression'] = 'none'
            push_sock.send_json(header, flags=zmq.SNDMORE)
            push_sock.send(res)
        else:
            push_sock.send_json(header)
    ai.close()
            
            
def hdf5_worker(filename, worker_id, nworkers, poni_file, pixel_size, n_splitting, mask, bins):
    context = zmq.Context()
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:5550')
    fh = h5py.File(filename, 'r')
    dset = fh['/entry/measurement/Eiger/data']
    nimages = len(dset)
    
    ai = AzimuthalIntegrator(poni_file, mask.shape, pixel_size, n_splitting, mask, bins, create=False)
    
    if worker_id == 0:
        header = {'htype': 'header',
                  'msg_number': -1,
                  'filename': filename}
        push_sock.send_json(header)
    
    for i in range(worker_id, nimages, nworkers):
        img = dset[i]
        res = ai.integrate(img)
        header = {'htype': 'image',
                  'msg_number': i,
                  'type': 'float32',
                  'shape': res.shape,
                  'compression': 'none'}
        push_sock.send_json(header, flags=zmq.SNDMORE)
        push_sock.send(res)
        
     if worker_id == 0:
        header = {'htype': 'series_end',
                  'msg_number': nimages}
        push_sock.send_json(header)
    ai.close()
            
def ordered_recv(sock):
    cache = {}
    next_msg_number = 0
    while True:
        parts = sock.recv_multipart()
        header = json.loads(parts[0])
        #print('ordered_recv', header)
        msg_number = header['msg_number']
        if header['htype'] == 'header':
            next_msg_number = msg_number
        if msg_number == next_msg_number:
            yield msg_number, parts
            next_msg_number += 1
            while next_msg_number in cache:
                data = cache.pop(next_msg_number)
                yield next_msg_number, data
                next_msg_number += 1
        else:
            cache[msg_number] = parts
    

def collector(radial_bins, phi_bins):
    base_folder = '/data/visitors/nanomax/20200364/2020120208/process/radial_integration/'
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.bind('tcp://*:5550')
    fh = None
    dset = None
    for index, parts in ordered_recv(pull_sock):
        header = json.loads(parts[0])
        print(index, header)
        htype = header['htype']
        if htype == 'image':
            res = np.frombuffer(parts[1], header['type']).reshape(header['shape'])
            if fh:
                if not dset:
                    fh.create_dataset('q', data=0.5*(radial_bins[:-1] + radial_bins[1:]))
                    fh.create_dataset('phi', data=0.5*(phi_bins[:-1] + phi_bins[1:]))
                    dset = fh.create_dataset('cake', dtype=np.float32, 
                                               shape=(0, *res.shape), 
                                               maxshape=(None, *res.shape),
                                               chunks=(1, *res.shape))
                n = dset.shape[0]
                dset.resize(n+1, axis=0)
                dset[n] = res

        elif htype == 'header':
            filename = header['filename']
            if filename:
                path, fname = os.path.split(filename)
                output_folder = os.path.join(base_folder, path.split(os.sep)[-1])
                output_file = os.path.join(output_folder, fname)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                print(output_file)
                fh = h5py.File(output_file, 'w-')
            else:
                fh = None
            dset = None
                
        elif htype == 'series_end':
            print('end')
            if fh:
                fh.close()

if __name__ == '__main__':
    poni_file = '/data/visitors/nanomax/20200364/2020120208/process/Detector_calibration/Si_scan2.poni'
    pixel_size = 75.0e-6
    n_splitting = 4
    nworkers = 12
    mask = fabio.open('/data/visitors/nanomax/20200364/2020120208/process/Detector_calibration/mask_scan2.edf').data
    radial_bins = np.linspace(0.0, 38.44, 301)
    phi_bins = np.linspace(-np.pi, np.pi, 721)
    AzimuthalIntegrator(poni_file, mask.shape, pixel_size, n_splitting, mask, [radial_bins, phi_bins])

    procs = []
    for i in range(nworkers):
        p = Process(target=zmq_worker, args=(poni_file, pixel_size, n_splitting, mask, [radial_bins, phi_bins]))
        p.start()
        procs.append(p)
        
        
    collector(radial_bins, phi_bins)
        
    for i in range(nworkers):
        procs[i].join()

    ai.close()
