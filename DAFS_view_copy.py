import pyFAI, h5py, numpy as np
import logging, time
import os.path

#file parameters:
name_of_experiment = "MAPbIBr_I05_50"
PONI_filename = "20191109_LaB6_guess_400mm_reactor_smallbeam.poni"
conversion_to_unit = "q_A^-1" #"2th_deg" #"q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm"
sardana_file = "20191110.h5"
sardana_scan_index = "8915"
data_channels = ["mono1_energy", "albaem01_ch3", "albaem01_ch4"]
image_shape = (1065,1030)

#azimutal integration parameters:
num_bins = 3000
dummy = 4294967295
delta_dummy = 1

#Energy dependence:
E_start = 12835.0           #start energy of scan in eV
E_step = 1.0                #setp energy in eV
number_of_points = 1201     #number of scan intervals + 1


from silx.gui.plot.ImageView import ImageViewMainWindow
from silx.gui import qt

from silx.gui.colors import Colormap

colormap = Colormap(name='inferno',             # Name of the colormap
                    normalization ='log',  # Either 'linear' or 'log'
                    vmin = 1,                # If not autoscale, data value to bind to min of colormap
                    #vmax=10e6               # If not autoscale, data value to bind to max of colormap
                    )                
            
def read_auxillary_data(filename, index, data_channel):
    auxillaryDataset = h5py.File(filename, 'r')["entry" + sardana_scan_index + "/measurement/" + data_channel]
    data = np.zeros(auxillaryDataset.shape[0], dtype='float64')
    auxillaryDataset.read_direct(data, np.s_[:]) #COPYS the dataset into the numpy array
    return data

def get_num_files_images(name_of_experiment):
    file_index = 1
    total_number_of_images = 0
    while os.path.exists(name_of_experiment + "_data_" + f"{file_index:06d}" + ".h5"):
        total_number_of_images += h5py.File(name_of_experiment + "_data_" + f"{file_index:06d}" + ".h5", 'r')['entry/data/data'].shape[0]    
        file_index += 1
    return (file_index - 1, total_number_of_images)

def create_plot2d():
    from silx.gui.plot import Plot2D
    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])
    plot = Plot2D()  # Create the plot widget
    plot.setAttribute(qt.Qt.WA_DeleteOnClose)
    plot.setKeepDataAspectRatio(False)  # To keep aspect ratio between X and Y axes
    plot.show() # Make the plot widget visible
    plot.setFocus(qt.Qt.OtherFocusReason)
    return plot

if __name__ == '__main__':
    print("running\n")
    #Acess to several "auxiliary" data and makes a copy
    energy_array = read_auxillary_data(sardana_file, sardana_scan_index, data_channels[0])
    i0_array = read_auxillary_data(sardana_file, sardana_scan_index, data_channels[1])
    iPIPS_array = read_auxillary_data(sardana_file, sardana_scan_index, data_channels[2])
    print(f"{len(energy_array):01d}" + " energy points found: "+ f"{energy_array[0]:10.1f}" + " eV ... " + f"{energy_array[len(energy_array)-1]:10.1f}" + " eV.")

    #Sets one azimutal inegrator and prints is parameters
    print("\nAzimutal Integration parameters:\n")
    filename = name_of_experiment + "_data_" + f"{0+1:06d}" + ".h5"
    ai = pyFAI.load(PONI_filename)
    ai.setChiDiscAtZero()
    print(ai)
    
    #sanity check
    #calculate number of files and TOTAL number of images
    number_of_files, total_number_of_images = get_num_files_images(name_of_experiment)
    print("\n" + f"{number_of_files:01d}" + " files found.")
    print("\n" + f"{total_number_of_images:01d}" + " images found.")
    
    #more sanity check
    if (total_number_of_images != len(energy_array)) :
        print("Number of images does not fit number of enegry points!")

    result = np.zeros((total_number_of_images,num_bins)) #result array where patterns are stored
    #Some pointers so that the loop doesnt have to create and destroy variables all the time:
    #first image
    data = h5py.File(filename, 'r')['entry/data/data'][np.s_[0,0::]] 
    #Why doesnt he use ai.create_mask ? -> is it because dummy only considers one value and not ">"?? DOUBT
    #its "homemade mask"
    mask = np.where(data > 4e8, 1, 0) # if a "data" value > 4e8, writes 1, otherwhise, writes 0
    #its azimutal integration with the "homemade mask
    DiffractionPattern = ai.integrate1d(data, num_bins, mask = mask, unit = conversion_to_unit) #perform azimutal integration
    #Dont undestand the why of this and why only for the first image with the"homemade mask" -> DOUBT!
    x_axis = DiffractionPattern[0] 
    x_start = DiffractionPattern[0][0]
    x_end = DiffractionPattern[0][num_bins-1]
    
    
    result_index = 0
    file_index = 1
    image_index = 1
    while(file_index <= number_of_files):
        for image in h5py.File(name_of_experiment + "_data_" + f"{file_index:06d}" + ".h5", 'r')['entry/data/data']: 
            #integration of one image
            t0 = time.time() 
            ai.wavelength = 1240 / energy_array[result_index] * 1E-9 
            mask = ai.create_mask(image, mask = None, dummy = dummy, delta_dummy = delta_dummy, mode = 'normal') # dummy (float) – value of dead pixels; delta_dumy – precision of dummy pixels
            DiffractionPattern = ai.integrate1d(image, num_bins, mask = mask, unit = conversion_to_unit) # azimuth_range=(175,185), #perform azimutal integration
            processTime = time.time() - t0 
            #write y of diffraction pattern to result, rebinned to correct x axis
            num_bins_index = 0
            while(num_bins_index < num_bins):
                result[result_index][num_bins_index] = np.interp(x_axis[num_bins_index], DiffractionPattern[0], DiffractionPattern[1])
                num_bins_index += 1
                
            print("%d: Azimutal integration performed for image %d in file %d in %.3fs. lambda=%e." % (result_index, image_index, file_index, processTime, ai.wavelength))
            result_index += 1
            image_index += 1
        
        file_index += 1
            
    #calculate header info
    xscale = (x_end - x_start) / num_bins
    xorigin = x_start
    
    #save integrated data to new file
    resultFile = h5py.File(name_of_experiment + ".h5", "w")
    
    # write header information
    resultInfoFolder = resultFile.create_group("info")
    resultInfoFolder.create_dataset("xscale", data = xscale)
    resultInfoFolder.create_dataset("xorigin", data = xorigin)

    #write data
    resultDataset = resultFile.create_dataset("rawresult", data = result)
    resultFile.create_dataset("i0", data = i0_array)
    resultFile.create_dataset("XAFS", data = iPIPS_array / i0_array)

    #create and save normalized dataset
    normresult = np.zeros((total_number_of_images,num_bins)) #result array where patterns are stored
    pattern = 0
    while(pattern < total_number_of_images):
        bin = 0
        while(bin < num_bins):
            normresult[pattern, bin] = result[pattern, bin] / (i0_array[pattern]*1e8)
            bin += 1
        pattern += 1
    
    resultDataset = resultFile.create_dataset("normresult", data = normresult)

    resultFile.close()
    
    #2D plot       
    plot = create_plot2d()
    plot.getXAxis().setLabel(conversion_to_unit) #Setup axes

    # Plot the 2D data set with default colormap
    plot.addImage(normresult, legend ='name_of_experiment', origin = (xorigin, 0), scale = (xscale,1), colormap = colormap)  

    app.exec_()