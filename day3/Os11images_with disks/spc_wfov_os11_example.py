# This script by John Krist (JPL)
# Copyright (c) 2022 by California Institute of Technology
# All rights reserved
# The author and CalTech do not guarantee support or the results of this script
 
import numpy as np
import astropy.io.fits as pyfits
from emccd_detect.emccd_detect import EMCCDDetectBase
from PhotonCount.corr_photon_count import get_count_rate

def spc_wfov_os11_example():

    # e-/sec (pre-gain) images at frame temporal sampling 
    suffix = '.fits'
    input_file = 'spc_wfov_os11_frames_with_disk' + suffix    

    # EMCCD parameters
 
    full_well_serial = 100000.0
    full_well = 60000.0
    dark_rate = 8.33e-4
    cic_noise = 0.02
    read_noise = 100.0
    bias = 0
    cr_rate = 0                    
    qe = 1.0                            # QE is already applied in input images
    pixel_pitch = 13e-6               
    e_per_dn = 1.0                      
    nbits = 14                          # ADC bits
    numel_gain_register = 604           

    # read noiseless input images as 3D array; they are sampled in time according
    # to the chosen frame time for each star and are in units of e-/sec (pre-gain)
  
    input_images, h = pyfits.getdata(input_file, header = True, ignore_missing_end = True)
    nimages = input_images.shape[0]
    ny = input_images.shape[1]
    nx = input_images.shape[2]

    # read parameters for each batch

    data = np.loadtxt('spc_wfov_os11_batch_info.txt', skiprows=2)

    batch = data[:,0].astype(int)
    start_time_h = data[:,1]
    star = data[:,2].astype(int)
    roll = data[:,3]
    frame_exptime_sec = data[:,4]
    gain = data[:,5]
    nframes = data[:,6].astype(int)
    nbatch = batch.shape[0]

    batch_flux_maps = np.zeros( (nbatch,ny,nx), dtype='float32' )
    true_batch_flux_maps = np.zeros( (nbatch,ny,nx), dtype='float32' )

    istart = 0 
    ref_image = 0
    n_ref = 0
    targ_roll_minus_image = 0
    n_targ_roll_minus = 0
    targ_roll_plus_image = 0
    n_targ_roll_plus = 0

    # process each batch separately

    for ibatch in range(nbatch):
        print( "Processing batch "+str(batch[ibatch]) )
        iend = istart + nframes[ibatch] 

        batch_images = input_images[istart:iend,:,:]
        batch_gain = gain[ibatch]
        batch_frametime_sec = frame_exptime_sec[ibatch]

        # add EMCCD noise

        emccd = EMCCDDetectBase( em_gain=batch_gain, full_well_image=full_well, 
                            full_well_serial=full_well_serial, dark_current=dark_rate, 
                            cic=cic_noise, read_noise=read_noise, bias=bias,
                            qe=qe, cr_rate=cr_rate, pixel_pitch=pixel_pitch, 
                            eperdn=e_per_dn, numel_gain_register=numel_gain_register, nbits=nbits )

        noisy_batch_images = np.zeros( batch_images.shape, dtype='float32' )

        print("  batch_frametime_sec = "+str(batch_frametime_sec))
        print("  batch gain = "+str(batch_gain))
 
        for iframe in range(nframes[ibatch]):
            noisy_batch_images[iframe,:,:] = emccd.sim_sub_frame( batch_images[iframe,:,:], batch_frametime_sec ).astype('float32')

        # do photon counting on high-gain frames and return mean flux expectation;
        # otherwise, assume analog imaging and just divide by gain; either way, result is 
        # in e-/sec (pre-gain)

        if batch_gain > 1000:   # image is intended to be photon counted
            threshold = read_noise * 5  # values above this are considered one photon
            image = get_count_rate( noisy_batch_images, threshold, batch_gain, niter=32 ) / batch_frametime_sec
        else:
            image = np.mean( noisy_batch_images, 0 ) / batch_gain / batch_frametime_sec

        if star[ibatch] == 1:
            if roll[ibatch] < 0:
                targ_roll_minus_image += image
                n_targ_roll_minus += 1
            else:
                targ_roll_plus_image += image
                n_targ_roll_plus += 1
        else:
            ref_image += image
            n_ref += 1

        batch_flux_maps[ibatch,:,:] = image
        true_batch_flux_maps[ibatch,:,:] = np.mean( batch_images, 0 )

        istart = iend 

    ref_image /= n_ref
    targ_roll_minus_image /= n_targ_roll_minus
    targ_roll_plus_image /= n_targ_roll_plus

    output_file = 'spc_wfov_os11_flux_maps' + suffix             # combined e-/sec (pre-gain), photon-counted images for each batch
    pyfits.writeto( output_file, batch_flux_maps, overwrite=True )

    truth_output_file = 'spc_wfov_os11_true_flux_maps' + suffix  # mean of input frames for each batch; e-/sec (pre-gain)
    pyfits.writeto( truth_output_file, true_batch_flux_maps, overwrite=True )

    # read in original time series to get reference and target fluxes

    hdu = pyfits.open( "spc_wfov_os11_polx_darkhole_noiseless" + suffix )
    hdr = hdu[0].header
    ref_flux = hdr['REFFLUX']
    targ_flux = hdr['TARGFLUX']
    hdu.close()

    # normalize images by predicted stellar flux and do brute-force RDI

    rdi_roll_minus_image = targ_roll_minus_image - ref_image / ref_flux * targ_flux
    rdi_roll_plus_image = targ_roll_plus_image - ref_image / ref_flux * targ_flux

    # write out results; values are e-/sec (pre-gain) 

    pyfits.writeto( "example_reference_image"+suffix, ref_image, overwrite=True )
    pyfits.writeto( "example_target_roll_minus_image"+suffix, targ_roll_minus_image, overwrite=True )
    pyfits.writeto( "example_target_roll_plus_image"+suffix, targ_roll_plus_image, overwrite=True )
    pyfits.writeto( "example_rdi_roll_minus_image"+suffix, rdi_roll_minus_image, overwrite=True )
    pyfits.writeto( "example_rdi_roll_plus_image"+suffix, rdi_roll_plus_image, overwrite=True )

if __name__ == '__main__':
    spc_wfov_os11_example()
