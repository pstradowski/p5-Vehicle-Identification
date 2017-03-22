import pickle
from hog_utils import model_check


ppc =8
cpb = 2
orient = 9
nbins = 32
spatial_size = 16
cspace = 'HLS'
result = model_check(ppc = ppc, cpb = cpb, orient = orient, 
                            nbins = nbins, cspace = cspace, spat_size = spatial_size)
model_data = {'classifier': result['classifier'],
              'scaler': result['scaler'],
              'hog': {
                  'pixel_per_cell': ppc,
                  'celss_per_block': cpb,
                  'orientations': orient},
              'spatial_size': spatial_size,
              'hsit_bins': nbins,
              'color_space': cspace}
              

pickle.dump(model_data, open("model_data.p", "wb"))