import pickle
from hog_utils import model_check


ppc =8
cpb = 3
orient = 9
nbins = 32
spatial_size = 16
cspace = 'HLS'

check = model_check(ppc = ppc, cpb = cpb, orient = orient, 
        nbins = nbins, spat_size = spatial_size, cspace = cspace)
pars = {'ppc': ppc, 'cpb': cpb, 'orient': orient, 'cspace': cspace,
        'spatial_size': spatial_size, 'nbins': nbins}
res = {**check, **pars}
pickle.dump(res, open("model_data.p", "wb"))