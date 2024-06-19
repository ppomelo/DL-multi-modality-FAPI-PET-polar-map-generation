import numpy as np
import nibabel as nib

from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass

from matplotlib import colors
import matplotlib.pyplot as plt

    
class PolarMap_PET():
    
    def __init__(self, PET_img, mask, start_slice, end_slice):
        
        self.PET_img = PET_img[:,:,start_slice:end_slice]
        self.mask = mask[:,:,start_slice:end_slice]

    def project_to_aha_polar_map(self, label=2):
       
        PET_img  = self.PET_img.copy()
        mask = self.mask.copy()

        PET_img[mask!=label] = 0
        #roll to center
        for nz in range (mask.shape[2]):
            # print(nz)
            cx, cy = center_of_mass(mask[:,:,nz]==label)[:2]
            # print(cx,cy)
            PET_img[:,:,nz] = self._roll_to_center(PET_img[:,:,nz], cx, cy)
            mask[:,:,nz] = self._roll_to_center(mask[:,:,nz], cx, cy)

        PET_img = PET_img.transpose((2,0,1))

        #project to polar
        V_proj = self._project_to_aha_polar_map(PET_img)
        results = {'V_proj':V_proj, 'mask':mask}

        return results
        
    def _project_to_aha_polar_map(self, E, nphi=360, nrad=30, dphi=10): # 360 100 1 #360 20 6
        
        nz = E.shape[0]
        angles = np.arange(0, nphi, dphi) 
        V      = np.zeros((nz, 360, 100))

        for rj in range(nz):

            PET_q  = self._inpter2(E[rj],k=10)
            PHI, R = self._polar_grid(*PET_q.shape) 
            PHI = PHI.ravel() 
            R   = R.ravel()

            for k, pmin in enumerate(angles):

                pmax = pmin + dphi
                # Get values for angle segment
                PHI_SEGMENT = (PHI>=pmin)&(PHI<=pmax)
                Rk   = R[PHI_SEGMENT] 
                Vk   = PET_q.ravel()[PHI_SEGMENT] 
    
                Rk = Rk[np.abs(Vk)!=0]
                Vk = Vk[np.abs(Vk)!=0] 

                if len(Vk) == 0:
                    continue 

                Rk = self._rescale_linear(Rk, rj, rj + 1)

                r = np.arange(rj, rj+1, 1.0/nrad)
                r = np.append(r,[rj+1])
                v=np.zeros((nrad,))

                for rr in range(nrad):
                    Rk_SEGMENT = (Rk>=r[rr])&(Rk<=r[rr+1])
                    if Rk_SEGMENT.max()==False:
                        # print(r[rr+1], pmin, rj, Rk.min(), Rk.max())
                        v[rr]=0
                        continue
                    v[rr] = np.mean(Vk[Rk_SEGMENT])

                for i in range(nrad):
                    const = 100//nrad
                    V[rj,dphi*k:dphi*(k+1),const*i:const*(i+1)] += v[i]

        return V        
        
        
    def construct_polar_map(self, tensor, start=30, stop=70, sigma=5):

        E  = tensor.copy()
        mu = E[:,:,start:stop].mean()

        nz = E.shape[0]
        E  = np.concatenate(np.array_split(E[:,:,start:stop], nz), axis=-1)[0] 
        E = np.stack(np.array_split(E,nz,axis=1))

        slices_left = E.shape[0]%4
        temp = slices_left//2
        num = (E.shape[0])//4
        E = [E[:num]] + [E[num:2*num+temp]] + [E[2*num+temp:3*num+temp]] + [E[3*num+temp:]] 
        sigma = 6

        E = [np.max(E[i], axis=0) for i in range(4)]
        E = np.concatenate(E, axis=1)

        E = gaussian_filter(E, sigma=sigma, mode='wrap')
        E = gaussian_filter(E, sigma=sigma, mode='wrap')

        mu = [mu] + self._get_17segments(E) 

        return E, mu 
    
    def _get_17segments(self, data):

        c1,c2,c3,c4 = np.array_split(data,4,axis=-1)
        c2 = np.roll(c2, 45, 0)

        c4 = [np.max(ci) for ci in np.array_split(c4,6,axis=0)]
        c3 = [np.max(ci) for ci in np.array_split(c3,6,axis=0)]
        c2 = [np.max(ci) for ci in np.array_split(c2,4,axis=0)]
        c1 = [np.max(c1)]

        c = c4 + c3 + c2 + c1
        # c = np.around(c/max(c),decimals=5)
        c = np.around(c,decimals=2)
        c = c.tolist()
        return c
    
    def _roll(self, x, rx, ry):
        x = np.roll(x, rx, axis=0)
        return np.roll(x, ry, axis=1)

    def _roll_to_center(self, x, cx, cy):
        nx, ny = x.shape[:2]
        return self._roll(x,  int(nx//2-cx), int(ny//2-cy))

    def _py_ang(self, v1, v2):
        """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.rad2deg(np.arctan2(sinang, cosang))

    def _polar_grid(self, nx=128, ny=128):
        x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(ny//2, -ny//2, ny))
        phi = (np.rad2deg(np.arctan2(y, x)))
        phi[np.where(phi<0)] = phi[np.where(phi<0)]+360
        # phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
        r    = np.sqrt(x**2+y**2+1e-8)
        return phi, r

    def _rescale_linear(self, array, new_min, new_max):
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max-new_min) / (maximum-minimum)
        b = new_min - m* minimum
        return m*array + b

    def _inpter2(self, Eij, k=2):
        nx, ny = Eij.shape
        
        x  = np.linspace(0,nx-1,nx)
        y  = np.linspace(0,ny-1,ny)
        xq = np.linspace(0,nx-1,nx*k)
        yq = np.linspace(0,ny-1,ny*k)
        
        f = interp2d(x,y,Eij,kind='linear')
        
        return f(xq,yq)    
        
    def _get_lv2rv_angle(self, mask):
        cx_lv, cy_lv = center_of_mass(mask>1)[:2]
        cx_rv, cy_rv = center_of_mass(mask==1)[:2]
        phi_angle    = self._py_ang([cx_rv-cx_lv, cy_rv-cy_lv], [0, 1])
        return phi_angle  