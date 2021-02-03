import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
from scipy.fftpack import dct, idct
from PIL import Image as im
import cvxpy as cvx
from simulate_1 import sim
from skimage.metrics import structural_similarity as ssim
from quality_index import indexes
from solve_matrix_tool import sov_tool
import cmath
import math
from sympy import *  

dir_path = r"D:\YRG_project\reconstruction_images\data\meas_201108\data_201108.npy"
dir_path_0 = r"D:\YRG_project\reconstruction_images\data\meas_201108\mask_idx_201108.npy"
dir_path_1 = r"D:\YRG_project\reconstruction_images\data\used_mask_reshaped.npy"

data = np.load(dir_path)
mask_array = np.load(dir_path_1)
mask_idx = np.load(dir_path_0)
mask_array_withoutG = mask_array[mask_idx[:], :]
st = sov_tool()
G_matrix = np.array(st.gaussian_mask(0.0, 14.0, (32, 32))).flat
#print(*G_matrix)
mask_array_withG = np.ones((len(mask_array_withoutG), 1024))
mask_array_withG[:] = mask_array_withoutG[:] * G_matrix

sim = sim()
index = indexes()

def generate_image(A, y):

    # do L1 optimization
    vx = cvx.Variable(32 * 32)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A @ vx == y]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat2 = np.array(vx.value).squeeze()

    # reconstruct signal
    img = idct2(Xat2.reshape(32, 32))
    #img = Xat2.reshape(32, 32)

    return img

def dct2(x): return dct(dct(x.T, norm = 'ortho', axis = 0, type=1).T, norm = 'ortho', axis = 0, type=1)
def idct2(x): return idct(idct(x.T, norm = 'ortho', axis = 0, type=1).T, norm = 'ortho', axis = 0, type=1)

def L1_noG(A, data, img_arr):
    
    L1_img_noG = generate_image(A, data)
    #L1_img_noG = np.clip(L1_img_noG, 0, 1)
    plt.imshow(L1_img_noG, cmap='gray')
    plt.axis("off")
    plt.savefig(r"D:\YRG_project\reconstruction_images\img_201108\measured\L1_noG_201108.png")
    plt.show()
    #np.save(r"C:\Users\user\AppData\Roaming\SPB_Data\data\simple_pattern_test\simple_pattern1_NoMis_L1_noG_using_scaled_idx_201115.npy", np.array(L1_img_noG))
    L1_img_noG = np.reshape(L1_img_noG, 1024)
    L1_img_noG = index.normalize(L1_img_noG)
    L1_img_noG = np.reshape(L1_img_noG, (32, 32))
    print("SSIM_L1_noG=", round(ssim(L1_img_noG, img_arr, data_range = 1.0), 6))
    print("MSE_L1_noG=", round(index.compute_mse(L1_img_noG, img_arr), 3))

    
def L1_withG(A_G, data, img_arr):
        
    L1_img_withG = generate_image(A_G, data)
    plt.imshow(L1_img_withG, cmap='gray')
    plt.axis("off")
    plt.savefig(r"D:\YRG_project\reconstruction_images\img_201108\measured\L1_withG_201108.png")
    plt.show()
    #np.save(r"C:\Users\user\AppData\Roaming\SPB_Data\data\simple_pattern_test\simple_pattern1_NoMis_L1_withG_using_scaled_idx_201115.npy", np.array(L1_img_withG))
    L1_img_withG = np.reshape(L1_img_withG, 1024)
    L1_img_withG = index.normalize(L1_img_withG)
    L1_img_withG = np.reshape(L1_img_withG, (32, 32))
    print("SSIM_L1_withG=", round(ssim(L1_img_withG, img_arr, data_range = 1.0), 6))
    print("MSE_L1_withG=", round(index.compute_mse(L1_img_withG, img_arr), 3))

def OMP_noG(A, data, img_arr):
    
    OMP_img_noG = st.omp(A, data, 50)
    plt.imshow(OMP_img_noG, cmap = 'gray')
    plt.axis("off")
    plt.savefig(r"D:\YRG_project\reconstruction_images\img_201108\measured\OMP_noG_201108.png")
    plt.show()
    #np.save(r"C:\Users\user\AppData\Roaming\SPB_Data\data\simple_pattern_test\simple_pattern1_NoMis_OMP_noG_using_scaled_idx_201115.npy", np.array(OMP_img_noG))
    OMP_img_noG = np.reshape(OMP_img_noG, 1024)
    OMP_img_noG = index.normalize(OMP_img_noG)
    OMP_img_noG = np.reshape(OMP_img_noG, (32, 32))
    print("SSIM_OMP_noG=", round(ssim(OMP_img_noG, img_arr, data_range = 1.0), 6))
    print("MSE_OMP_noG=", round(index.compute_mse(OMP_img_noG, img_arr), 3))
    
    
def OMP_withG(A_G, data, img_arr):

    OMP_img_withG = st.omp(A_G, data, 50)
    plt.imshow(OMP_img_withG, cmap = 'gray')
    plt.axis("off")
    plt.savefig(r"D:\YRG_project\reconstruction_images\img_201108\measured\OMP_withG_201108.png")
    plt.show()
    #np.save(r"C:\Users\user\AppData\Roaming\SPB_Data\data\simple_pattern_test\simple_pattern1_NoMis_OMP_withG_using_scaled_idx_201115.npy", np.array(OMP_img_withG))
    OMP_img_withG = np.reshape(OMP_img_withG, 1024)
    OMP_img_withG = index.normalize(OMP_img_withG)
    OMP_img_withG = np.reshape(OMP_img_withG, (32, 32))
    print("SSIM_OMP_withG=", round(ssim(OMP_img_withG, img_arr, data_range = 1.0), 6))
    print("MSE_OMP_withG=", round(index.compute_mse(OMP_img_withG, img_arr), 3))
    
def IFD(recon_image, lamda, d):
    
    image_withIFD = np.ones((recon_image.shape[0], recon_image.shape[1]))
    k = 2 * math.pi / lamda
    x, y = symbols("x y")
    jkd = complex(0, k * d)                 # j(k*d)
    jld = complex(0, lamda * d)             # j(lamda * d)
    jk_div_2d = complex(0, k / 2 / d)       # j(k/2d)
    
    for i in range(recon_image.shape[0]):
        for j in range(recon_image.shape[1]):
            image_withIFD[i][j] = -exp(-jkd)/jld * integrate(recon_image[i][j] * exp(-jk_div_2d) * ((x - i)^2 + (y - j)^2), (x, -oo, oo), (y, -oo, oo))
            
    plt.imshow(OMP_img_withG, cmap = 'gray')
    return image_withIFD
    
def main():
    #L1
    DCT = np.kron(
    spfft.dct(np.identity(32), norm='ortho', axis=0, type=1),
    spfft.dct(np.identity(32), norm='ortho', axis=0, type=1)
    )
    
    #DCT = np.kron(
    #spfft.dct(np.eye(32), norm='ortho', axis=0),
    #spfft.dct(np.eye(32), norm='ortho', axis=0)
    #)
    

    A = mask_array_withoutG
    A = A @ DCT
    A_G = mask_array_withG @ DCT
    

    img_arr = sim.pattern_choose(1)                 # change different patterns
    L1_noG(A, data, img_arr)
    L1_withG(A_G, data, img_arr)
    OMP_noG(A, data, img_arr)
    OMP_withG(A_G, data, img_arr)
    

if __name__ == '__main__':
    main()




