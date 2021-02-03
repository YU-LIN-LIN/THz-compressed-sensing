import numpy as np
from solve_matrix_tool import sov_tool
from random import randint 

def_mis = 3
def_snr = 40
def_pNum = 1
def_OMPsigma = 14.0

dir_path = r"D:\YRG_project\reconstruction_images\data\meas_201108\mask_idx_201108.npy"
dir_path_1 = r"D:\YRG_project\reconstruction_images\data\used_mask_reshaped.npy"

    
class sim:
    def __init__(self):
        pass

    def pattern_choose(self, num):
        img_arr = np.ones((32, 32))
        bar = np.zeros((1, 32))
        
        # test pattern
        if num == 0:
            bar_list = [0, 4, 8, 12, 16, 20, 24, 27, 29]
            img_arr[bar_list, :] = bar
        
        # test_pattern_1 11/08
        if num == 1:
            bar_list = [0, 2, 5, 10, 21, 26, 29, 31]
            img_arr[bar_list, :] = bar
            img_arr[:, bar_list] = bar.T
        
        # test_pattern_2 11/15
        if num == 2:
            bar_list = [7, 12, 15, 17, 20, 25]
            img_arr[bar_list, :] = bar
            img_arr[:, bar_list] = bar.T
    
        return img_arr
    
    def load_mask(self, idx_path, mask_path):
        mask_idx = np.load(idx_path)
        mask_all = np.load(mask_path)
        mask = mask_all[mask_idx, :]
        
        return mask
    
    def gen_misalign(self, idx_path, mask_path, pattern_num, mis_dis, gaus_sigma = 7.0):
        
        mask_idx = np.load(idx_path)
        mask_all = np.load(mask_path)
        mask = mask_all[mask_idx, :]
        
        mask = mask.reshape(len(mask_idx), 32, 32)
        mask = np.kron(mask[:], np.ones((10, 10)))
        mask = np.pad(mask, ((0, 0), (5, 5), (5, 5)), 'constant', constant_values = (1, 1))
        
        st = sov_tool()
        sigma = gaus_sigma * 10.0
        G_matrix = st.gaussian_mask(0.0, sigma, (320, 320))
        mask[:, 5:-5, 5:-5] = mask[:, 5:-5, 5:-5] * G_matrix
        mask_mis_G = np.ones((len(mask), 320, 320))
        
        for i in range(len(mask)) :
            ri_1 = randint(5 - mis_dis, 5 + mis_dis)
            ri_2 = randint(5 - mis_dis, 5 + mis_dis)
            mask_mis_G[i] = mask[i, (0 + ri_1) : (320 + ri_1), (0 + ri_2) : (320 + ri_2)] * G_matrix
        
        img_arr = sim.pattern_choose(pattern_num)
        img_arr = np.kron(img_arr, np.ones((10, 10)))
    
        data = np.ones(mask.shape[0])
        for i in range(mask.shape[0]) :
            data[i] = np.sum((mask_mis_G[i] * img_arr).flat)
            data[i] /= 100
        
        mis_dis = float(mis_dis / 10)
        savePath = f"D:\\YRG_project\\reconstruction_images\\data\\meas_201108\\misaligned\\data_mis_" + str(mis_dis) + "mm_noNoise_sim_201108.npy"
    
        np.save(savePath, np.array(data))
        
        sim.addNoise(savePath, def_snr)
        
    def addNoise(self, data_path, snr):
        data_ori = np.load(data_path)
        data_withNoise = np.ones(data_ori.shape[0])
        for i in range(data_ori.shape[0]):
            data_withNoise[i] = data_ori[i] * np.random.uniform(1, (1 + (1 / 10 ** (snr / 20))))
            
        data_path = data_path[:-4] + f"_snr_" + str(snr)
            
        np.save(data_path + r".npy", np.array(data_withNoise))
        
    def modulation_depth(self, )
        mask = sim.load_mask(dir_path, dir_path_1) 

            
    

if __name__ == "__main__":
    sim = sim()
    sim.gen_misalign(dir_path, dir_path_1, def_pNum, def_mis, def_OMPsigma)






