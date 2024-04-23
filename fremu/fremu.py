import torch
import torch.nn as nn
import pickle
import numpy as np
from scipy.interpolate import CubicSpline
import os
import camb

class BkANN(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(BkANN, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1)
        self.hidden_activation1 = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_activation1(x)
        x = self.output_layer(x)
        return x
    
class emulator:
    def __init__(self):
        
        module_path = os.path.dirname(__file__)
        cache_path = os.path.join(module_path, 'cache') 
        
        self.ks = np.load(os.path.join(cache_path, 'k.npy'))
        self.ks = self.ks[:250]
        
        self.scaler = None
        with open(os.path.join(cache_path,'scaler.pkl'), 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
        
        self.pc = {}
        self.mean = {}
        
        for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
            pc = np.load(os.path.join(cache_path,f'pc_{z:.1f}.npy'))
            self.mean[z] = pc[0, :]
            self.pc[z] = pc[1:, :]
        n_hidd = 650
        n_out = 30
        self.model0 = BkANN(7, n_hidd, n_out)
        self.model05 = BkANN(7, n_hidd, n_out)
        self.model1 = BkANN(7, n_hidd, n_out)
        self.model2 = BkANN(7, n_hidd, n_out)
        self.model3 = BkANN(7, n_hidd, n_out)
            
        self.model0.load_state_dict(torch.load(os.path.join(cache_path,'pc_ann_0.0.pth')))
        self.model05.load_state_dict(torch.load(os.path.join(cache_path,'pc_ann_0.5.pth')))
        self.model1.load_state_dict(torch.load(os.path.join(cache_path,'pc_ann_1.0.pth')))
        self.model2.load_state_dict(torch.load(os.path.join(cache_path,'pc_ann_2.0.pth')))
        self.model3.load_state_dict(torch.load(os.path.join(cache_path,'pc_ann_3.0.pth')))
        
        self.sigma = np.load(os.path.join(cache_path,'sigma.npy'))
        
    def get_error(self, k=None):      
        if k is None:
            return self.sigma
        sigma_interp = CubicSpline(self.ks,self.sigma)
        err = sigma_interp(k)
        return err

    def get_k_values(self):
        return self.ks

    def set_cosmo(self, Om=0.3, Ob=0.05, h=0.7, ns=1.0, sigma8=0.8, mnu=0.05, fR0=-3e-5, redshifts=[3.0,2.0,1.0,0.5,0.0]):
        self.Om = Om
        self.Ob = Ob
        self.h = h
        self.ns = ns
        self.sigma8 = sigma8
        self.mnu = mnu
        self.fR0 = fR0
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h * 100, ombh2=Ob * h**2, omch2=(Om-Ob) * h**2, mnu=0, omk=0, num_massive_neutrinos=3)
        pars.InitPower.set_params(ns=ns)
        #pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version='mead2020_feedback')
        get_transfer = True
        pars.set_matter_power(redshifts=[0], kmax=10.0, nonlinear=False)
        initial_scalar_amp = 1e-9
        pars.InitPower.set_params(As=initial_scalar_amp)
        results = camb.get_results(pars)
        sigma8_computed = results.get_sigma8_0()
        sigma8_desired = sigma8
        ratio = sigma8_desired / sigma8_computed
        adjustment_factor = ratio ** 2
        # 调整 scalar_amp(1) 的值
        new_scalar_amp = initial_scalar_amp * adjustment_factor
        pars.set_matter_power(redshifts=redshifts, kmax=10.0, nonlinear=True)
        pars.InitPower.set_params(As=new_scalar_amp)
        self.results = camb.get_results(pars)

    def get_boost(self, k=None, z=None, to_linear = False, return_k_values=False):
        if z is None:
            z = 0
            print('WARNING: No redshift value given, the default value is z=0.0')

        if k is None:
            k = self.ks
        if max(k) > max(self.ks) or min(k) < min(self.ks):
            raise ValueError('The k range should be between {} and {}'.format(min(self.ks), max(self.ks)))
        
        try:
            params = np.array([self.Om, self.Ob, self.h, self.ns, self.sigma8, self.mnu, self.fR0])
            X = params.reshape(1,-1)
            X = self.scaler.transform(X)
            X_tensor = torch.tensor(X, dtype=torch.float32)
        
            z_range = None
            for range_ in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]:
                if range_[0] <= z <= range_[1]:
                    z_range = range_
                    break
        
            if z_range is None:
                raise ValueError('Redshift should be between 0.0 and 3.0')
        
            pc0 = self.pc[0.0]
            pc05 = self.pc[0.5]
            pc1 = self.pc[1.0]
            pc2 = self.pc[2.0]
            pc3 = self.pc[3.0]
            mean0 = self.mean[0.0]
            mean05 = self.mean[0.5]
            mean1 = self.mean[1.0]
            mean2 = self.mean[2.0]
            mean3 = self.mean[3.0]
        
            with torch.no_grad():
                w_tensor0 = self.model0(X_tensor)
                w_tensor05 = self.model05(X_tensor)
                w_tensor1 = self.model1(X_tensor)
                w_tensor2 = self.model2(X_tensor)
                w_tensor3 = self.model3(X_tensor)
                
                Bk0 = np.dot(w_tensor0.numpy(), pc0) + mean0
                Bk05 = np.dot(w_tensor05.numpy(), pc05) + mean05
                Bk1 = np.dot(w_tensor1.numpy(), pc1) + mean1
                Bk2 = np.dot(w_tensor2.numpy(), pc2) + mean2
                Bk3 = np.dot(w_tensor3.numpy(), pc3) + mean3
                
            Bk = CubicSpline(np.array([0.0,0.5,1.0,2.0,3.0]),np.array([Bk0,Bk05,Bk1,Bk2,Bk3]),axis=0)
            Bk = Bk(z)
            Bk = Bk.reshape(-1, 1)
            Bk = Bk[:250]
            Bk_interp = CubicSpline(self.ks,Bk)
            bk = Bk_interp(k)
            bk = bk.ravel()
            if to_linear:               
                pk = self.results.get_matter_power_interpolator(nonlinear=True).P(z,k)
                pk_fid = pk.reshape(-1, 1)
                pk_mg =  pk_fid * bk
                pk_lin = self.results.get_matter_power_interpolator(nonlinear=False).P(z,k)
                bk = pk_mg/pk_lin
                

            if return_k_values:
                bk = np.column_stack((k, bk)) 

            return bk
        except ValueError as e:
            print(e) 

    def get_power_spectrum(self, k=None, z=None, return_k_values=False, get_fid=False):
        if z is None:
            z = 0
            print('WARNING: No redshift value given, the default value is z=0.0')

        if k is None:
            k = self.ks
        if max(k) > max(self.ks) or min(k) < min(self.ks):
            raise ValueError('The k range should be between {} and {}'.format(min(self.ks), max(self.ks)))
        
        try:
            params = np.array([self.Om, self.Ob, self.h, self.ns, self.sigma8, self.mnu, self.fR0])
            X = params.reshape(1,-1)
            X = self.scaler.transform(X)
            X_tensor = torch.tensor(X, dtype=torch.float32)
        
            z_range = None
            for range_ in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]:
                if range_[0] <= z <= range_[1]:
                    z_range = range_
                    break
        
            if z_range is None:
                raise ValueError('Redshift should be between 0.0 and 3.0')
        

            pc0 = self.pc[0.0]
            pc05 = self.pc[0.5]
            pc1 = self.pc[1.0]
            pc2 = self.pc[2.0]
            pc3 = self.pc[3.0]
            mean0 = self.mean[0.0]
            mean05 = self.mean[0.5]
            mean1 = self.mean[1.0]
            mean2 = self.mean[2.0]
            mean3 = self.mean[3.0]
        
            with torch.no_grad():
                w_tensor0 = self.model0(X_tensor)
                w_tensor05 = self.model05(X_tensor)
                w_tensor1 = self.model1(X_tensor)
                w_tensor2 = self.model2(X_tensor)
                w_tensor3 = self.model3(X_tensor)
                
                Bk0 = np.dot(w_tensor0.numpy(), pc0) + mean0
                Bk05 = np.dot(w_tensor05.numpy(), pc05) + mean05
                Bk1 = np.dot(w_tensor1.numpy(), pc1) + mean1
                Bk2 = np.dot(w_tensor2.numpy(), pc2) + mean2
                Bk3 = np.dot(w_tensor3.numpy(), pc3) + mean3
                
            Bk = CubicSpline(np.array([0.0,0.5,1.0,2.0,3.0]),np.array([Bk0,Bk05,Bk1,Bk2,Bk3]),axis=0)
            Bk = Bk(z)
            Bk = Bk.reshape(-1, 1)
            Bk = Bk[:250]
            Bk_interp = CubicSpline(self.ks,Bk)
            pk = self.results.get_matter_power_interpolator(nonlinear=True).P(z,k)
            pk_fid = pk.reshape(-1, 1)
            pk_mg =  pk_fid * Bk_interp(k)
            pk_mg = pk_mg.ravel()
            if return_k_values:
                pk_mg = np.column_stack((k, pk_mg))   
             
            if get_fid:
                return pk_fid.ravel() 
             
            return pk_mg
        except ValueError as e:
            print(e) 
