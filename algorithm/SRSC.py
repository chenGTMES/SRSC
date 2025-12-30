from utils.utils import *
from utils.FRSGM.condrefinenet_fourier import CondRefineNetDilated

class SRSC:
    def __init__(self, max_iter=50, SR=True, SC=False, DP=False):

        print('=====================================')
        print('=============== SRSC+ ===============')
        print('=====================================')
        import main
        self.max_iter, self.save_path = max_iter, main.save_path
        self.T_mask, self.T_ksdata, self.T_ksfull = main.mask, main.ksdata, main.ksfull
        self.T_Ker, self.T_Ker_Tra, self.T_sensitivity = main.Ker, main.Ker_Tra, main.sensitivityLi

        self.SR, self.SC, self.DP = SR, SC, DP

        self.FS = lambda x: FFT2_3D_N(self.T_sensitivity * x.unsqueeze(-1))
        self.FST = lambda x: torch.sum(torch.conj(self.T_sensitivity) * IFFT2_3D_N(x), dim=-1)
        self.A = lambda xx: VidHaarDec3S(IFFT2_3D_N((1 - self.T_mask) * self.FS(xx)))
        self.AT = lambda xx: self.FST((1 - self.T_mask) * FFT2_3D_N(VidHaarRec3S(xx)))

        self.rho, self.delta = 1.999, 0.999 / 1.999
        self.index = [1, 2, 3, 4, 6, 7, 8, 9]
        self.targetSen = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scoreNet = CondRefineNetDilated(2, 2, 64).to(device)
        self.scoreNet.load_state_dict(torch.load('./utils/checkpoint/net.pth'))
        self.scoreNet.eval()
        self.gamma = 1.3
        self.lam = 0.006

        self.ref, self.start_time = sos(IFFT2_3D_N(self.T_ksfull)), time.time() - main.t_kernel

    def process(self, Thr=0):
        c = VidHaarDec3S(IFFT2_3D_N(self.T_ksdata))
        s = torch.zeros_like(c)
        x = sos(IFFT2_3D_N(self.T_ksdata))
        v = x

        for iter in range(self.max_iter):
            if (iter + 1) % int(self.max_iter / 5) == 0 or iter == 0:
                print(f"At iteration {iter + 1}, err = {torch.norm(torch.abs(self.ref - abs(x))):.4f}")
                if iter != 0 and self.SC:
                    self.updateSensitivity(x)

            gradf = self.T_mask * self.FS(x) - self.T_ksdata
            gradfF = ((self.FST(gradf.real)).real - (self.FST(gradf.imag)).imag)
            t = s - self.delta * self.A(self.rho * self.AT(s) - (v.conj() - self.rho * gradfF)) + self.delta * c

            # set regularization parameter
            if iter in range(0, self.max_iter, 5):
                if self.DP:
                    Thr = abs(VidHaarDec3S(sos_T((DenoiseByDiffusion(self, x)).real, self.T_sensitivity)))
                else:
                    Thr = abs(VidHaarDec3S(sos_T(x,self.T_sensitivity)))
                Thr = imfilter_symmetric_4D(Thr[..., self.index])
                Thr = EnergyScaling_4D(1 / Thr, t[..., self.index])

            s[..., self.index] = t[..., self.index] - wthresh(t[..., self.index], Thr)
            v = x - self.rho * gradfF - self.rho * self.AT(s)
            x = v.real

        useTime = time.time() - self.start_time
        modelName = 'SR' + ''.join(['SC' if self.SC else '', '+' if self.DP else ''])
        print(f"{modelName} Elapsed Time: {useTime:.2f} seconds")
        return PSNR_SSIM_HaarPSI(self.ref, abs(x), modelName, self.save_path, useTime)

    def updateSensitivity(self, u, max_iter=5):
        if self.targetSen is None:
            self.targetSen = Kernel_Rec_ks_C_I_Pro(self.T_ksdata, self.T_Ker)
            self.targetSen = Kernel_Rec_ks_C_I_Pro(self.targetSen, self.T_Ker_Tra)
            self.targetSen = IFFT2_3D_N((1 - self.T_mask) * self.targetSen)

        x = self.T_sensitivity * abs(u).unsqueeze(-1)
        x = SolverForSubProblem(lambda xx: self.SensitivityFunction(xx), x - self.targetSen, x, iter=max_iter)

        multiCoils = IFFT2_3D_N((1 - self.T_mask) * FFT2_3D_N(x) + self.T_ksdata)
        sliceImage = sos(multiCoils) + torch.finfo(torch.float32).eps
        self.T_sensitivity = multiCoils / sliceImage.unsqueeze(-1)

        return sliceImage

    def SensitivityFunction(self, x):
        s = Kernel_Rec_ks_C_I_Pro((1 - self.T_mask) * FFT2_3D_N(x), self.T_Ker)
        s = IFFT2_3D_N((1 - self.T_mask) * Kernel_Rec_ks_C_I_Pro(s, self.T_Ker_Tra))
        return s + x

    
