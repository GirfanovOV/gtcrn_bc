import torch

DEFAULT_SPEC_CONFIG = {
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'center': True,
    'pad_mode': 'reflect',
    'onesided': True
}

class spec_transformator():
    def __init__(self, spec_config={}):
        cfg = DEFAULT_SPEC_CONFIG
        cfg.update(spec_config)

        self.n_fft = cfg['n_fft']
        self.hop_length = cfg['hop_length']
        self.win_length = cfg['win_length']
        self.center = cfg['center']
        self.pad_mode = cfg['pad_mode']
        self.onesided = cfg['onesided']
        self.window = torch.hann_window(self.win_length)
    
    def stft(self, X):
        X_stft = torch.stft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=X.device, dtype=X.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=self.onesided,
            return_complex=True,
        )
        return X_stft

    def istft(self, X):
        X_istft = torch.istft(
            X,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=X.device, dtype=X.dtype),
            center=self.center,
            normalized=False,
            onesided=self.onesided,
            length=self.length
        )
        return X_istft