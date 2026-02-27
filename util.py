import torch

DEFAULT_SPEC_CONFIG = {
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'center': True,
    'pad_mode': 'reflect',
    'onesided': True,
    'length': 32000
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
        self.length = cfg['length']
    
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
        # X_stft = torch.view_as_real(X_stft)
        return X_stft

    def istft(self, X):
        X_istft = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=X.device),
            center=self.center,
            normalized=False,
            onesided=self.onesided,
            length=self.length
        )
        return X_istft
    
def _istft(spec: torch.Tensor) -> torch.Tensor:
    """ 
        spec: (B, F, T, 2)
        out : (B, 1?, 32000)
    """
    LEN_OUT = 32000
    device = spec.device
    spec = torch.complex(spec[...,0], spec[...,1]).to(device)
    out = torch.istft(
        spec,
        512,
        256,
        512,
        window=torch.hann_window(512).pow(0.5).to(device),
        length=LEN_OUT
    )
    return out

def _stft(signal: torch.Tensor) -> torch.Tensor:
    """ 
        signal: (B, 1?, 32000)
        out   : (B, F, T)  complex
    """
    
    device = signal.device

    out = torch.stft(
        signal,
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5).to(device),
        return_complex=True
    )
    return out