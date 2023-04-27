
import numpy as np
import torch as th
from numpy.fft import irfft, rfftfreq
from .distributions import SquashedDiagGaussianDistribution


def powerlaw_psd_gaussian(exponent, size, fmin=0, rng=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    rng : np.random.Generator, optional
        Random number generator (for reproducibility). If not passed, a new
        random number generator is created by calling
        `np.random.default_rng()`.


    Returns
    -------
    out : array
        The samples.

    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)    # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    if rng is None:
        rng = np.random.default_rng()
    sr = rng.normal(scale=s_scale, size=size)
    si = rng.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)    # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)    # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


class ColoredNoiseProcess():
    """Infinite colored noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the colored noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, beta, size, scale=1, max_period=None, rng=None):
        """Infinite colored noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        beta : float
            Exponent of colored noise power-law spectrum.
        size : int or tuple of int
            Shape of the sampled colored noise signals. The last dimension (`size[-1]`) specifies the time range, and
            is thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled colored noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        self.beta = beta
        if max_period is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / max_period
        self.scale = scale
        self.rng = rng

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[-1]

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        """Reset the buffer with a new time series."""
        self.buffer = powerlaw_psd_gaussian(
                exponent=self.beta, size=self.size, fmin=self.minimum_frequency, rng=self.rng)
        self.idx = 0

    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]


class PinkNoiseProcess(ColoredNoiseProcess):
    """Infinite pink noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the pink noise process.
    reset()
        Reset the buffer with a new time series.
    """
    def __init__(self, size, scale=1, max_period=None, rng=None):
        """Infinite pink noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        size : int or tuple of int
            Shape of the sampled pink noise signals. The last dimension (`size[-1]`) specifies the time range, and is
            thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled pink noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, size, scale, max_period, rng)


class ColoredNoiseDist(SquashedDiagGaussianDistribution):
    def __init__(self, beta, seq_len, action_dim=None, rng=None, epsilon=1e-6, device='cpu'):
        """
        Gaussian colored noise distribution for using colored action noise with stochastic policies.

        The colored noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        self.device = device
        assert (action_dim is not None) == np.isscalar(beta), \
            "`action_dim` has to be specified if and only if `beta` is a scalar."

        if np.isscalar(beta):
            super().__init__(action_dim, epsilon)
            self.beta = beta
            self.gen = ColoredNoiseProcess(beta=self.beta, size=(action_dim, seq_len), rng=rng)
        else:
            super().__init__(len(beta), epsilon)
            self.beta = np.asarray(beta)
            self.gen = [ColoredNoiseProcess(beta=b, size=seq_len, rng=rng) for b in self.beta]

    def sample(self) -> th.Tensor:
        if np.isscalar(self.beta):
            cn_sample = th.tensor(self.gen.sample()).float().to(self.device)
        else:
            cn_sample = th.tensor([cnp.sample() for cnp in self.gen]).float().to(self.device)
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev*cn_sample
        return th.tanh(self.gaussian_actions)

    def __repr__(self) -> str:
        return f"ColoredNoiseDist(beta={self.beta})"


class PinkNoiseDist(ColoredNoiseDist):
    def __init__(self, seq_len, action_dim, rng=None, epsilon=1e-6, device='cpu'):
        """
        Gaussian pink noise distribution for using pink action noise with stochastic policies.

        The pink noise is only used for sampling actions. In all other respects, this class acts like its parent
        class (`SquashedDiagGaussianDistribution`).

        Parameters
        ----------
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        epsilon : float, optional, by default 1e-6
            A small value to avoid NaN due to numerical imprecision.
        """
        super().__init__(1, seq_len, action_dim, rng, epsilon, device)
