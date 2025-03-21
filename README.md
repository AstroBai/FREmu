<h1 align="center">
<img src="https://github.com/astrobai/fremu/blob/master/fremu.png" width="700">
</h1><br>

**=================================================================**

**Please notice: This is not the latest version of FREmu, to get the latest one, please contact me through e-mail: astrobaijc@gmail.com. The new version of FREmu can be extended to large scales, which is critical for real data analyses.**

**New Version Here: https://github.com/AstroBai/FREmus**

**=================================================================**

$\texttt{FREmu}$ offers fast and accurate predictions of non-linear power spectra for the specific Hu-Sawicki $f(R)$ gravity model. $\texttt{FREmu}$ employs principal component analysis and artificial neural networks to establish a parameter-to-power-spectrum map, with training data sourced from the Quijote-MG simulation suite. With a parameter space encompassing 7 dimensions, including $\Omega_m$, $\Omega_b$, $h$, $n_s$, $\sigma_8$, $M_{\nu}$, and $f_{R_0}$, the emulator achieves an accuracy better than 5% for the majority of cases and it's proved highly efficient for parameter constraints.

## Installation

You can install $\texttt{FREmu}$ via pip:

```
pip install fremu
```

## Usage

```python
from fremu import fremu

# Initialize the emulator
emu = fremu.emulator()

# Set cosmological parameters
emu.set_cosmo(Om=0.3, Ob=0.05, h=0.7, ns=1.0, sigma8=0.8, mnu=0.05, fR0=-1e-5)

# Get power spectrum
k_values = emu.get_k_values()
power_spectrum = emu.get_power_spectrum(k=k_values, z=0.5)

# Get boost factor
boost_factor = emu.get_boost(k=k_values, z=0.5)

```

## Documentation

You can find the documentation of the emulator here: [FREmu documentation](https://astrobai.github.io/codes/fremu).

## Contributing

If you find any issues or have any suggestions for improvement, feel free to submit an issue or pull request on GitHub.

