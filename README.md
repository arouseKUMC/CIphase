# CIphase
This Matlab toolbox calculates the condition-independent phase (CIphi) of neural signals

Since the trial data contains corrective movements in addition to the large initial movements that were not precisely time aligned to trial events for averaging condition-independent neural activity, we developed a novel algorithm to iteratively average the firing rates, calculate CIphi, then average the firing rates again based on the CIphi. This iterative process involves three steps:   i) Each unit’s firing rate is averaged across all trials to determine its condition-independent firing rate.  ii) Dimensionality reduction is performed using PCA and jPCA on the condition-independent firing rates to identify the neural plane with the most rotational/cyclic condition-independent activity.  iii) The instantaneous phase is calculated using the Hilbert transform on the first two jPC dimensions for all data points.  A schematic of the iterative algorithm is shown in Figure A1.
	Trial averaging to identify condition-independent activity
The condition-independent neural activity is the average firing rate for each recorded spiking unit for all experimental trials regardless of the movement condition (ie target location).  For classic neurophysiology experiments, this can be calculated by averaging time-aligned data.  However, since trials were of varying durations and many included corrective movements, simple time alignment of trials was difficult.  Data was therefore aligned based on each time-point’s calculated CIphi rather than absolute time.  
Since CIphi depends on the averaged condition-independent activity and the condition-independent activity was averaged by CIphi alignment, an iterative approach was required.  An initial estimate using simple time-alignment averaging of all initial and corrective submovements was used for the first iteration.   For each subsequent iteration, the condition-independent firing rates for each unit were calculated by averaging all data points when the CIphi values were similar.  The average firing rate was estimated using a sliding window of CIphi values with a step size of p/50 and a window size of p/25 to generate 100 equally spaced samples ranging from -p to p.
	PCA and jPCA to identify rotational/cyclic neural activity
Next, the condition-independent firing rates were submitted to PCA and jPCA (Churchland et al., 2012) to identify the neural dimensions with the most cyclic activity.  PCA was performed on the high-dimensional neural space to reduce the condition-independent firing rates to the six dimensions with the most variance.  jPCA was then performed on this six-dimensional space.  jPCA is a dimensionality reduction technique to identify the neural planes with the most rotational activity and is more fully described in Churchland et al (2012).  Briefly, jPCA fits a first order dynamical system model to the neural activity :  			X ?=AX
to predict change in firing rate (X ?) based on the current firing rate (X).  The transform matrix of this model (A) can be separated into a symmetric matrix representing pure scaling and a skew-symmetric matrix representing pure rotational dynamics.  By taking the eigendecomposition of the skew-symmetric matrix, we obtain pairs of purely imaginary eigenvalues and corresponding eigenvectors that define planes of rotation in the neural space rank-ordered from greatest to least rotation.  In the present analysis, only the first plane with the greatest condition-independent rotation was used and we defined the two dimensions of the plane as CIx and CIy.   Additionally, to obtain a consistent CIphi across recording sessions, CIx was defined as the dimension in the CIx/CIy plane with the most variance.  The positive CIx direction was defined as having more positive than negative coefficients in the neural space which corresponds to the direction where more units have an increased firing rate.  Choosing this convention causes an increase in CIx to generally align with the onset of initial movement, since a majority of units increase firing rates at the onset of movement.
	Instantaneous phase estimate
Finally, the instantaneous phase were estimated by i) bidirectional bandpass filtering of the activity in both the CIx and CIy dimensions between 0.5-5 Hz with a 1st order Butterworth filter, ii) performing the Hilbert transform of both filtered signal (sx and sy) to generate a transformed signal (s ^_x  and s ^_y  ) that is a 90° phase shift of every Fourier component in the frequency domain to create an analytic representation of CIx and CIy, and iii) calculating the angle of the resulting analytic signal (s_1+is ^_1) to estimate the instantaneous phase in each dimension, and iv) which we then sum (with a p/2 phase shift added to ?y) to obtain a single instantaneous phase estimate for the neural activity within the plane which we call the condition-independent phase (CIphi).  

	

		
The bandpass filtering reduces the low-frequency drift and high-frequency variability in CIx to generate a more consistent subsequent phase estimate.  The condition-independent phase (CIphi) thus represents the instantaneous phase in the dimension of the neural space that has the most cyclic, condition-independent activity.  

Adam G. Rouse
arouse@kumc.edu
