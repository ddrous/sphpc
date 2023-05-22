"""
					2D learning hierarchy 
   
"""


T = 15				#number of time steps in integration (prediction step)
# coarse_mult = 1;  #coarse graining in time (number of dts to skip)
n_itrs = 1000			#number of iteration
vis_rate = 1;		#sampling frequency for output
lr = 5e-2 			#initial lr (later adapted with ADAM)
mag = 1.0			#Amplitude of external forcing
r = 5.0;			#number of smoothing (r*hkde) lengths for determining bounds of integration in KL
h_kde = 0.9;	    #smoothing parameter h; using simpsons rule;
nb = "all";			# number of samples in batching
n_int = 220;  	    #number of integration points in numerical computation of KL
t_start = 1;        # time just after stationarity is reached
window = 0.2;
height = 10;
t_decay = int(0.6*n_itrs);			#iteration decay of lr begins