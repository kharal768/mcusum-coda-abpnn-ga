"""Check the MCUSUM simulation against Crosier (1988): h = 5.5, k = 0.5, two dimensions gives ARL0 of about 200."""
from simulate import arl0_mcusum
a, se = arl0_mcusum(2, 5.5, 0.5, n=20000, seed=1)
print("ARL0 = %.1f (SE %.1f)" % (a, se))
