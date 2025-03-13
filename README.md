# raisin

`raisin` is a lightweight tool for estimating electron doping in single-layer graphene (SLG) and bilayer graphene (BLG) 
samples.
Its only inputs are the Raman spectrum (assumed to be taken at 514 nm)  in `.csv` format and the number of layers in the 
sample.
'raisin' fits the G peak with a fano lineshape and the 2D peak with a lorentzian function (or 4 lorentzians in the case 
of BLG). It then uses the area ratio A(2D)/A(G), peak positions Pos(G) and Pos(2D) and width FWHM(G) to estimate the 
electron doping based on the experimental data of Das *et al.*, Nat. Nano. Lett. **3** (2008).