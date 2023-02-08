import numpy as np
import matplotlib.pyplot as plt
import ROOT


f = ROOT.TF1("f", "-e^(-x)",0,10)

c = ROOT.TCanvas("c", "mycanvas", 500,500)

f.Draw()

c.Update()

c.SaveAs("negexp.png")
