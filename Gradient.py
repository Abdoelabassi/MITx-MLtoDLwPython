import ROOT

f=ROOT.TF2("f","x**2+y**2",-5,5,-5,5)
c=ROOT.TCanvas("c","mycanvas", 500,500)

f.Draw("surf2")

c.Update()

c.SaveAs("2dfunc.png")
