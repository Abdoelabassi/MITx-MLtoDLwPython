import ROOT
import matplotlib.pyplot as plt
import numpy as np



def func_1():
	f = ROOT.TF1("f", "1/(1+exp(-x))", -10, 10)
	c = ROOT.TCanvas("c","my canvas", 500,500)
	f.SetLineColor(ROOT.kBlue)
	f.SetLineWidth(2)
	f.Draw()
	c.Update()
	c.SaveAs("function_1.png")

	ROOT.gSystem.ProcessEvents()

def length_2Dvector(x,y):
	a = np.sqrt(x*x + y*y)
	return a

def angle_vectors(vec_1, vec_2,x,y,u,v):

	a = length_2Dvector(x,y)
	b = length_2Dvector(u,v)
	angle = np.arccos(np.dot(vec_1, vec_2)/a*b)
	return angle
x= 0.4
y=0.3


def distance_from_2Dplane(x,theta, theta_0):

	d = (np.dot(x,theta) + theta_0) / (np.sqrt(theta[0]**2 + theta|1]**2)
	return d

if __name__ == "__main__":

	angle = angle_vectors(np.array([0.4,0.3]), np.array([-0.15,0.2]), 0.4,0.3,-0.15,0.2)
	print(angle)
