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
	c.SaveAs("plots/function_1.png")

	ROOT.gSystem.ProcessEvents()

def length_2Dvector(x,y):
	a = np.sqrt(x*x + y*y)
	return a

def angle_vectors(vec_1, vec_2,x,y,u,v):

	a = length_2Dvector(x,y)
	b = length_2Dvector(u,v)
	angle = np.arccos(np.dot(vec_1, vec_2)/a*b)
	return angle



def Perceptron(x:np.array, y:np.array):
	"""
	function that performs a perceptron algorithm, which is a linear classification algorithm
	---------
	Args:
		X:(np.array) a feature vecotr ( training points)
		Y:(np.array) labels
	--------
	Returns:
		theta:(np.array) normal vector
		theta_0: (np.array) offset (optional w/ offset included)
		mistakes: (int) number of mistakes the algorithm committed
	"""
	n = range(len(x))
	theta = np.array([0,0], dtype="float32")
	theta_0 = 0
	mistakes = 0
	for i in n:
		for j in n:
			if y[j]*np.dot(theta,x[j]) <= 0:
				theta += y[j]*x[j]
				theta_0 += y[j]
				mistakes += 1
	return theta, theta_0, mistakes


if __name__ == "__main__":

	x = np.array([[-1,0],[0,1]])
	y = np.array([1,1])
	theta, theta_0, mistakes = Perceptron(x,y)
	print(f"theta :{theta} \n theta_0: {theta_0} \n number of mistakes: {mistakes}")
