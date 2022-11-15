from matplotlib import pyplot as plt
import numpy as np

def plot(matrix,weights=None,title="Prediction Matrix"):

	if len(matrix[0])==3: # if 1D inputs, excluding bias and ys 
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("i1")
		ax.set_ylabel("Classifications")

		if weights!=None:
			y_min=-0.1
			y_max=1.1
			x_min=0.0
			x_max=1.1
			y_res=0.001
			x_res=0.001
			ys=np.arange(y_min,y_max,y_res)
			xs=np.arange(x_min,x_max,x_res)
			zs=[]
			for cur_y in np.arange(y_min,y_max,y_res):
				for cur_x in np.arange(x_min,x_max,x_res):
					zs.append(predict([1.0,cur_x],weights))
			xs,ys=np.meshgrid(xs,ys)
			zs=np.array(zs)
			zs = zs.reshape(xs.shape)
			cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)
		
		c1_data=[[],[]]
		c0_data=[[],[]]

		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_y  = matrix[i][-1]

			if cur_y==1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(1.0)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(0.0)

		plt.xticks(np.arange(x_min,x_max,0.1))
		plt.yticks(np.arange(y_min,y_max,0.1))
		plt.xlim(0,1.05)
		plt.ylim(-0.05,1.05)

		c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
		c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

		plt.legend(fontsize=10,loc=1)
		plt.show()
		return

	if len(matrix[0])==4: # if 2D inputs, excluding bias and ys
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("i1")
		ax.set_ylabel("i2")

		if weights!=None:
			map_min=0.0
			map_max=5
			y_res=0.001
			x_res=0.001
			ys=np.arange(map_min,map_max,y_res)
			xs=np.arange(map_min,map_max,x_res)
			zs=[]
			for cur_y in np.arange(map_min,map_max,y_res):
				for cur_x in np.arange(map_min,map_max,x_res):
					zs.append(predict([1.0,cur_x,cur_y],weights))
			xs,ys=np.meshgrid(xs,ys)
			zs=np.array(zs)
			zs = zs.reshape(xs.shape)
			cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

		c1_data=[[],[]]
		c0_data=[[],[]]
		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_i2 = matrix[i][2]
			cur_y  = matrix[i][-1]
			if cur_y==1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(cur_i2)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(cur_i2)

		plt.xticks(np.arange(0.0,5,0.1))
		plt.yticks(np.arange(0.0,5,0.1))
		plt.xlim(0,5)
		plt.ylim(0,5)

		c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class 1')
		c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 0')

		plt.legend(fontsize=10,loc=1)
		plt.show()
		return
	
	print("Matrix dimensions not covered.")


# seed the pseudorandom number generator
def GenerateSyntheticFn1(UBx, LBx, N1, N2):
	x1=(UBx[0]-LBx[0])/2+np.random.rand(1,np.int16(N1)) *((UBx[0]-LBx[0])/2) # % -- Generate random point of the first dimension
	IDX=x1>UBx[0] 
	#Temp2=IDX[IDX==True]
	Temp=np.random.rand(len(IDX[IDX==True]),1)*0.5*(UBx[0]-LBx[0])
	Temp2=np.matlib.repmat(UBx[0],len(IDX[IDX==True]), 1)
	if (len(IDX[IDX==True])>0):
		x1[IDX]=Temp2-Temp

	# Second dimension
	x2=np.random.rand(1,np.int16(N1))*(UBx[1]-LBx[1]) # -- Generate random point of the first dimension        
	C1=np.transpose(x1)
	C1=np.hstack([C1,np.transpose(x2)])
	# The second class
	# First dimension
	x1=np.random.rand(1,np.int16(N2))*(UBx[0]-LBx[0])/2 # -- Generate random point of the first dimension
	# Second dimension
	x2=np.random.rand(1,np.int16(N2))*(UBx[1]-LBx[1])
	IDX=x1<0
	if (len(IDX[IDX==True])>0):
		Temp2=np.matlib.repmat(LBx[0], len(IDX[IDX==True]),1) 
		Temp=np.random.rand(len(IDX[IDX==True]),1)*0.5*(UBx[0]-LBx[0])
		x1[IDX]=Temp2-Temp

	C2=np.transpose(x1)
	C2=np.hstack([C2,np.transpose(x2)])
	#C3=np.c_[np.transpose(x1), np.transpose(x2)]
	Patterns=C1
	Patterns=np.vstack([Patterns,C2])
	Labels1=np.ones([np.int16(N1),1])
	Labels2=np.zeros([np.int16(N2),1])
	#Labels=[[Labels1], [Labels2]]
	Labels=np.r_[Labels1,Labels2]

	return Patterns, Labels

def accuracy(Matrix,weights):
	num_correct = 0.0
	preds       = []
	for i in range(len(Matrix)):
		pred   = predict(Matrix[i][:-1],weights) # get predicted classification
		#pred   = predict(Matrix[i],weights) # get predicted classification
		preds.append(pred)
		if pred==Matrix[i][-1]: num_correct+=1.0 
		#if pred==Matrix[i]: num_correct+=1.0 
	#print("Predictions:",preds)
	return num_correct/float(len(Matrix))

def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0

def UpdateWeights(Matrix,weights):
    for i in range(len(Matrix)):
        prediction = predict(Matrix[i][:-1],weights)
        Error = Matrix[i][-1]-prediction
        for j in range(len(weights)): 
			#print(f'Weights[{j}] is {weights[j]:.2f}')
            weights[j] = weights[j]+(l_rate*Error*Matrix[i][j]) 
			#print(f'Weights[{j}] is {weights[j]:.2f}')
    return weights
