import skipthoughts
import play
import os
path_to_file="/Users/wheatwaves/deeplearning/skip-thoughts/data/data/sentencesOfPureText/"
path_to_save_file="/Users/wheatwaves/deeplearning/skip-thoughts/data/data/sentencesOfPureText/"
g=os.walk(path_to_file)
names=[]
for root, dirs, files in g:
	names=files
model=skipthoughts.load_model()
for name in names[1:]:
	try:
		f=open(path_to_file+name)
		s=[line.strip().decode('utf-8') for line in f.readlines()]
		f.close()
		M=skipthoughts.encode(model,s)
		scores=[0]*len(s)
	except:
		f=open(path_to_file+name)
		s=[line.strip() for line in f.readlines()]
		f.close()
		M=skipthoughts.encode(model,s)
		scores=[0]*len(s)
	for i in xrange(len(M)):
		for j in xrange(len(M)):
			if i!=j:
				scores[i]+=play.cos_similarity(M[i],M[j])
	f=open(path_to_save_file+name+'.scores','w')
	for num in scores:
		f.write(str(num)+'\n')
	f.close()