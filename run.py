"""
Script to train and evaluate the neural net.
Set the parameters here.
"""
import loader
import ann
import ann2

trd, vd, td = loader.load_data_wrapper()
print len(trd)
print trd[0][0]
# print trd[:2]
# print len(trd), len(vd)
net = ann2.Network([784,100,10])
net.stGradientDescent(trd, 60, 10, 0.5, 5.0, vd)
#score = net.evaluate(vd)
