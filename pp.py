import pickle

mydata = pickle.load(open("mydata/val/101.pkl",'rb'))

print(mydata['node_feat'].shape)
print(mydata['edge_feat'].shape)
print(mydata['edge_index'].shape)
print(mydata['inverse_edge_index'].shape)
print(mydata['density_feat'].shape)