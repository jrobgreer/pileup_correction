import numpy as np
import matplotlib.pyplot as plt
#import uproot
import ROOT

# Figure out what events look like from COMPASS - load in here
# Would also be good to have an option to load in from Wavedump using Rob's GDW code

# Open the ROOT file
root_file = ROOT.TFile("TIMESTAMPDATA/RAW/DataR_TIMESTAMPDATA.root", "READ")

# Access the TTree or TBranch containing the TArray
tree = root_file.Get('Data_R;1')
# print(dir(tree))
# print(tree.GetListOfLeaves())

for leaf in tree.GetListOfLeaves():
    print(leaf.GetName())

tree.GetEntry(1)
print(tree.Energy)
times = []
energies = []

for i in range(10000):
    tree.GetEntry(i)
    times.append(tree.Timestamp)
    energies.append(tree.Energy)

times = np.array(times)
energies = np.array(energies)
plt.scatter(times[energies<10], energies[energies<10])
plt.xlabel("Time")
plt.ylabel("Energy")
# print(times)
# print(np.max(times))
plt.show()
plt.close()

plt.xlabel("Timestamp")
plt.ylabel("Counts")
plt.hist(times[energies<10], bins=100)
plt.show()