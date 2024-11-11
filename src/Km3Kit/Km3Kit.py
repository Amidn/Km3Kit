"""Main module."""
from IO.io import Read

DATA, NEUTRINO, MUON = Read("arca21_dst", verbose=True)
#DATA, NEUTRINO, MUON = Read("arca21_bdt", verbose=True)
# Example: Access data from "branch1"
branch1_data = DATA["E"]

# Create a histogram of branch1 data (for example)
#plt.hist(branch1_data, bins=50)
#plt.xlabel("Branch1 Values")
#plt.ylabel("Frequency")
#plt.title("Histogram of Branch1")
#plt.show()