from Km3Kit import KM3NetIRFGenerator


# Provide the necessary file names and optional weight factor
filename_nu = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_numuCC.ALL.dst.bdt.root'
filename_nubar = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_anumuCC.ALL.dst.bdt.root'
filename_mu10 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_10T.sirene_mupage.ALL.bdt.root'  
filename_mu50 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_50T.sirene_mupage.ALL.bdt.root'
save_dir = "/"

# Create an instance of KM3NetIRFGenerator
irf_generator = KM3NetIRFGenerator(
    filename_nu=filename_nu,
    filename_nubar=filename_nubar,
    filename_mu10=filename_mu10,
    filename_mu50=filename_mu50,
    save_dir = save_dir
)

# Call the create method on the instance
irf_generator.create()