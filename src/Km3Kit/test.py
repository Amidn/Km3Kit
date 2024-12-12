from Km3Kit import KM3NetIRFGenerator


# Provide the necessary file names and optional weight factor
filename_nu =    "/sps/km3net/users/amid/production/arca21/Coords_Merged/v2/mcv8.1.all.gsg_numu-CCHEDIS_1e2-1e8GeV.sirene.jterbr.jchain.aashower.dst.bdt_casc.bdt_trk.root"
filename_nubar = "/sps/km3net/users/amid/production/arca21/Coords_Merged/v2/mcv8.1.all.gsg_anumu-CCHEDIS_1e2-1e8GeV.sirene.jterbr.jchain.aashower.dst.bdt_casc.bdt_trk.root"
filename_mu =    "/sps/km3net/users/amid/production/arca21/Coords_Merged/v2/mcv8.1.all.mupage_tuned_100G.sirene.jterbr.jchain.aashower.dst.bdt_casc.bdt_trk.root"
save_dir = "/"

# Create an instance of KM3NetIRFGenerator
irf_generator = KM3NetIRFGenerator(
    filename_nu=filename_nu,
    filename_nubar=filename_nubar,
    filename_mu=filename_mu,
    save_dir = save_dir
)

# Call the create method on the instance
irf_generator.create()