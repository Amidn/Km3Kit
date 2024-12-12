from collections import namedtuple
import logging
import uproot

class DST:
    log = logging.getLogger("offline")

    # Directly include mc_header here
    mc_header = {
        "DAQ": ["livetime"],
        "seed": ["program", "level", "iseed"],
        "PM1_type_area": ["type", "area", "TTS"],
        "PDF": ["i1", "i2"],
        "model": ["interaction", "muon", "scattering", "numberOfEnergyBins"],
        "can": ["zmin", "zmax", "r"],
        "genvol": ["zmin", "zmax", "r", "volume", "numberOfEvents"],
        "merge": ["time", "gain"],
        "coord_origin": ["x", "y", "z"],
        "translate": ["x", "y", "z"],
        "genhencut": ["gDir", "Emin"],
        "k40": ["rate", "time"],
        "norma": ["primaryFlux", "numberOfPrimaries"],
        "livetime": ["numberOfSeconds", "errorOfSeconds"],
        "flux": ["type", "key", "file_1", "file_2"],
        "spectrum": ["alpha"],
        "fixedcan": ["xcenter", "ycenter", "zmin", "zmax", "radius"],
        "start_run": ["run_id"],
    }

    for key in "cut_primary cut_seamuon cut_in cut_nu".split():
        mc_header[key] = ["Emin", "Emax", "cosTmin", "cosTmax"]

    for key in "generator physics simul".split():
        mc_header[key] = ["program", "version", "date", "time"]

    @staticmethod
    def to_num(value):
        """Convert a value to a numerical one if possible"""
        for converter in (int, float):
            try:
                return converter(value)
            except (ValueError, TypeError):
                pass
        return value

    class Header:
        """The header"""

        def __init__(self, header, mc_header, log):
            self._data = {}
            self._mc_header = mc_header
            self._log = log

            for attribute, fields in header.items():
                values = fields.split()
                field_list = self._mc_header.get(attribute, [])

                n_values = len(values)
                n_fields = len(field_list)

                if n_values == 1 and n_fields == 0:
                    entry = DST.to_num(values[0])
                    self._data[attribute] = entry
                    if attribute.isidentifier():
                        setattr(self, attribute, entry)
                    continue

                n_max = max(n_values, n_fields)
                values += [None] * (n_max - n_values)
                field_list += ["field_{}".format(i) for i in range(n_fields, n_max)]

                if not values:
                    continue

                cls_name = attribute if attribute.isidentifier() else "HeaderEntry"
                entry = namedtuple(cls_name, field_list)(*[DST.to_num(v) for v in values])

                self._data[attribute] = entry

                if attribute.isidentifier():
                    setattr(self, attribute, entry)
                else:
                    self._log.warning(
                        f"Invalid attribute name for header entry '{attribute}'"
                        ", access only as dictionary key."
                    )

        def __dir__(self):
            return list(self.keys())

        def __str__(self):
            lines = ["MC Header:"]
            keys = set(self._mc_header.keys())
            for key, value in self._data.items():
                if key in keys:
                    lines.append(f"  {value}")
                else:
                    lines.append(f"  {key}: {value}")
            return "\n".join(lines)

        def __getitem__(self, key):
            return self._data[key]

        def keys(self):
            return self._data.keys()

        def items(self):
            return self._data.items()

        def values(self):
            return self._data.values()

    def __init__(self, file_path):
        self.file_path = file_path

    def head(self):
        """
        Load the header from the ROOT file.
        
        Returns
        -------
        Header
            The Header instance populated with data from the ROOT file.
        """
        with uproot.open(self.file_path) as f:
            if "Head" in f:
                head_data = f["Head"].tojson()["map<string,string>"]
                return self.Header(head_data, self.mc_header, self.log)
            else:
                raise ValueError("The 'Head' branch is not present in the ROOT file.")