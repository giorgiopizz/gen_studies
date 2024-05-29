# import pylhe
import glob
import xml.etree.ElementTree as ET
from math import ceil

import awkward as ak
import uproot
import vector

lhe_file = "/gwpool/users/lmagnani/eventi2/EFT_LHE/gp_VBF_Z2_results/lhe/1877630_0.lhe"
lhe_file = "lhe_file.lhe"
lhe_file = "lhe/*lhe"
lhe_file = "/gwpool/users/santonellini/eft/gp_submission/lhe/*.lhe"
lhe_files = glob.glob(lhe_file)[:4]


vector_fields = [
    "px",
    "py",
    "pz",
    "E",
    # "x",
    # "y",
    # "z",
    # "E",
]


fieldnames = (
    [
        "id",
        "status",
        "mother1",
        "mother2",
        "color1",
        "color2",
    ]
    + vector_fields
    + [
        "m",
        "lifetime",
        "spin",
        "nparticles",
    ]
)


def get_columns(events):
    columns = []
    for column_base in ak.fields(events):
        column_suffixes = ak.fields(events[column_base])
        if len(column_suffixes) == 0:
            columns.append((column_base,))
        else:
            for column_suffix in column_suffixes:
                columns.append((column_base, column_suffix))
    return columns


chunksize = 100

tot_xs = []
right_xs = 0.20933649499999987

nbig_iterations = ceil(len(lhe_files) / chunksize)
for big_iteration in range(nbig_iterations):
    events_tot = 0

    start = big_iteration * chunksize
    stop = min((big_iteration + 1) * chunksize, len(lhe_files))

    for iteration, lhe_file in enumerate(lhe_files[start:stop]):
        neve = 0
        root_file = "root/" + lhe_file.split("/")[-1].replace(".lhe", ".root")

        initDict = {}
        for _, element in ET.iterparse(lhe_file, events=["end"]):
            if element.tag == "initrwgt":
                initDict["weightgroup"] = {}
                for child in element:
                    # Find all weightgroups
                    if child.tag == "weightgroup" and child.attrib != {}:
                        try:
                            wg_type = child.attrib["type"]
                        except KeyError:
                            try:
                                wg_type = child.attrib["name"]
                            except KeyError:
                                print(
                                    "weightgroup must have attribute 'type' or 'name'"
                                )
                                raise
                        _temp = {"attrib": child.attrib, "weights": {}}
                        # Iterate over all weights in this weightgroup
                        for w in child:
                            if w.tag != "weight":
                                continue
                            try:
                                wg_id = w.attrib["id"]
                            except KeyError:
                                print("weight must have attribute 'id'")
                                raise
                            txt = w.text
                            if txt:
                                txt = txt.strip()
                            _temp["weights"][wg_id] = {
                                "attrib": w.attrib,
                                "name": txt,
                            }

                        initDict["weightgroup"][wg_type] = _temp
                break

        if big_iteration == 0 and iteration == 0:
            print(initDict["weightgroup"]["mg_reweighting"]["weights"].keys())
            # for weight in initDict["weightgroup"]["mg_reweighting"]["weights"]:
            #     print(initDict["weightgroup"]["mg_reweighting"]["weights"][weight])

        weightgroups = initDict["weightgroup"]
        weight_tables = {
            "LHEReweightingWeight": "mg_reweighting",
            "LHEPdfWeight": "NNPDF31_nnlo_as_0118_mc_hessian_pdfas",
            "LHEScaleWeight": "Central scale variation",
        }

        builder = ak.ArrayBuilder()
        sumw = 0.0
        ievent = 0
        for event, element in ET.iterparse(lhe_file, events=["end"]):
            if element.tag == "event":
                with builder.record(name="event"):
                    data = element.text.strip().split("\n")
                    eventdata, particles = data[0], data[1:]
                    weight = float(eventdata.split()[2])
                    sumw += weight
                    builder.field("genWeight")
                    builder.real(weight)
                    nparticles = int(eventdata.split()[0])
                    particles = particles[:nparticles]

                    builder.field("Particle")
                    with builder.list():
                        for particle in particles:
                            with builder.record():
                                values = map(float, particle.split())
                                d = dict(zip(fieldnames, values))
                                # print({k:d[k] for k in vector_fields})
                                vec = vector.obj(**{k: d[k] for k in vector_fields})
                                d = [
                                    ("pt", vec.pt, "real"),
                                    ("eta", vec.eta, "real"),
                                    ("phi", vec.phi, "real"),
                                    ("mass", vec.mass, "real"),
                                    ("mass_orig", d["m"], "real"),
                                    ("pdgId", int(d["id"]), "integer"),
                                    ("status", int(d["status"]), "integer"),
                                ]
                                for fieldname, value, _type in d:
                                    # if _type == 'float':
                                    #     builder.field(fieldname).real(value)
                                    # elif _type == 'int':
                                    #     builder.field(fieldname).integer(value)
                                    getattr(builder.field(fieldname), _type)(value)
                                    # exec(f"builder.field(fieldname).{_type}(value)")
                    weights = {}
                    for sub in element:
                        if sub.tag == "rwgt":
                            for r in sub:
                                if r.tag == "wgt":
                                    # FIXME I'm normalizing the rwgts to the original weight
                                    weights[r.attrib["id"]] = (
                                        float(r.text.strip()) / weight
                                    )

                    for weight_table, weightgroup in weight_tables.items():
                        builder.field(weight_table)
                        with builder.list():
                            for weight_id in weightgroups[weightgroup]["weights"]:
                                builder.real(weights[weight_id])

                    ievent += 1
                    # if ievent >= 1000:
                    #     break

        print("xs", sumw)
        tot_xs.append([sumw, ievent])
        events = builder.snapshot()
        print("Done reading", len(events), "events")
        print(
            "Iteration",
            iteration + 1,
            "/",
            len(lhe_files[start:stop]),
            " big iter",
            big_iteration + 1,
            "/",
            nbig_iterations,
        )
        scale = sumw / right_xs
        events["genWeight"] = events["genWeight"] / (sumw / ievent) * scale
        print(events["genWeight"])

        if isinstance(events_tot, int):
            events_tot = ak.copy(events)
        else:
            events_tot = ak.concatenate([events_tot, events], axis=0)

    print("Writing", len(events_tot), "events")
    file = uproot.recreate(f"root/out_{big_iteration}.root")
    # file["Events"] = {"Particle": events.particles, "Weight": events.weights}
    file["Events"] = {k: events_tot[k] for k in ak.fields(events_tot)}
    file.close()

tot_xs_sum = 0.0
nevents_sum = 0
for xs, nevents in tot_xs:
    tot_xs_sum += xs * nevents
    nevents_sum += nevents
print("tot xs", tot_xs_sum / nevents_sum)
