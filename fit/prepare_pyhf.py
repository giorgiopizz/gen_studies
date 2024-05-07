import itertools
import json
import os
import numpy as np
import uproot
import hist

# ops = ["cWtil"]
op = "cWtil"
ops = ["cWtil", "cHWtil", "cHGtil", "cHWBtil", "cHBtil", "ctWIm", "ctBIm"]
ops = [
    "ctBIm",
    "cHtbIm",
    "cbBIm",
    "cHWtil",
    "cQtQb8Im",
    "ctWIm",
    "cHGtil",
    "cHBtil",
    "cHWBtil",
    "cQtQb1Im",
    "cbWIm",
    "cWtil",
    "cbHIm",
]

variables = ["mjj", "detajj", "dphijj", "ptj1", "events"]

input = uproot.open("../analysis/histos.root")
# variables = list(set(list(map(lambda k: k.split("/")[0].split(";")[0], input.keys()))))
# print(variables)

for variable in variables:
    channels = []
    observations = []
    channel_dict = {
        "name": "test",
        "samples": [],
    }
    data_dict = {
        "name": "test",
        "data": 0,
    }
    # os.makedirs(f"datacards/{variable}", exist_ok=True)
    # output = uproot.recreate(f"datacards/{variable}/histos.root")
    print(input[variable].keys())
    process_names = [
        "sm",
    ]
    for op in ops:
        process_names += [
            f"quad_{op}",
            f"lin_{op}",
        ]
    for couple in list(itertools.combinations(ops, 2)):
        op1 = couple[0]
        op2 = couple[1]
        if f"histo_mixed_{op1}_{op2};1" not in input[variable].keys():
            print("not", op1, op2)
            op1 = couple[1]
            op2 = couple[0]
        process_names += [f"mixed_{op1}_{op2}"]
    # process_idx = [i for i in range(len(process_names))]
    # process_rates = [-1 for i in range(len(process_names))]
    for key in process_names:
        sample = {
            "name": key,
            "data": 0,
            "modifiers": [],
        }
        h = input[variable][f"histo_{key}"].to_hist()
        vals = h.values()
        sample["data"] = vals
        # if isinstance(data_dict['data'], int):
        if key == "sm":
            data_dict["data"] = vals.copy()

        sample["modifiers"].append(
            {
                "name": "lumi",
                "type": "normsys",
                "data": {
                    "lo": 0.98,
                    "hi": 1.02,
                },
            }
        )

        # # if key == 'sm' or 'lin' in key:
        # # stat_err = np.sqrt(h.variances()) / vals
        # neff = np.float32(np.int32(np.power(vals, 2) / h.variances()))
        # stat_err = np.power(neff, -1/2)
        # print('Effective number of events', np.power(stat_err, -2))
        # sample["modifiers"].append(
        #     {
        #         "name": "Stat_" + key,
        #         "type": "shapesys",
        #         "data": stat_err,
        #     }
        # )

        if key == 'sm':
            stat_err = np.sqrt(h.variances())
            sample["modifiers"].append(
                {
                    "name": "Stat",
                    "type": "staterror",
                    "data": stat_err,
                }
            )

        if key == "sm":
            sample["modifiers"].append(
                {
                    "name": "mu",
                    "type": "normfactor",
                    "data": None,
                }
            )
        if "lin" in key:
            sample["modifiers"].append(
                {
                    "name": key[len("lin_") :],
                    "type": "eftlin",
                    "data": None,
                }
            )
        elif "quad" in key:
            sample["modifiers"].append(
                {
                    "name": key[len("quad_") :],
                    "type": "eftquad",
                    "data": None,
                }
            )
        elif "mix" in key:
            _ops = key[len("mixed_") :].split("_")
            for op in _ops:
                sample["modifiers"].append(
                    {
                        "name": op,
                        "type": "eftlin",
                        "data": None,
                    }
                )
        channel_dict["samples"].append(sample)

    channels.append(channel_dict)
    observations.append(data_dict)

    workspace = {
        "channels": channels,
        "observations": observations,
        "measurements": [
            {"name": "Measurement", "config": {"poi": op, "parameters": []}}
        ],
        "version": "1.0.0",
    }

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    print(json.dumps(workspace, cls=json_serialize))
    os.makedirs("cards", exist_ok=True)
    with open(f"cards/{variable}.json", "w") as file:
        json.dump(workspace, file, indent=2, cls=json_serialize)


# # Create empty data
# h = input[variable]["histo_sm"].to_hist().copy()
# a = h.view(True)
# a.value = np.zeros_like(a.value)
# a.variance = np.zeros_like(a.variance)
# output["histo_Data"] = h
