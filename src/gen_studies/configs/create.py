import os
import shutil
import sys

import gen_studies.configs.config as config


def main():
    if len(sys.argv) != 2:
        print("Should provide analysis name")
        sys.exit(1)
    analysis_name = sys.argv[1]
    path = os.path.abspath(config.__file__)
    try:
        os.makedirs(analysis_name, exist_ok=False)
    except Exception as _:
        raise Exception(
            f"Could not create folder for analysis name: '{analysis_name}'"
            + "\nMaybe its already here?"
        )

    shutil.copyfile(path, f"{analysis_name}/config.py")
    print("Created folder", analysis_name, "with the config.py file")


if __name__ == "__main__":
    main()
