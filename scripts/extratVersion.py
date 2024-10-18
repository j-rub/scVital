#made with copilot

import toml

def get_version():
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)
    return pyproject["project"]["version"]

if __name__ == "__main__":
    print(get_version())