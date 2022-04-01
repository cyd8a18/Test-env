import pathlib


PATH_ur5 = pathlib.Path(__file__).parent.resolve()
folder_URDF= PATH_ur5 / "urdf/objects/table.urdf"
print(type(folder_URDF))
print(str(folder_URDF))