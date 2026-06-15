from imgctrl.controller import Controller
from rich import print


if __name__ == "__main__":

    controller = Controller.from_dicom_directory("data/KiTS-00008")
    print(controller.get_patient_info())

    print(controller.list_atoms())