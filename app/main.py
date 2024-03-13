from data import Orchestrator

DEVICE_MAPPING = {
    "Appliance1":"Fridge",
    "Appliance2":"Chest Freezer",
    "Appliance3":"Upright Freezer",
    "Appliance4":"Tumble Dryer",
    "Appliance5":"Washing Machine",
    "Appliance6":"Dishwasher",
    "Appliance7":"Computer Site",
    "Appliance8":"Television Site",
    "Appliance9":"Electric Heater"
}


if __name__ == "__main__":
    print("Starting...")

    orchestrator = Orchestrator.Orchestrator(device = "Fridge", device_mapping=DEVICE_MAPPING)
    orchestrator.run()
