import requests
import json

class PlantAPI:

    def __init__(self):
        # https://trefle.io/api/v1/plants/<ID>/
        # https://trefle.io/api/v1/plants/search?token=tmRZ6jLbqmFmGITDKOZ_MsWIOLiPDQp7AEc5UunWbN8&q=Strawberry&range[atmospheric_humidity]=0&page=1
        self.OUTPUT_FILE = './plant_archive.json'

    def get_plant_growth_by_name(self, name):
        with open(self.OUTPUT_FILE) as f:
            plants = json.load(f)
            plant = plants[name]
            growth = plant['data']['main_species']['growth']
            return growth


plant_api = PlantAPI()
result = plant_api.get_plant_growth_by_name('Strawberry')
print(json.dumps(result, indent=4))

