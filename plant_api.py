import requests
import json

from weather_api import WeatherAPI

class PlantAPI:

    def __init__(self):
        # https://trefle.io/api/v1/plants/<ID>/
        # https://trefle.io/api/v1/plants/search?token=tmRZ6jLbqmFmGITDKOZ_MsWIOLiPDQp7AEc5UunWbN8&q=Strawberry&range[atmospheric_humidity]=0&page=1
        self.OUTPUT_FILE = './plant_archive.json'

    def get_plant_by_name(self, name):
        with open(self.OUTPUT_FILE) as f:
            plants = json.load(f)
            plant = plants[name]
            return plant

    def calculate_plant_watering_needs(self, name):
        plant = self.get_plant_by_name(name)
        # Get water needs from "Waterwise" plant database
        water_needs_min_mm_year = float(plant['water_needs_min_mm'])
        water_needs_max_mm_year = float(plant['water_needs_max_mm'])
        water_needs_avg_mm_year = (water_needs_min_mm_year + water_needs_max_mm_year) / 2
        # Convert to average cm per week
        water_needs_avg_cm_year = water_needs_avg_mm_year / 10
        water_needs_avg_cm_week = water_needs_avg_cm_year / 52
        water_needs_avg_cm_day = water_needs_avg_cm_week / 7

        # Retrieve weekly forecast
        weather_api = WeatherAPI()
        weekly_forecast = weather_api.simulate_forecast()
        precip_cm_day = list(map(lambda daily_forecast: daily_forecast['precipAccumulation'], weekly_forecast))[0:7]

        # Calculate daily delta
        delta_cm_day = list(map(lambda daily_precip_cm_day: round(max(water_needs_avg_cm_day - daily_precip_cm_day, 0), 4), precip_cm_day))

        return water_needs_avg_cm_day, precip_cm_day, delta_cm_day


plant_api = PlantAPI()
water_needs_avg_cm_day, precip_cm_day, delta_cm_day = plant_api.calculate_plant_watering_needs('Apple')
print('Avg. plant water needs (cm/day):', water_needs_avg_cm_day)
print('Daily precip (cm/day):', precip_cm_day)
print('Daily delta (cm/day):', delta_cm_day)

