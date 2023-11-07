import requests
import json
import random


class WeatherAPI:

    def __init__(self):
        self.ACCESS_KEY = 'f0c7Frm9iWgnm43y'
        self.OUTPUT_FILE = './simulated_forecasts.txt'

    def live_forecast(self, latitude, longitude):
        params = {
            'exclude': 'current,minutely,hourly,alerts',
            'extend': '168',
            'units': 'si'
        }

        response = requests.get(f"https://api.pirateweather.net/forecast/{self.ACCESS_KEY}/{latitude},{longitude}", params)

        result = response.json()

        forecast_seven_days = result['daily']['data']

        return forecast_seven_days

    def simulate_forecast(self):
        with open(self.OUTPUT_FILE) as f:
            lines = f.readlines()

            random_line = random.choice(lines)

            result = json.loads(random_line)

            return result

    def generate_list_of_simulated_forecasts(self):
        cities = [
            (51.1, 17.03333), # Wrocław
            (47.00556, 28.8575), # Chisinau
            (45.46427, 9.18951), # Milan
            (37.98376, 23.72784), # Athens
            (39.91987, 32.85427), # Ankara
            (39.9179, 32.86268), # Çankaya
            (40.85216, 14.26811), # Naples
            (59.32938, 18.06871), # Stockholm
            (50.06143, 19.93658), # Kraków
            (44.80401, 20.46513), # Belgrade
            (53.5303, 49.3461), # Tolyatti
            (55.86515, -4.25763), # Glasgow
            (48.71939, 44.50183), # Volgograd
            (54.32824, 48.38657), # Ulyanovsk
            (37.38283, -5.97317), # Sevilla
            (40.4165, -3.70256), # Madrid
            (41.01384, 28.94966), # Istanbul
            (48.023, 37.80224), # Donetsk
            (53.41058, -2.97794), # Liverpool
            (46.48572, 30.74383), # Odesa
            (41.38879, 2.15899), # Barcelona
            (37.87135, 32.48464), # Konya
            (53.33306, -6.24889), # Dublin
            (37.05944, 37.3825), # Gaziantep
            (53.55073, 9.99302), # Hamburg
            (45.04484, 38.97603), # Krasnodar
            (51.77058, 19.47395), # Łódź
            (51.22172, 6.77616), # Düsseldorf
            (45.81444, 15.97798), # Zagreb
            (38.73222, 35.48528), # Kayseri
            (38.41273, 27.13838), # İzmir
            (52.37403, 4.88969), # Amsterdam
            (53.20007, 50.15), # Samara
            (43.29695, 5.38107), # Marseille
            (51.50853, -0.12574), # London
            (39.77667, 30.52056), # Eskişehir
            (36.90812, 30.69556), # Antalya
            (48.46664, 35.04066), # Dnipro
            (43.84864, 18.35644), # Sarajevo
            (44.43225, 26.10626), # Bucharest
            (41.03903, 28.85671), # Bağcılar
            (52.48142, -1.89983), # Birmingham
            (47.85167, 35.11714), # Zaporizhzhya
            (47.23135, 39.72328), # Rostov-na-Donu
            (55.67594, 12.56553), # Copenhagen
            (40.19559, 29.06013), # Bursa
            (52.22977, 21.01178), # Warsaw
            (36.98615, 35.32531), # Adana
            (51.67204, 39.1843), # Voronezh
            (45.07049, 7.68682), # Turin
            (59.93863, 30.31413), # Saint Petersburg
            (48.85341, 2.3488), # Paris
            (39.47391, -0.37966), # Valencia
            (53.9, 27.56667), # Minsk
            (50.85045, 4.34878), # Brussels
            (51.54056, 46.00861), # Saratov
            (49.98081, 36.25272), # Kharkiv
            (57.62987, 39.87368), # Yaroslavl
            (37.91363, 40.21721), # Diyarbakır
            (55.78874, 49.12214), # Kazan
            (48.13743, 11.57549), # Munich
            (50.08804, 14.42076), # Prague
            (48.78232, 9.17702), # Stuttgart
            (41.65606, -0.87734), # Zaragoza
            (38.1166, 13.3636), # Palermo
            (39.90861, 41.27694), # Erzurum
            (47.90572, 33.39404), # Kryvyy Rih
            (60.16952, 24.93545), # Helsinki
            (38.35018, 38.31667), # Malatya
            (49.83826, 24.02324), # Lviv
            (50.45466, 30.5238), # Kyiv
            (41.89193, 12.51133), # Rome
            (47.49835, 19.04045), # Budapest
            (50.11552, 8.68417), # Frankfurt am Main
            (52.52437, 13.41053), # Berlin
            (50.93333, 6.95), # Köln
            (56.946, 24.10589), # Riga
            (42.69751, 23.32415), # Sofia
            (56.84976, 53.20448), # Izhevsk
            (48.20849, 16.37208), # Vienna
            (56.32867, 44.00205), # Nizhniy Novgorod
            (55.75222, 37.61556), # Moscow
        ]

        with open(self.OUTPUT_FILE, 'w') as f:
            for latitude, longitude in cities:

                results = self.live_forecast(latitude, longitude)

                json.dump(results, f)

                f.write('\n')


# weather = WeatherAPI()
# weather.generate_list_of_simulated_forecasts()
# result = weather.simulate_forecast()
# print(json.dumps(result, indent=4))

