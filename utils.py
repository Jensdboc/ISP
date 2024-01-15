# Standard libraries
import base64
from base64 import b64decode, b64encode
import warnings
import json
import random

# Third-party libraries
import cv2
import numpy as np
import PIL
import io
import matplotlib.cm as cm
import requests
import urllib.request
# from js2py import eval_js

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Google Colab related imports
from google.colab.output import eval_js

# Widgets
import ipywidgets as widgets
from IPython.display import display, Javascript

# Filter or ignore the specific warning
warnings.filterwarnings("ignore")

explainableAI_path = "/content/drive/MyDrive/BotanicBrAIn/model.h5"
model = tf.keras.models.load_model(explainableAI_path)

model_filename = "model_v0_train.keras"
response = requests.get("https://raw.githubusercontent.com/Jensdboc/ISP/master/model_v0_train_20/model_v0_train.keras")

with open(model_filename, 'wb') as model_file:
    model_file.write(response.content)

model_predict = keras.models.load_model(model_filename)


classes_predict = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)__Common_rust',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape__Esca(Black_Measles)',
                   'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange__Haunglongbing(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,bell__Bacterial_spot',
                   'Pepper,bell__healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


class WeatherAPI:

    def __init__(self):
        self.ACCESS_KEY = 'f0c7Frm9iWgnm43y'
        self.OUTPUT_FILE = './simulated_forecasts.txt'
        response = requests.get('https://raw.githubusercontent.com/Jensdboc/ISP/master/simulated_forecasts.txt')
        self.FILE = response.content
        self.cities = {
            "Wrocław": (51.1, 17.03333),
            "Chisinau": (47.00556, 28.8575),
            "Milan": (45.46427, 9.18951),
            "Athens": (37.98376, 23.72784),
            "Ankara": (39.91987, 32.85427),
            "Çankaya": (39.9179, 32.86268),
            "Naples": (40.85216, 14.26811),
            "Stockholm": (59.32938, 18.06871),
            "Kraków": (50.06143, 19.93658),
            "Belgrade": (44.80401, 20.46513),
            "Tolyatti": (53.5303, 49.3461),
            "Glasgow": (55.86515, -4.25763),
            "Volgograd": (48.71939, 44.50183),
            "Ulyanovsk": (54.32824, 48.38657),
            "Sevilla": (37.38283, -5.97317),
            "Madrid": (40.4165, -3.70256),
            "Istanbul": (41.01384, 28.94966),
            "Donetsk": (48.023, 37.80224),
            "Liverpool": (53.41058, -2.97794),
            "Odesa": (46.48572, 30.74383),
            "Barcelona": (41.38879, 2.15899),
            "Konya": (37.87135, 32.48464),
            "Dublin": (53.33306, -6.24889),
            "Gaziantep": (37.05944, 37.3825),
            "Hamburg": (53.55073, 9.99302),
            "Krasnodar": (45.04484, 38.97603),
            "Łódź": (51.77058, 19.47395),
            "Düsseldorf": (51.22172, 6.77616),
            "Zagreb": (45.81444, 15.97798),
            "Kayseri": (38.73222, 35.48528),
            "İzmir": (38.41273, 27.13838),
            "Amsterdam": (52.37403, 4.88969),
            "Samara": (53.20007, 50.15),
            "Marseille": (43.29695, 5.38107),
            "London": (51.50853, -0.12574),
            "Eskişehir": (39.77667, 30.52056),
            "Antalya": (36.90812, 30.69556),
            "Dnipro": (48.46664, 35.04066),
            "Sarajevo": (43.84864, 18.35644),
            "Bucharest": (44.43225, 26.10626),
            "Bağcılar": (41.03903, 28.85671),
            "Birmingham": (52.48142, -1.89983),
            "Zaporizhzhya": (47.85167, 35.11714),
            "Rostov-na-Donu": (47.23135, 39.72328),
            "Copenhagen": (55.67594, 12.56553),
            "Bursa": (40.19559, 29.06013),
            "Warsaw": (52.22977, 21.01178),
            "Adana": (36.98615, 35.32531),
            "Voronezh": (51.67204, 39.1843),
            "Turin": (45.07049, 7.68682),
            "Saint Petersburg": (59.93863, 30.31413),
            "Paris": (48.85341, 2.3488),
            "Valencia": (39.47391, -0.37966),
            "Minsk": (53.9, 27.56667),
            "Brussels": (50.85045, 4.34878),
            "Saratov": (51.54056, 46.00861),
            "Kharkiv": (49.98081, 36.25272),
            "Yaroslavl": (57.62987, 39.87368),
            "Diyarbakır": (37.91363, 40.21721),
            "Kazan": (55.78874, 49.12214),
            "Munich": (48.13743, 11.57549),
            "Prague": (50.08804, 14.42076),
            "Stuttgart": (48.78232, 9.17702),
            "Zaragoza": (41.65606, -0.87734),
            "Palermo": (38.1166, 13.3636),
            "Erzurum": (39.90861, 41.27694),
            "Kryvyy Rih": (47.90572, 33.39404),
            "Helsinki": (60.16952, 24.93545),
            "Malatya": (38.35018, 38.31667),
            "Lviv": (49.83826, 24.02324),
            "Kyiv": (50.45466, 30.5238),
            "Rome": (41.89193, 12.51133),
            "Budapest": (47.49835, 19.04045),
            "Frankfurt am Main": (50.11552, 8.68417),
            "Berlin": (52.52437, 13.41053),
            "Köln": (50.93333, 6.95),
            "Riga": (56.946, 24.10589),
            "Sofia": (42.69751, 23.32415),
            "Izhevsk": (56.84976, 53.20448),
            "Vienna": (48.20849, 16.37208),
            "Nizhniy Novgorod": (56.32867, 44.00205),
            "Moscow": (55.75222, 37.61556),
        }

    def live_forecast(self, city):
        if city.capitalize() in self.cities:
            latitude, longitude = self.cities[city.capitalize()]
        else:
            return None

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
        lines = self.FILE.splitlines()

        random_line = random.choice(lines)

        result = json.loads(random_line)

        return result


class PlantAPI:

    def __init__(self):
        # https://trefle.io/api/v1/plants/<ID>/
        # https://trefle.io/api/v1/plants/search?token=tmRZ6jLbqmFmGITDKOZ_MsWIOLiPDQp7AEc5UunWbN8&q=Strawberry&range[atmospheric_humidity]=0&page=1
        # self.OUTPUT_FILE = './plant_archive.json'
        # response = requests.get('/content/plant_archive.txt')
        # self.FILE = response.content

        url = "https://raw.githubusercontent.com/Jensdboc/ISP/master/plant_archive.json"
        file_name = "/content/plant_archive.txt"
        self.weather_api = WeatherAPI()

        # Download the file
        urllib.request.urlretrieve(url, file_name)

        with open('/content/plant_archive.txt', 'r') as file:
            self.FILE = file.read()

    def get_plant_by_name(self, name):
        plants = json.loads(self.FILE)
        if name in plants:
            return plants[name]
        else:
            return None

    def calculate_plant_watering_needs(self, name, city):
        plant = self.get_plant_by_name(name)
        if plant is None:
            return None

        # Get water needs from "Waterwise" plant database
        water_needs_min_mm_year = float(plant['water_needs_min_mm'])
        water_needs_max_mm_year = float(plant['water_needs_max_mm'])
        water_needs_avg_mm_year = (water_needs_min_mm_year + water_needs_max_mm_year) / 2
        # Convert to average cm per week
        water_needs_avg_cm_year = water_needs_avg_mm_year / 10
        water_needs_avg_cm_week = water_needs_avg_cm_year / 52
        water_needs_avg_cm_day = water_needs_avg_cm_week / 7

        # Retrieve weekly forecast
        # weekly_forecast = weather_api.simulate_forecast()
        weekly_forecast = self.weather_api.live_forecast(city)
        precip_cm_day = list(map(lambda daily_forecast: daily_forecast['precipAccumulation'], weekly_forecast))[0:7]

        # Calculate daily delta
        delta_cm_day = list(map(lambda daily_precip_cm_day: np.round(np.max(water_needs_avg_cm_day - daily_precip_cm_day, 0), 4), precip_cm_day))

        return water_needs_avg_cm_day, precip_cm_day, delta_cm_day


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)/255
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
    """
    Params:
            js_reply: JavaScript object containing image from webcam
    Returns:
            img: OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(',')[1])
    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
    """
    Params:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
    Returns:
          bytes: Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format='png')
    # format return string
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

    return bbox_bytes


def superimpose(img_array, heatmap, cam_path="cam.jpg", alpha=0.5):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    rgba = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = int(alpha * 255)

    return rgba


# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
    js = Javascript('''
      var video;
      var layout = null;
      var stream;
      var captureCanvas;
      var pendingResolve = null;
      var shutdown = false;

      video = document.createElement('video');
      video.style.display = 'block';
      video.style.width = '600px';
      video.style.height = '450px';
      video.style.position = 'absolute';
      video.style.top = '50px';
      video.style.left = '50px';
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };

      imgElement = document.createElement('img');
      imgElement.onclick = () => { shutdown = true; };
      imgElement.style.position = 'absolute';
      imgElement.style.width = '600px';
      imgElement.style.height = '450px';
      imgElement.style.top = '50px';
      imgElement.style.left = '50px';

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;

      var text1;
      var text2;
      var text3;
      var text4;
      var text5;

      textElement = document.createElement('span');
      textElement.style.color = 'black';
      textElement.style.fontSize = '24px';
      textElement.style.position = 'absolute';
      textElement.style.top = '530px';
      textElement.style.left = '70px';
      textElement.style.fontFamily = 'Arial';

      textElement2 = document.createElement('span');
      textElement2.style.color = 'black';
      textElement2.style.fontSize = '24px';
      textElement2.style.position = 'absolute';
      textElement2.style.top = '580px';
      textElement2.style.left = '70px';
      textElement2.style.fontFamily = 'Arial';

      text1 = document.createElement('span');
      text1.style.color = 'black';
      text1.style.fontSize = '24px';
      text1.style.position = 'absolute';
      text1.style.top = '125px';
      text1.style.left = '730px';
      text1.style.fontFamily = 'Arial';

      text2 = document.createElement('span');
      text2.style.color = 'black';
      text2.style.fontSize = '24px';
      text2.style.position = 'absolute';
      text2.style.top = '239px';
      text2.style.left = '730px';
      text2.style.fontFamily = 'Arial';

      text3 = document.createElement('span');
      text3.style.color = 'black';
      text3.style.fontSize = '24px';
      text3.style.position = 'absolute';
      text3.style.top = '351px';
      text3.style.left = '730px';
      text3.style.fontFamily = 'Arial';

      text4 = document.createElement('span');
      text4.style.color = 'black';
      text4.style.fontSize = '24px';
      text4.style.position = 'absolute';
      text4.style.top = '463px';
      text4.style.left = '730px';
      text4.style.width = '357px';
      text4.style.fontFamily = 'Arial';

      text5 = document.createElement('span');
      text5.style.color = 'black';
      text5.style.fontSize = '24px';
      text5.style.position = 'absolute';
      text5.style.top = '600px';
      text5.style.width = '357px';
      text5.style.left = '730px';
      text5.style.fontFamily = 'Arial';

      async function start(){
        stream = await navigator.mediaDevices.getUserMedia({video: { facingMode: "environment"}});
        video.srcObject = stream;
        await video.play();
      }


      function removeDom() {
        stream.getVideoTracks()[0].stop();
        video.remove();
        layout.remove();
        video = null;
        layout = null
        stream = null;
        captureCanvas = null;
      }

      function onAnimationFrame() {
        if (!shutdown) {
          window.requestAnimationFrame(onAnimationFrame);
        }
        if (pendingResolve) {
          var result = "";
          if (!shutdown) {
            captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            result = captureCanvas.toDataURL('image/jpeg', 0.8)
          }
          var lp = pendingResolve;
          pendingResolve = null;
          lp(result);
        }
      }

      async function createDom() {
        if (layout !== null) {
          return stream;
        }

        layout = document.createElement('div');
        layout.style.backgroundSize = 'cover';
        layout.style.width = '1200px';
        layout.style.height = '750px';
        document.body.appendChild(layout);

        layout.append(textElement, textElement2, text1, text2, text3, text4, text5, video, imgElement);

        window.requestAnimationFrame(onAnimationFrame);

        return;
      }

      async function updateDom(imgData, a, b, c, d, e, f, backgroundData) {
        if (shutdown) {
          removeDom();
          shutdown = false;
          return '';
        }

        await createDom();

        text1.innerText = a;
        text2.innerText = b;
        text3.innerText = c;
        text4.innerText = d;
        text5.innerText = e;
        imgElement.src = imgData;
        layout.style.backgroundImage = "url('" + backgroundData + "')";

        if(b){
          if (b == 'healthy'){
            textElement.innerText = 'The plant is healthy.';
          } else {
            textElement.innerText = 'The plant is ill.';
          }
        }

        if(f){
          textElement2.innerText = (f === '0') ? "You don't need to water today." : "Remember you should water today!";
        }

        var result = await new Promise(function(resolve, reject) {
          pendingResolve = resolve;
        });

        shutdown = false;

        return result;
      }
      ''')

    display(js)


def call_apis_from_prediction(plant, city):
    random.seed(10)
    plant_api = PlantAPI()
    results = plant_api.calculate_plant_watering_needs(plant, city)
    if results is None:
        return "", "", ""

    return results


def run_application(city):
    target_size = (256, 256)
    video_stream()
    imgData = ''

    info = {
        "full_prediction": "",
        "plant": "",
        "disease": "",
        "certainty": "",
        "city": "",
        "water_consumption": "",
        "rain_fall": "",
        "water_needs": "",
        "should_water_today": ""
    }

    # Create background
    url = "https://raw.githubusercontent.com/Jensdboc/ISP/master/demo/background.png"
    file_name = "/content/background.png"
    urllib.request.urlretrieve(url, file_name)

    image = cv2.imread('/content/background.png')
    base64_encoded = base64.b64encode(cv2.imencode('.png', image)[1]).decode("utf-8")
    backgroundData = 'data:image/png;base64,' + base64_encoded

    eval_js('start()')

    while True:
        js_reply = eval_js('updateDom("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'.format(imgData, info['plant'], info['disease'], str(info['water_consumption'])[:6], info['rain_fall'], info['water_needs'], info['should_water_today'], backgroundData))
        if not js_reply:
            break

        # Create image
        img = js_to_image(js_reply)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Compute prediction and probabilities
        img_resized = cv2.resize(img[:, 80:-80, :], target_size)
        img_array = keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)/255
        predictions = model_predict.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(predictions)
        c = np.argmax(probabilities, axis=1)
        max = np.max(probabilities, axis=1)[0]
        prediction = classes_predict[int(c)]

        # Build dictionary
        info['full_prediction'] = prediction
        info['plant'] = prediction.split('___')[0].replace('_', ' ')
        if len(prediction.split('___')) >= 2:
            info['disease'] = prediction.split('___')[1].replace('_', ' ')
        info['certainty'] = f'{max*100}%'
        info['city'] = city
        info['water_consumption'], info['rain_fall'], info['water_needs'] = call_apis_from_prediction(info['plant'], info['city'])
        if info['water_needs']:
            info['water_needs'] = [x if x >= 0 else 0 for x in info['water_needs']]
            if info['water_needs'][0] == 0:
                info['should_water_today'] = 0
            else:
                info['should_water_today'] = 1

        # Create the heatmap if not healthy
        if info['disease'] != 'healthy':
            img_resized = cv2.resize(img, target_size)
            img_array = keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)/255
            heatmap = make_gradcam_heatmap(img_array, model, "conv2d_5")
            superimposed_img = superimpose(np.array(img), heatmap)
            base64_encoded = base64.b64encode(cv2.imencode('.png', superimposed_img)[1]).decode("utf-8")
        else:
            transparent_image = np.zeros((target_size[0], target_size[1], 4), dtype=np.uint8)
            base64_encoded = base64.b64encode(cv2.imencode('.png', transparent_image)[1]).decode("utf-8")

        imgData = 'data:image/png;base64,' + base64_encoded


# Create a list of options for the dropdown
options = [
    "Wrocław", "Chisinau", "Milan", "Athens", "Ankara", "Çankaya", "Naples", "Stockholm", "Kraków",
    "Belgrade", "Tolyatti", "Glasgow", "Volgograd", "Ulyanovsk", "Sevilla", "Madrid", "Istanbul", "Donetsk",
    "Liverpool", "Odesa", "Barcelona", "Konya", "Dublin", "Gaziantep", "Hamburg", "Krasnodar", "Łódź",
    "Düsseldorf", "Zagreb", "Kayseri", "İzmir", "Amsterdam", "Samara", "Marseille", "London", "Eskişehir",
    "Antalya", "Dnipro", "Sarajevo", "Bucharest", "Bağcılar", "Birmingham", "Zaporizhzhya", "Rostov-na-Donu",
    "Copenhagen", "Bursa", "Warsaw", "Adana", "Voronezh", "Turin", "Saint Petersburg", "Paris", "Valencia",
    "Minsk", "Brussels", "Saratov", "Kharkiv", "Yaroslavl", "Diyarbakır", "Kazan", "Munich", "Prague", "Stuttgart",
    "Zaragoza", "Palermo", "Erzurum", "Kryvyy Rih", "Helsinki", "Malatya", "Lviv", "Kyiv", "Rome", "Budapest",
    "Frankfurt am Main", "Berlin", "Köln", "Riga", "Sofia", "Izhevsk", "Vienna", "Nizhniy Novgorod", "Moscow"
]

# Searchable dropdown menu
search_dropdown = widgets.Combobox(
    placeholder='Type to search',
    options=options,
    value="Barcelona",
    description='Select city:',
)
