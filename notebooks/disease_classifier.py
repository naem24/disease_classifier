import os
from keras import backend as K
from keras.models import model_from_json, load_model
from tensorflow.keras.preprocessing.image import img_to_array
import keras.utils as image
import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import biosppy
from keras.applications.vgg16 import preprocess_input
import json

import math
import cv2
from scipy import ndimage
import os
from PIL import Image
from scipy import signal

# START UTILS FUNCTION
# load model, set cache to prevent reloading
# cache_data is used to cache anything which CAN be stored in a database (python primitives, dataframes, API calls)
# cache_resource is used to catche anything which CANNOT be stored in a database (ML models, DB connections)
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache_resource
def load_chest_xray_models():
    # load model json
    json_file = open('saved_model/cdc_model.json', 'r')
    loaded_cdc_model_json = json_file.read()
    json_file.close()
    loaded_cdc_model = model_from_json(loaded_cdc_model_json)
    
    # load weights into new model
    loaded_cdc_model.load_weights('saved_model/cdc_model_weights.h5')

    # load the pickled pos and neg weight series    
    with open('saved_model/pos_weights.pkl', 'rb') as pw_file:
        pos_weights = pickle.load(pw_file)

    with open('saved_model/neg_weights.pkl', 'rb') as nw_file:
        neg_weights = pickle.load(nw_file)

    # It is important to compile the loaded model before it is used. 
    # This is so that predictions made using the model can use the appropriate efficient computation from the Keras backend.
    loaded_cdc_model.compile(optimizer='adam', loss=get_weighted_loss(neg_weights, pos_weights))

    # Now, load the xray classifier model too
    loaded_xray_classifier_model = load_model('saved_model/final_xray_classifier_model.h5')

    return loaded_cdc_model, loaded_xray_classifier_model

def get_weighted_loss(neg_weights, pos_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        # L(X, y) = −w * y log p(Y = 1|X) − w *  (1 − y) log p(Y = 0|X)
        # from https://arxiv.org/pdf/1711.05225.pdf
        loss = 0
        for i in range(len(neg_weights)):
            loss -= (neg_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + 
                        pos_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
        
        loss = K.sum(loss)
        return loss
    return weighted_loss

def IsImageAnXRay(imgfilepath, xray_classifier_model):
    # The VGG16 model was trained on a specific ImageNet challenge dataset. As such, it is configured to expected input images 
    # to have the shape 224×224 pixels. We will use this as the target size when loading photos from our dataset.
    img = image.load_img(img_file_path,  target_size=(224, 224), grayscale=True)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    
    # Center pixel data based on values derived from ImageNet training dataset
    img = img - [123.68, 116.779, 103.939]
    
    result = xray_classifier_model.predict(img)
    result = result[0].astype('int')

    # Ternary operation to check on both the classes
    model_answer = False if result <= 0 else True
    
    return model_answer

@st.cache_resource
def load_ecg_model():
    ecg_model = load_model('saved_model/ecgScratchEpoch2.hdf5')

    return ecg_model

def model_predict(uploaded_file, model):
    flag = 1
    
    #index1 = str(path).find('sig-2') + 6
    #index2 = -4
    #ts = int(str(path)[index1:index2])
    APC, NORMAL, LBB, PVC, PAB, RBB, VEB = [], [], [], [], [], [], []
    result = {"APC": APC, "Normal": NORMAL, "LBB": LBB, "PAB": PAB, "PVC": PVC, "RBB": RBB, "VEB": VEB}
    
    indices = []
    kernel = np.ones((4,4),np.uint8)
    
    csv = pd.read_csv(uploaded_file)
    csv_data = csv['ecg_signal']
    data = np.array(csv_data)
    signals = []
    count = 1
    peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 2000)[0]

    st.write('No of peaks found: ' + str(len(peaks)))
    #st.stop()

    for i in (peaks[1:-1]):
        diff1 = abs(peaks[count - 1] - i)
        diff2 = abs(peaks[count + 1]- i)
        x = peaks[count - 1] + diff1//2
        y = peaks[count + 1] - diff2//2
        signal = data[x:y]
        signals.append(signal)
        count += 1
        indices.append((x,y))

    st.write('No of signals found: ' + str(len(signals)))
    #st.stop()

    for count, i in enumerate(signals):
        fig = plt.figure(frameon=False)
        plt.plot(i) 
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = 'data/uploaded_ecg_files/temp/fig.png'
        st.write(filename)

        fig.savefig(filename)

        im_gray = image.load_img(filename,  target_size=(128, 128), color_mode="rgb")
        im_gray = img_to_array(im_gray)
        
        im_gray = np.expand_dims(im_gray, axis=0)
        pred = model.predict(im_gray)

        pred_class = pred.argmax(axis=-1)

        if pred_class == 0:
            APC.append(indices[count]) 
        elif pred_class == 1:
            NORMAL.append(indices[count]) 
        elif pred_class == 2:    
            LBB.append(indices[count])
        elif pred_class == 3:
            PAB.append(indices[count])
        elif pred_class == 4:
            PVC.append(indices[count])
        elif pred_class == 5:
            RBB.append(indices[count]) 
        elif pred_class == 6:
            VEB.append(indices[count])

        result = sorted(result.items(), key = lambda y: len(y[1]))[::-1]   
        #output.append(result)
        
        data = {}
        data['result'+str(flag)] = str(result)

        json_filename = 'data/uploaded_ecg_files/temp/data.txt'
        with open(json_filename, 'a+') as outfile:
            json.dump(data, outfile) 
        flag+=1 
    
        with open(json_filename, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace('}{', ',')
        
        with open(json_filename, 'w') as file:
            file.write(filedata) 

        #st.write(result)
        #st.stop()

        return result
# END UTILS FUNCTION 

# UTILS FOR ECG IMAGE TO SIGNAL
import warnings

def image_rotation(image: np.ndarray, angle: int = None) -> np.ndarray:
    if angle is None:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 15,
                                minLineLength=40, maxLineGap=5)

        angles = []

        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)
        if abs(median_angle) < 45:
            img_rotated = ndimage.rotate(image, median_angle, cval=255)
        else:
            return image
    else:
        img_rotated = ndimage.rotate(image, angle)

    return img_rotated


def automatic_brightness_and_contrast(image: np.ndarray,
                                      clip_hist_percent: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result


def shadow_remove(image: np.ndarray) -> np.ndarray:
    rgb_planes = cv2.split(image)
    result_norm_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        shadow_remov = cv2.merge(result_norm_planes)

    return shadow_remov


def warming_filter(image: np.ndarray) -> np.ndarray:
    originalValues = np.array([0, 50, 100, 150, 200, 255])
    redValues = np.array([0, 80, 150, 190, 220, 255])
    blueValues = np.array([0, 20, 40, 75, 150, 255])

    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(image)

    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)

    result = cv2.merge([B, G, R])

    return result

def adjust_image(image: np.ndarray) -> np.ndarray:
    auto_bc_image = automatic_brightness_and_contrast(image)
    adjusted_image = shadow_remove(auto_bc_image)
    warm_image = warming_filter(adjusted_image)
    rotated_image = image_rotation(warm_image)

    return rotated_image


def binarization(image: np.ndarray, threshold: float = None,
                 inverse: bool = True) -> np.ndarray:
    assert image is not None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = grayscale.astype('float32')
    intensity_shift = 50
    grayscale += intensity_shift
    grayscale = np.clip(grayscale, 0, 255)
    grayscale = grayscale.astype('uint8')

    if threshold is None:
        if inverse:
            _, binaryData = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binaryData = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    else:
        if inverse:
            _, binaryData = cv2.threshold(
                grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binaryData = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binaryData, 3)

    return binary

def find_interval(gap: np.ndarray) -> float:
    new_arr = np.diff(gap)
    result = np.delete(new_arr, np.where(new_arr == 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan if np.isnan(np.mean(result)) else np.mean(result)
        
def grid_detection(image: np.ndarray) -> float:
    assert image is not None

    if image.shape[1] < 2000:
        img_w = image.shape[1]
        kf = int(np.ceil(img_w * 15 / 2000))
        kernel_size = kf + 1 if kf % 2 == 0 else kf
        c_kf = int(np.ceil(img_w * 6 / 2000))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
        grid = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, c_kf)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = grayscale.astype('float32')
        intensity_shift = 20
        grayscale += intensity_shift
        grayscale = np.clip(grayscale, 0, 255)
        grayscale = grayscale.astype('uint8')
        blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
        grid = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 6)
        grid = cv2.medianBlur(grid, 3)

    intervals = []
    for n_row in range(grid.shape[0]):
        intervals.append(find_interval(np.where(grid[n_row, :] == 0)[0]))
    for n_col in range(grid.shape[1]):
        intervals.append(find_interval(np.where(grid[:, n_col] == 0)[0]))

    scale = np.nanmean(np.array(intervals))
    scale = scale.round(1)

    return scale

def signal_extraction(image: np.ndarray, scale: float) -> float:
    assert image is not None

    ecg = image
    max_value = 0
    x_row = 0
    for row in range(len(ecg)):
        if ecg[row].sum() > max_value:
            max_value = ecg[row].sum()
            x_row = row

    Y = []
    for col in range(len(ecg[1])):
        for row in range(len(ecg[:, col].flatten())):
            if ecg[:, col].flatten()[row] == 255:
                y = (x_row - row) / scale
                Y.append(y)
                break

    sig = Y
    win = signal.windows.hann(10)
    filtered = signal.convolve(sig, win, mode='same') / sum(win)

    return filtered

def convert_image_to_signal(image: Image.Image) -> np.ndarray:
    """This functions converts image with ECG signal to an array representation
    of the signal.

    Args:
        image (Image.Image): input image with ECG signal
                             (one lead; see readme for the image requirements)

    Returns:
        np.ndarray or Failed: array representation of ECG signal or Failed
    """
    image = np.asarray(image)
    adjusted_image = adjust_image(image)
    scale = grid_detection(adjusted_image)
    binary_image = binarization(adjusted_image)
    ecg_signal = signal_extraction(binary_image, scale)

    return ecg_signal

# END UTILS FOR ECG IMAGE TO SIGNAL

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Multiple Disease Detection",
    initial_sidebar_state = 'auto',
    layout='wide'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
    div.block-container{padding-top:2rem;}
    div.stButton {text-align:center;}
    </style>
"""

# hide the CSS code from the screen as they are embedded in markdown text. 
# Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Sidebar for Navigation
with st.sidebar:
    selected_option = option_menu('Multiple Disease Classifer', ['Lung Disease', 'ECG Arrhythmia','Heart Disease'],
    icons=['activity','activity','heart'],default_index=1, menu_icon="cast")

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
            'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

# This ensure we have slotted the main page in 3 columns - with the middle column taking 80% of the space 
with st.columns([0.10, 0.80, 0.10])[1]:
    if selected_option == "Lung Disease":
        st.header("LUNG DISEASE CLASSIFIER")
        #st.divider()
        st.markdown('''
        <b><p>This classifier is powered by a CNN model which is trained on a dataset consisting of more than 1 million frontal X-ray images from more than 32K patients.<br><br>
        The model will attempt to classify a frontal chest xray as either one of the diseases:<br>
        Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis,
        Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema, Consolidation<br><br>
        <i>This tool is for education purposes only and should not be used for clinical diagnosis of the above mentioned diseases.</i>
        </p>
        </b>
        ''', unsafe_allow_html=True)
        st.write()

        # This will be called once as we have the decorator before the load_model function
        with st.spinner('Loading Model....'):
            # Load both the models
            cdc_model, xray_classifier_model = load_chest_xray_models()

            # Add the file uploader widget
            uploaded_file = st.file_uploader('Upload an unaltered frontal chest xray, 8 MB or smaller, black/white image', type=['png','jpg'])

            # Add three columns with only second col used to center the content    
            if uploaded_file is None:
                st.text("Please upload an image file")
            else:
                # https://discuss.streamlit.io/t/file-uploader-file-to-opencv-imread/8479/2
                # file_uploader does not store the uploaded file on the webserver but keeps it in memory. 
                # It returns a ‘stream’ class variable to access it
                img_file_path = os.path.join("data/uploaded_xray_images/", uploaded_file.name)
                with open(img_file_path, "wb") as user_file:
                        user_file.write(uploaded_file.read())

                # Validate the uploaded image and check if its an xray or not
                if IsImageAnXRay(img_file_path, xray_classifier_model) == False:
                    st.warning("Please upload a valid x-ray image and try again")
                    st.stop()

                col1, col2 = st.columns((0.5, 0.50))

                with col1:
                    # Load the non-normalized image here (image without mean and std adjustments)     
                    x = image.load_img(img_file_path, target_size=(320, 320))
                    st.image(x, use_column_width=True)
                
                with col2:
                    # Read the picked variables to get the mean and std for normalization
                    with open('saved_model/cdc_mean_std.pkl', 'rb') as ms_file:
                        cdc_mean_std = pickle.load(ms_file)

                    mean = cdc_mean_std["saved_mean"]
                    std = cdc_mean_std["saved_std"]
                    
                    # Load the image saved in the app folder earlier
                    x = image.load_img(img_file_path, target_size=(320, 320))
                    
                    # Normalize the image before sending it to predict to the model 
                    x -= mean
                    x /= std
                    processed_image = np.expand_dims(x, axis=0)
                    preds = cdc_model.predict(processed_image)
                    pred_class = labels[np.argmax(preds)]
                    pred_df = pd.DataFrame(preds, columns = labels)

                    corr_message = """
                    Clinical correlation is strongly recommended to determine the significance of the radiology findings.
                    This will allow your doctor to make an accurate diagnosis using all available information  - your medical history,
                    physical examination, laboratory tests or other imaging studies.
                    """
                    
                    st.warning("Model predicts a higher probability of **" + pred_class + "**")
                    st.write(corr_message)
                    
                    # Find the largest value in the row
                    # Commented for now but can be useful info for later
                    #max_pred_value = pred_df.iloc[0, :].max()
    
    elif selected_option == "ECG Arrhythmia":
        st.header("ECG ARRHYTHMIA CLASSIFIER")

        st.markdown('''
        <b><p>This classifier analyzed an ECG and classifies them into seven categories - one being normal and the other six being
        one of the below arrhythmia beats:<br><br>
        Atrial Premature Contraction (APC) beat, Left Bundle Branch Block (LBB) beat, Premature ventricular contraction (PVC) beat,
        Paced Beat (PAB), Right Bundle Branch Block (RBB) beat, Ventricular Escape Beat (VEB)<br><br>
        <i>This tool is for education purposes only and should not be used for clinical diagnosis of the above mentioned diseases.</i>
        </p>
        </b>
        ''', unsafe_allow_html=True)

        # This will be called once as we have the decorator before the load_model function
        with st.spinner('Loading Model....'):
            # Load both the models
            ecg_model = load_ecg_model()

            # Add the file uploader widget
            uploaded_file = st.file_uploader('Upload a single-lead ECG image file in png format', type=['png'])

            # Add three columns with only second col used to center the content    
            if uploaded_file is None:
                st.text("Please upload a PNG file")
            else:
                # https://discuss.streamlit.io/t/file-uploader-file-to-opencv-imread/8479/2
                # file_uploader does not store the uploaded file on the webserver but keeps it in memory. 
                # It returns a ‘stream’ class variable to access it
                img_file_path = os.path.join("data/uploaded_ecg_files/", uploaded_file.name)
                with open(img_file_path, "wb") as user_file:
                        user_file.write(uploaded_file.read())

                col1, col2 = st.columns((0.5, 0.50))

                with col1:
                    # CODE FOR READING AN IMAGE
                    # We will now read an ECG image, convert it to a signal and then pass it ahead
                    # 1) Open the image using PIL
                    img = image.load_img(img_file_path)
                    img = img_to_array(img)
                    final_signal = convert_image_to_signal(img[:,:,:3])
                    
                    # 2 - Save the file to csv and then transpose it
                    # Saving the array to a CSV file
                    np.savetxt('data/uploaded_ecg_files/output.csv', final_signal, delimiter=',', header='ecg_signal', comments = '')
                    
                    model_ret = model_predict('data/uploaded_ecg_files/output.csv', ecg_model)
                    st.warning(model_ret)


                    # CODE for reading a CSV directly
                    #model_ret = model_predict(img_file_path, ecg_model)
                    #st.warning(model_ret)
                                        
                    # Find the largest value in the row
                    # Commented for now but can be useful info for later
                    #max_pred_value = pred_df.iloc[0, :].max()
    elif selected_option == "Heart Disease":
        st.header("HEART DISEASE CLASSIFIER")

        st.markdown('''
        <b><p>This classifier is powered by a CNN model which is trained on a dataset consisting of more than 1 million frontal X-ray images from more than 32K patients.<br><br>
        The model will attempt to classify a frontal chest xray as either one of the diseases:<br>
        Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis,
        Pneumothorax, Pleural_Thickening, Pneumonia, Fibrosis, Edema, Consolidation<br><br>
        <i>This tool is for education purposes only and should not be used for clinical diagnosis of the above mentioned diseases.</i>
        </p>
        </b>
        ''', unsafe_allow_html=True)

        age = st.slider('Age', 18, 100, 50)
        sex_options = ['Male', 'Female']
        sex = st.selectbox('Sex', sex_options)
        sex_num = 1 if sex == 'Male' else 0 
        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp = st.selectbox('Chest Pain Type', cp_options)
        cp_num = cp_options.index(cp)
        trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
        chol = st.slider('Cholesterol', 100, 600, 250)
        fbs_options = ['False', 'True']
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
        fbs_num = fbs_options.index(fbs)
        restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
        restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
        restecg_num = restecg_options.index(restecg)
        thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
        exang_options = ['No', 'Yes']
        exang = st.selectbox('Exercise Induced Angina', exang_options)
        exang_num = exang_options.index(exang)
        oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
        slope_options = ['Upsloping', 'Flat', 'Downsloping']
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
        slope_num = slope_options.index(slope)
        ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thal = st.selectbox('Thalassemia', thal_options)
        thal_num = thal_options.index(thal)

        if st.button('Predict'):
            user_input = pd.DataFrame(data={
                'age': [age],
                'sex': [sex_num],  
                'cp': [cp_num],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_num],
                'restecg': [restecg_num],
                'thalach': [thalach],
                'exang': [exang_num],
                'oldpeak': [oldpeak],
                'slope': [slope_num],
                'ca': [ca],
                'thal': [thal_num]
            })

            #prediction = model.predict(user_input)
            #prediction_proba = model.predict_proba(user_input)

            bg_color = 'green'
            prediction_result = 'Negative'

            #if prediction[0] == 1:
            #    bg_color = 'red'
            #    prediction_result = 'Positive'
            #else:
            #    bg_color = 'green'
            #    prediction_result = 'Negative'
            
            #confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}</p>", unsafe_allow_html=True)

            #st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

