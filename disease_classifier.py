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

# START UTILS FUNCTION
# load model, set cache to prevent reloading
# cache_data is used to cache anything which CAN be stored in a database (python primitives, dataframes, API calls)
# cache_resource is used to catche anything which CANNOT be stored in a database (ML models, DB connections)
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache_resource
def load_models():
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

# END UTILS FUNCTION 

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
    selected_option = option_menu('Multiple Disease Classifer', ['Lung Disease', 'Heart Disease'],
    icons=['activity','heart'],default_index=0, menu_icon="cast")

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
            'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

# This ensure we have slotted the main page in 3 columns - with the middle column taking 80% of the space 
with st.columns([0.10, 0.80, 0.10])[1]:
    if selected_option == "Lung Disease":
        st.header("LUNG DISEASE CLASSIFIER")
        #st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        st.divider()
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
            cdc_model, xray_classifier_model = load_models()

            # Add the file uploader widget
            uploaded_file = st.file_uploader('Upload an unaltered frontal chest xray, 8 MB or smaller, black/white image', type=['png','jpg'])

            # Add three columns with only second col used to center the content    
            if uploaded_file is None:
                st.text("Please upload an image file")
            else:
                # https://discuss.streamlit.io/t/file-uploader-file-to-opencv-imread/8479/2
                # file_uploader does not store the uploaded file on the webserver but keeps it in memory. 
                # It returns a ‘stream’ class variable to access it
                img_file_path = os.path.join("data/uploaded_images/", uploaded_file.name)
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
    elif selected_option == "Heart Disease":
        st.header("HEART DISEASE CLASSIFIER")
        st.markdown("<b><p>Coming soon...</p></b>", unsafe_allow_html=True)
