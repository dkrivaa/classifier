import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

def process_image(model, image, size, preprocess_input):
    # tf_image = tf.io.read_file(image)
    decoded_image = tf.image.decode_image(image)
    image_resized = tf.image.resize(decoded_image,size)
    image_batch = tf.expand_dims(image_resized, axis=0)
    image_batch = preprocess_input(image_batch)

    preds = model.predict(image_batch)
    decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(preds=preds)

    # fig = plt.subplot()
    # plt.imshow(decoded_image)
    label = decoded_preds[0][0][1]
    score = decoded_preds[0][0][2] * 100
    title = label + ' ' + str('{:.2f}%'.format(score))
    # plt.title(title, fontsize=16)
    #
    # st.pyplot(fig)

    st.write(f'this is: {title})



def check_pred():
    # getting image ready
    st.image(upload_image, caption="Uploaded Image", use_column_width=True)
    tf_image = upload_image.read()

    model = my_model
    size = (224,224)
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    process_image(model, tf_image, size, preprocess_input)


st.set_page_config(page_title='Image Classifier',)
st.markdown(f'<span style="color: #4b7fd1; '
                f'font-size: 24px"><b>Image Classifier</b></span>'
                , unsafe_allow_html=True)

my_model = tf.keras.applications.resnet50.ResNet50()

upload_image = st.file_uploader('Upload your picture', ['jpg'])

if upload_image:

    st.button('Guess what', on_click=check_pred)


