import streamlit as st
import pickle
import numpy as np

# importing the model
# we have built this model using Random Forest Regression
#ColumnTransformer
#OneHotEncoding
pipe = pickle.load(open('pipe.pkl','rb')) #This is the model
df = pickle.load(open('df.pkl','rb')) #this file contains dataset on which the model is trained
st.title("Laptop Price Predictor")

# brand
brand = st.selectbox('Brand',df['Company'].unique())

# type
types = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touch = st.selectbox('Touchscreen',['No','Yes'])

# IPS In-plane switching display
ips = st.selectbox('IPS Display',['No','Yes'])

# screen size
screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# Hard Disk Drive
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# Solid-State Drive
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# Graphics Processing Unit
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

#operating system
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict'):
    # query
    ppi = None
    if touch == 'Yes':
        touch = 1
    else:
        touch = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
    query = np.array([brand,types,ram,weight,touch,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title(" Estimated Laptop Price  ₹" + str(int(np.exp(pipe.predict(query)[0]))))
    # in the above step we are placing the price in exp function because
    # at the time of model training we predicted price using logarithmic function