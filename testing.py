#importing the libraries

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json 
import requests 
from streamlit_lottie import st_lottie 
import tensorflow as tf

#creating animations

url1 = requests.get( 
    "https://lottie.host/04a3a571-c36f-4da7-a525-ea2e9c69063f/GVNsJGl4XP.json") 
url1_json = dict() 
  
if url1.status_code == 200: 
    url1_json = url1.json() 
else: 
    print("Error in the URL") 

url2 = requests.get( 
    "https://lottie.host/5a2933c7-50d0-46dd-b530-3c3896b2e5a6/hs22lYVhRD.json") 
url2_json = dict() 
  
if url2.status_code == 200: 
    url2_json = url2.json() 
else: 
    print("Error in the URL") 
  
url3 = requests.get( 
    "https://lottie.host/1d37f75f-4152-4748-a96b-ddffc4ff4775/Qu1TPlwctc.json") 
url3_json = dict() 
  
if url3.status_code == 200: 
    url3_json = url3.json() 
else: 
    print("Error in the URL") 

url4 = requests.get( 
    "https://lottie.host/670b1cae-c95f-4f77-8981-576a599a3378/zwE7lzSbmG.json") 
url4_json = dict() 
  
if url4.status_code == 200: 
    url4_json = url4.json() 
else: 
    print("Error in the URL") 

url5 = requests.get( 
    "https://lottie.host/fb8edd1b-74ca-489c-986b-ee5fdd5d8ebb/aFNiou8993.json") 
url5_json = dict() 
  
if url5.status_code == 200: 
    url5_json = url5.json() 
else: 
    print("Error in the URL") 

url6 = requests.get( 
    "https://lottie.host/21c4b64a-0034-4636-98ac-575f6ef9b082/AvRxmvn6LS.json") 
url6_json = dict() 
  
if url6.status_code == 200: 
    url6_json = url6.json() 
else: 
    print("Error in the URL") 

url7 = requests.get( 
    "https://lottie.host/979cdf11-2db6-4a47-8a8a-1bcdd142aa3d/ANPC3zbXel.json") 
url7_json = dict() 
  
if url7.status_code == 200: 
    url7_json = url7.json() 
else: 
    print("Error in the URL") 
  
  


#Creating the ML model



#Creating streamlit app

st.sidebar.title('Navigation')
nav = st.sidebar.radio('Go to: ',('Home', 'General Information', 'General Awareness', 'Predictions', 'About'))
if nav=='Predictions':

    st.title("Diabetes Prediction")
    st_lottie(url3_json)
    st.write('''Here on our Diabetes Prediction page, we're using machine learning smart tools to help predict
              diabetes. Remember, these tools are just helpers and not a substitute for talking to a doctor. 
             It's super important to always chat with a healthcare professional for the best advice about your health.''')
    
    bp = st.checkbox('Have high blood pressure?')
    if bp:
        bpvar=1
    else:
        bpvar=0


    cl = st.checkbox('Have high cholestrol level?')
    if cl:
        clvar=1
    else:
        clvar=0
    


    clcheck = st.checkbox('Have you checked cholestrol level in last 5 years?')
    clcheckvar=0
    if clcheck:
        clcheckvar=1
        

    bmi = st.number_input("Enter your body mass index", step=1)
    age = st.number_input("Enter your Age", step=1)

    smoke = st.checkbox('Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes')
    smokevar=0
    if smoke:
        smokevar=1
    
        
    
    stroke = st.checkbox('(Ever told) you had a stroke')
    strokevar=0
    if stroke:
        strokevar=1

        

    heartdis = st.checkbox('coronary heart disease (CHD) or myocardial infarction (MI)')
    heartdisvar=0
    if heartdis:
        heartdisvar=1

        

    phys = st.checkbox('physical activity in past 30 days - not including job')
    physvar=0
    if phys:
        physvar=1

        

    fruits = st.checkbox('Consume Fruit 1 or more times per day')
    fruitsvar=0
    if fruits:
        fruitsvar=1

        

    veggies = st.checkbox('Consume Vegetables 1 or more times per day')
    veggiesvar=0
    if veggies:
        veggiesvar=1

        

    diffwalk = st.checkbox('Do you have serious difficulty walking or climbing stairs?')
    diffwalkvar=0
    if diffwalk:
        diffwalkvar=1
        

    alcohol = st.checkbox('Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)')
    alcoholvar=0
    if alcohol:
        alcoholvar=1

        
    
    sex = st.radio('Gender', ['male', 'female'])
    if sex=='male':
        sexvar = 1
    elif sex=='female':
        sexvar = 0
    
    genhealth = st.selectbox(
    'Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor',
    ('1', '2', '3', '4', '5'))
    if genhealth=='1':
        genhealthvar=1
    elif genhealth=='2':
        genhealthvar=2
    elif genhealth=='3':
        genhealthvar=3
    elif genhealth=='4':
        genhealthvar=4
    elif genhealth=='5':
        genhealthvar=5

    physhealth = st.selectbox(
    'Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 have you had them?',
    ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'))
    if physhealth=='1':
        physhealthvar=1
    elif physhealth=='2':
        physhealthvar=2
    elif physhealth=='3':
        physhealthvar=3
    elif physhealth=='4':
        physhealthvar=4
    elif physhealth=='5':
        physhealthvar=5
    elif physhealth == '6':
        physhealthvar = 6
    elif physhealth == '7':
        physhealthvar = 7
    elif physhealth == '8':
        physhealthvar = 8
    elif physhealth == '9':
        physhealthvar = 9
    elif physhealth == '10':
        physhealthvar = 10
    elif physhealth == '11':
        physhealthvar = 11
    elif physhealth == '12':
        physhealthvar = 12
    elif physhealth == '13':
        physhealthvar = 13
    elif physhealth == '14':
        physhealthvar = 14
    elif physhealth == '15':
        physhealthvar = 15
    elif physhealth == '16':
        physhealthvar = 16
    elif physhealth == '17':
        physhealthvar = 17
    elif physhealth == '18':
        physhealthvar = 18
    elif physhealth == '19':
        physhealthvar = 19
    elif physhealth == '20':
        physhealthvar = 20
    elif physhealth == '21':
        physhealthvar = 21
    elif physhealth == '22':
        physhealthvar = 22
    elif physhealth == '23':
        physhealthvar = 23
    elif physhealth == '24':
        physhealthvar = 24
    elif physhealth == '25':
        physhealthvar = 25
    elif physhealth == '26':
        physhealthvar = 26
    elif physhealth == '27':
        physhealthvar = 27
    elif physhealth == '28':
        physhealthvar = 28
    elif physhealth == '29':
        physhealthvar = 29
    elif physhealth == '30':
        physhealthvar = 30


    mentalhealth = st.selectbox(
    'Now thinking about your mental health, which includes stress, depression, and problems with emotion for how many days during the past 30 have you had them?',
    ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'))
    if mentalhealth=='1':
        mentalhealthvar=1
    elif mentalhealth=='2':
        mentalhealthvar=2
    elif mentalhealth=='3':
        mentalhealthvar=3
    elif mentalhealth=='4':
        mentalhealthvar=4
    elif mentalhealth=='5':
        mentalhealthvar=5
    elif mentalhealth == '6':
        mentalhealthvar = 6
    elif mentalhealth == '7':
        mentalhealthvar = 7
    elif mentalhealth == '8':
        mentalhealthvar = 8
    elif mentalhealth == '9':
        mentalhealthvar = 9
    elif mentalhealth == '10':
        mentalhealthvar = 10
    elif mentalhealth == '11':
        mentalhealthvar = 11
    elif mentalhealth == '12':
        mentalhealthvar = 12
    elif mentalhealth == '13':
        mentalhealthvar = 13
    elif mentalhealth == '14':
        mentalhealthvar = 14
    elif mentalhealth == '15':
        mentalhealthvar = 15
    elif mentalhealth == '16':
        mentalhealthvar = 16
    elif mentalhealth == '17':
        mentalhealthvar = 17
    elif mentalhealth == '18':
        mentalhealthvar = 18
    elif mentalhealth == '19':
        mentalhealthvar = 19
    elif mentalhealth == '20':
        mentalhealthvar = 20
    elif mentalhealth == '21':
        mentalhealthvar = 21
    elif mentalhealth == '22':
        mentalhealthvar = 22
    elif mentalhealth == '23':
        mentalhealthvar = 23
    elif mentalhealth == '24':
        mentalhealthvar = 24
    elif mentalhealth == '25':
        mentalhealthvar = 25
    elif mentalhealth == '26':
        mentalhealthvar = 26
    elif mentalhealth == '27':
        mentalhealthvar = 27
    elif mentalhealth == '28':
        mentalhealthvar = 28
    elif mentalhealth == '29':
        mentalhealthvar = 29
    elif mentalhealth == '30':
        mentalhealthvar = 30

    
    submit = st.button("Submit")
    if submit:
        dataset = pd.read_csv('diabetesdata.csv')
        X = dataset.iloc[:, 1:18].values
        y = dataset.iloc[:, 0].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        ann.fit(X_train, y_train, batch_size = 32, epochs = 3)



        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred_prob = ann.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1) 
        prediciton = np.argmax(ann.predict([[bpvar, clvar, clcheckvar, bmi, smokevar, strokevar, heartdisvar, physvar, fruitsvar, veggiesvar, alcoholvar, genhealthvar, mentalhealthvar, physhealthvar, diffwalkvar, sexvar, age]]))
        if prediciton==[0]:
            st.write("***You dont have chances of being diabetic***")
            st.markdown(":green[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
            st.write("Accuray of prediction is : ")
            st.write((accuracy_score(y_test, y_pred))*100)
        elif prediciton==[1]:
            st.write("***You may have chances of being pre-diabetic***")
            st.markdown(":yellow[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
            st.write("Accuray of prediction is : ")
            st.write((accuracy_score(y_test, y_pred))*100)
        elif prediciton==[2]:
            st.write("***You may have chances of being diabetic***")
            st.markdown(":red[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
            st.write("Accuray of prediction is : ")
            st.write((accuracy_score(y_test, y_pred))*100)
elif nav=="Home":
    st.title("Diabetes Dashboard")
    st_lottie(url1_json) 
    st.write('''***Welcome to Your Diabetes Dashboard!***

Embark on a journey to better health with our Diabetes Dashboard—a comprehensive platform designed to empower you with information, awareness, and diabetes prediction.

***Explore Information***:
Get to know about Diabetes and its types

***Ignite General Awareness***:
Get to know about information and precautive measures regarding diabetes and lifestyle recommendations.

***Predict Your Path***:
Take control of your health with our advanced prediction tool. Assess your diabetes risk by inputting key factors, 

Your Diabetes Dashboard is your ally in managing diabetes and promoting overall well-being. Explore, learn, and embrace a healthier life with the support of our comprehensive features. Welcome to a community that cares about your health and well-being!
''')

elif nav=="General Information":
    st.title("General Information")
    st_lottie(url2_json)
    st.write('''***Diabetes, a complex and chronic medical condition, arises from the intricate interplay of various factors impacting the body's ability to regulate blood sugar (glucose) effectively. Glucose, a vital energy source for cells, relies on insulin—produced by the pancreas—to facilitate its absorption and utilization by cells throughout the body. This intricate balance is disrupted in diabetes, leading to prolonged elevated blood sugar levels.***

***1. Type 1 Diabetes***: This form of diabetes results from an autoimmune response where the body's immune system erroneously attacks and destroys insulin-producing cells in the pancreas. Individuals with Type 1 diabetes are reliant on external insulin sources, typically administered through injections or insulin pumps, to meticulously manage their blood sugar levels.

***2. Type 2 Diabetes***: More prevalent than Type 1, Type 2 diabetes tends to develop gradually. In this scenario, the body develops resistance to the effects of insulin, and the pancreas may struggle to produce sufficient insulin to maintain optimal blood sugar levels. Contributing factors include lifestyle choices, genetic predisposition, and obesity. Managing Type 2 diabetes often involves a comprehensive approach, encompassing lifestyle modifications, oral medications, and, in some cases, insulin therapy.

***3. Prediabetes***: An intermediate stage in the continuum of glucose regulation, prediabetes signals elevated blood sugar levels that are not yet within the diabetic range. It serves as a crucial warning sign, indicating an increased risk of progressing to Type 2 diabetes. Lifestyle interventions, including adopting a nutritious diet, engaging in regular physical activity, and managing body weight, play pivotal roles in preventing the progression to full-blown diabetes.

Beyond the specific types, effective diabetes management involves a holistic approach to mitigate potential complications. Complications can include heart disease, kidney damage, nerve issues, and vision problems. Regular monitoring of blood sugar levels, adherence to prescribed medications, and embracing a balanced lifestyle are integral components of this proactive approach.

Understanding the intricacies of diabetes and prediabetes empowers individuals to make informed decisions about their health. This journey toward optimal health not only involves managing the condition but also adopting a proactive stance to prevent and minimize potential complications, fostering a lifestyle that nurtures overall well-being.''')

elif nav=="General Awareness":
    st.title("Genaral Awareness")

    st.write('''Understanding Diabetes: Awareness and Precautions
***Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). This can lead to a variety of symptoms and health complications if not managed properly. Here's an overview of general awareness and precautions related to diabetes***:

Types of Diabetes:

***Type 1 diabetes***: The body doesn't produce enough insulin, a hormone that helps cells absorb glucose.
             
***Type 2 diabetes***: The body either doesn't use insulin effectively or doesn't produce enough.
             
***Gestational diabetes***: Develops during pregnancy but usually goes away afterward.
             
General Awareness:

***Risk factors***: Age, family history, ethnicity, weight, physical activity level, and diet can all increase your risk.
Symptoms: Frequent urination, increased thirst, fatigue, blurred vision, slow healing wounds, and tingling or numbness in the hands or feet can be signs of diabetes.
Complications: Unmanaged diabetes can lead to serious health problems like heart disease, stroke, kidney disease, nerve damage, and vision loss.
Early detection and management: Getting diagnosed and starting treatment early can help prevent complications and improve quality of life.
Precautions:

***Lifestyle changes***: Eating a balanced diet, being physically active, and managing stress are crucial for managing diabetes and reducing your risk of complications.
Regular checkups: Schedule regular appointments with your doctor to monitor your blood sugar levels, discuss your management plan, and get recommended screenings.
Knowledge is power: Learn as much as you can about diabetes from reliable sources like the American Diabetes Association (ADA) or the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
''')

    

    st.write('''***Diet***:
Choose whole foods: Focus on fruits, vegetables, whole grains, and lean protein sources. These foods are rich in nutrients and fiber, which can help regulate blood sugar.
Limit processed foods: Sugary drinks, processed snacks, and fast food are often high in unhealthy fats, added sugars, and refined carbohydrates, which can worsen blood sugar control.
Mindful portions: Pay attention to serving sizes and use tools like measuring cups or plates to avoid overeating.
''')

    st_lottie(url4_json)

    st.write('''***Physical Activity***:
Aim for at least 30 minutes of moderate-intensity exercise most days of the week. Brisk walking, swimming, cycling, dancing, or team sports are all great options.
Find activities you enjoy: You're more likely to stick with an exercise routine if you find it fun and engaging.
Start slow and gradually increase: Don't try to do too much too soon. Listen to your body and gradually increase the intensity and duration of your workouts.
''')

    st_lottie(url5_json)

    st.write('''***Stress Management***:
Identify your stress triggers: What situations or activities tend to make you feel stressed?
Develop healthy coping mechanisms: Explore relaxation techniques like deep breathing, meditation, yoga, or spending time in nature.
Connect with others: Social support is crucial for managing stress. Talk to friends, family, or join a support group.
''')
    st_lottie(url6_json)
    

    st.write('''***Limit Alcohol and Quit Smoking***:
Moderation in alcohol consumption and the decision to quit smoking not only contribute to overall health but are pivotal in diabetes prevention and management. Excessive alcohol intake can disrupt blood sugar levels, while smoking has been linked to insulin resistance. Embracing a smoke-free and moderate lifestyle not only reduces diabetes risk but also enhances the effectiveness of diabetes management strategies, fostering a healthier, more resilient life.   

***Remember:
This information is for general awareness only and should not be considered a substitute for professional medical advice.
If you have concerns about your diabetes risk or are experiencing symptoms, please consult a healthcare professional.***''')
    st_lottie(url7_json)

elif nav=="About":
    st.title("About")
    st.divider()
    st.subheader('Created by ❤️ by Ishaan Verma')
    st.divider()
    st.write("This website is created with the intent of providing a comprehensive place to know about diabetes and its types, to know about the recommended lifestyle to follow and other preautions. Along with the diabetes prediction feature this website is a one stop solution to know about diabetes")
    st.divider()
    st.write(''' ***Tech stack***: 
             This website is made completely using python and is made on streamlit framework.
            The prediction model uses Logistic Regression  to predit diabetes. The model has an accuracy rate of 84.6499%. The dataset on which the model is trained on has 253681 patient data.
             ''')



    


    


    







