#importing the libraries

import streamlit as st
from PIL import Image
import json 
import requests 
from streamlit_lottie import st_lottie 
import tensorflow as tf

#loading animations

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

urltum = requests.get( 
    "https://lottie.host/3f4dfb31-cb6a-4acc-816a-804b97f89299/SvVtWlanni.json") 
urltum_json = dict() 
  
if urltum.status_code == 200: 
    urltum_json = urltum.json() 
else: 
    print("Error in the URL") 

#Creating streamlit app


st.sidebar.title('Navigation')
nav = st.sidebar.radio('Go to: ',('Home', 'General Information', 'Predictions', 'About'))
if nav=='Predictions':

    option =st.selectbox("What do you want to know : ", ("Diabetes Prediciton" ,"Brain tumor detection","Stress detection through sleep pattern","Human Stress Detection based on physiological data", "Alzheimer Prediction", "Heart Disease Prediction"))
    
    
    if option=="Heart Disease Prediction":
        st.title("Heart Disease Prediction")
        st.write('''Here on our Heart Disease Prediction page, we're using machine learning smart tools to help predict
                diabetes. Remember, these tools are just helpers and not a substitute for talking to a doctor. 
                It's super important to always chat with a healthcare professional for the best advice about your health.''')
        
        sex = st.radio('Gender', ['male', 'female'])
        if sex=='male':
            sexvar = 1
        elif sex=='female':
            sexvar = 0
        agevar = st.number_input("Enter your Age", step=1)
        chestpain = st.selectbox(
        'Enter your chest pain on a scale of 1-5 (1-No pain, 5-High Pain)',
        ('1', '2', '3', '4', '5'))
        if chestpain=='1':
            chestpainvar=1
        elif chestpain=='2':
            chestpainvar=2
        elif chestpain=='3':
            chestpainvar=3
        elif chestpain=='4':
            chestpainvar=4
        elif chestpain=='5':
            chestpainvar=5

        bpvar = st.number_input("Enter your BP (Systolic)", step=1)
        cholestrolvar = st.number_input("Enter your Cholestrol Level", step=1)
        fbs = st.selectbox("Is your FBS over 1 ? ", ("Yes", "No"))
        #ekgresult, exercise angina left
        if fbs=="Yes":
            fbsoveronevar = 1
        else:
            fbsoveronevar = 0
        ecgvar = st.number_input("Enter your rest ECG level (0-2)")
        exerciseangina = st.selectbox("Do you have exercise induced Angina ? ", ("Yes", "No"))
        if fbs=="Yes":
            exerciseanginavar = 1
        else:
            exerciseanginavar = 0
        maxhrvar = st.number_input("Enter your Maximum HR", step=1)
        stdepressionvar = st.number_input("Enter your ST Depression", step=1)
        slopeofstvar =st.number_input("Enter the slope of ST", step=1)
        noofvesselsflurovar =st.number_input("Enter the Number of vessels fluro", step=1)
        thalliumvar =st.number_input("Enter your Thallium", step=1)
        submit = st.button("Submit")
        st.write("You must wait for few seconds for your prediction")
        if submit:
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            # Importing the dataset
            dataset = pd.read_csv('Heart_Disease_Prediction.csv')
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)


            #   Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            print(X_train)
            print(X_test)

            # Training the Random Forest Classification model on the Training set
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)

            # Predicting a new result

            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
        

            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            new_data= [agevar, sexvar, chestpainvar, bpvar, cholestrolvar, fbsoveronevar, ecgvar, maxhrvar, exerciseanginavar, stdepressionvar, slopeofstvar, noofvesselsflurovar, thalliumvar]

            result = classifier.predict(sc.transform([new_data]))
            if result==[1]:
                st.markdown(":red[***You have chances of having heart disease***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
                st.write("Accuray of prediction is : ")
                st.write(acc*100)
            elif result==[0]:
                st.markdown(":green[***You dont have chances of having heart disease***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
                st.write("Accuray of prediction is : ")
                st.write(acc*100)

    elif option=="Human Stress Detection based on physiological data":

        st.title("Stress prediction using Physiological Data")
        st.write('''This is our stress detection page for detecting stress usinf physiological factors.''')
                 

        st.write('''Humidity - When you feel stress, your body temperature rises, prompting your sweat glands to kick in. This sweat is considered to be the Humidity Level.''')
                 
        st.write('''Temperature - Body Temperature of a person during stress.''')
                 
        st.write('''Stepcount - Number of steps covered by the person during his stressful situation.''')        
                 
        st.write('''Stress_Level - Based on all the above 3 Factors our Stress Level will be predicted as High, Medium and Low accordingly.''')       
                 
        
        humidityvar = st.number_input("Enter your body humidity level during stress")
        temperaturevar = st.number_input("Enter your body temperature during stress in Fahrenheit")
        stepsvar =st.number_input("Number of steps covered by the person during his stressful situation")

        



        submit = st.button("Submit")
        st.write("You must wait for few seconds for your prediction")
        if submit:

            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            # Importing the dataset
            dataset = pd.read_csv('Stress-Lysis.csv')
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            #from sklearn.preprocessing import LabelEncoder
            #le = LabelEncoder()
            #y = le.fit_transform(y)


            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            print(X_train)
            print(X_test)

            # Training the Random Forest Classification model on the Training set
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)



            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            

            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)


            # Predicting a new result
            result = classifier.predict(sc.transform([[humidityvar, temperaturevar, stepsvar]]))
            if result==[0]:
                st.markdown(":green[***You dont low stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
                st.write("Accuray of prediction is : ")
                st.write(acc*100)
            elif result==[1]:
                st.markdown(":yellow[***You dont medium stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
                st.write("Accuray of prediction is : ")
                st.write(acc*100)
            elif result==[2]:
                st.markdown(":red[***You dont high stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute to medical proffessional]")
                st.write("Accuray of prediction is : ")
                st.write(acc*100)

    elif option=="Stress detection through sleep pattern":
        st.title("Stress Prediction using sleep Pattern")
        st.write('''
    Considering today’s lifestyle, people just sleep forgetting the benefits sleep provides to the human body. Smart-Yoga Pillow (SaYoPillow) is proposed to help in understanding the relationship between stress and sleep and to fully materialize the idea of “Smart-Sleeping” by proposing an edge device. An edge processor with a model analyzing the physiological changes that occur during sleep along with the sleeping habits is proposed. Based on these changes during sleep, stress prediction for the following day is proposed. The secure transfer of the analyzed stress data along with the average physiological changes to the IoT cloud for storage is implemented. A secure transfer of any data from the cloud to any third-party applications is also proposed. A user interface is provided allowing the user to control the data accessibility and visibility. SaYoPillow is novel, with security features as well as consideration of sleeping habits for stress reduction''')

        st.write('''you will see the relationship between the parameters- snoring range of the user, respiration rate, body temperature, limb movement rate, blood oxygen levels, eye movement, number of hours of sleep, heart rate and Stress Levels          
''')
        
        snoringratevar = st.number_input("Enter your snoring rate as read by sensor")
        respirationratevar = st.number_input("Enter your respiration rate as read by sensor")
        bodytempvar = st.number_input("Enter your body temperature as read by sensor")
        limbmovementvar = st.number_input("Enter your limb movement as read by sensor")
        bloodoxygenvar = st.number_input("Enter your blood oxygen level as read by sensor")
        eyemovementvar = st.number_input("Enter your eye movement level as read by sensor")
        sleepinghoursvar = st.number_input("Enter your number of sleeping hours as read by sensor")
        heartratevarvar = st.number_input("Enter your heart rate as read by sensor")
        submit = st.button("Submit")
        st.write("You must wait for few seconds for your prediction")

        if submit:


            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            # Importing the dataset
            dataset = pd.read_csv('SaYoPillow.csv')
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            #from sklearn.preprocessing import LabelEncoder
            #le = LabelEncoder()
            #y = le.fit_transform(y)


            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            print(X_train)
            print(X_test)

            # Training the Random Forest Classification model on the Training set
            from sklearn.svm import SVC
            classifier = SVC(kernel = 'linear', random_state = 0)
            classifier.fit(X_train, y_train)



            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            

            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)


            #Predicting a new result
            

            result = classifier.predict(sc.transform([[snoringratevar, respirationratevar, bodytempvar, limbmovementvar, bloodoxygenvar, eyemovementvar, sleepinghoursvar, heartratevarvar]]))
            if result == [0]:
                st.markdown(":green[***You have a low/normal stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute for medical professional]")
                st.write("Accuracy of prediction is: ")
                st.write(acc * 100)
            elif result == [1]:
                st.markdown(":yellow[***You have a medium-low stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute for medical professional]")
                st.write("Accuracy of prediction is: ")
                st.write(acc * 100)
            elif result == [2]:
                st.markdown(":orange[***You have a medium stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute for medical professional]")
                st.write("Accuracy of prediction is: ")
                st.write(acc * 100)
            elif result == [3]:
                st.markdown(":red[***You have a medium-high stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute for medical professional]")
                st.write("Accuracy of prediction is: ")
                st.write(acc * 100)
            elif result == [4]:
                st.markdown(":red[***You have a high stress level***]")
                st.markdown(":orange[You must always consult a doctor regarding your health. Do not take this tool as a substitute for medical professional]")
                st.write("Accuracy of prediction is: ")
                st.write(acc * 100)

    elif option=="Brain tumor detection":

        import streamlit as st
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        import numpy as np
        import os

        def load_and_preprocess_images(directory, target_size=(64, 64)):
            images = []
            labels = []
            label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
            for label, label_index in label_map.items():
                label_dir = os.path.join(directory, label)
                for filename in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, filename)
                    image = load_img(img_path, target_size=target_size)
                    image = img_to_array(image) / 255.0  # Normalization
                    images.append(image)
                    labels.append(label_index)
            return np.array(images), np.array(labels)

        def build_model(weights_path):
            cnn = tf.keras.models.Sequential()
            cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
            cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
            cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            cnn.add(tf.keras.layers.Flatten())
            cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
            cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))
            cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
            # Load pre-trained weights
            cnn.load_weights(weights_path)
    
            return cnn

        def predict_single_image(image, model):
            image = img_to_array(image.resize((64, 64))) / 255.0
            image = np.expand_dims(image, axis=0)
            result = model.predict(image)
            predicted_label_index = np.argmax(result)
            label_map_inverse = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
            prediction = label_map_inverse[predicted_label_index]
            return prediction

        def main():
            st.title('Brain Tumor Classification')
            uploaded_file = st.file_uploader("Choose an image...", type="jpg")
            # Display the submit button
            submitted = st.button("Submit")
            st.write("It may take 2-3 minutes to predict the image, after all it's a computer and not a MBBS graduate 😜")
            if submitted:
                # Load and preprocess the data
                training_images, training_labels = load_and_preprocess_images('Traintumordetection')
                test_images, test_labels = load_and_preprocess_images('Testtumordetection')

                # Build the model with pre-trained weights
                model = build_model("weights.25-0.13.h5")

                # File uploader for image selection
                if uploaded_file is not None:
                    image = load_img(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    prediction = predict_single_image(image, model)
                    st.write('Prediction:', prediction)

        if __name__ == '__main__':
            main()


         

                
                        


    elif option=="Alzheimer Prediction":

        st.title("Alzheimer Disease Prediction")
        import streamlit as st
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array, load_img
        import numpy as np
        import os

        def load_and_preprocess_images(directory, target_size=(64, 64)):
            images = []
            labels = []
            label_map = {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
            for label, label_index in label_map.items():
                label_dir = os.path.join(directory, label)
                for filename in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, filename)
                    image = load_img(img_path, target_size=target_size)
                    image = img_to_array(image) / 255.0  # Normalization
                    images.append(image)
                    labels.append(label_index)
            return np.array(images), np.array(labels)

        def build_model():
            cnn = tf.keras.models.Sequential()
            cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
            cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
            cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            cnn.add(tf.keras.layers.Flatten())
            cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
            cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))  
            cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
            return cnn

        def predict_single_image(image, model):
            image = img_to_array(image.resize((64, 64))) / 255.0
            image = np.expand_dims(image, axis=0)
            result = model.predict(image)
            predicted_label_index = np.argmax(result)
            label_map_inverse = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
            prediction = label_map_inverse[predicted_label_index]
            return prediction

        def main():
            
            uploaded_file = st.file_uploader("Choose an image...", type="jpg")
            # Display the submit button
            submitted = st.button("Submit")
            st.write("It may take 2-3 minutes to predict the image, afterall its a computer and not a MBBS graduate 😜")
            if submitted:
                # Load and preprocess the data
                training_images, training_labels = load_and_preprocess_images('trainalzheimer')
                test_images, test_labels = load_and_preprocess_images('testalzheimer')

                # Build the model
                model = build_model()

                # Train the model
                model.fit(x=training_images, y=training_labels, epochs=25, validation_data=(test_images, test_labels))
    
                # File uploader for image selection
                

                if uploaded_file is not None:
                    image = load_img(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    prediction = predict_single_image(image, model)
                    st.write('Prediction:', prediction)
    
        if __name__ == '__main__':
            main()

    elif option=="Diabetes Prediciton":

        st.title("Diabetes Prediction")
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
        ('0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'))
        if physhealth=='1':
            physhealthvar=1
        elif physhealth=='2':
            physhealthvar=2
        elif physhealth=='0':
            physhealthvar=0
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
        ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'))
        if mentalhealth=='1':
            mentalhealthvar=1
        elif mentalhealth=='2':
            mentalhealthvar=2
        elif mentalhealth=='0':
            mentalhealthvar=0
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
        st.write("You must wait for 20 seconds for your prediction")
        if submit:
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd
            #Creating ML Model
            dataset = pd.read_csv('diabetesdata.csv')
            X = dataset.iloc[:, 1:18].values
            y = dataset.iloc[:, 0].values

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
            classifier.fit(X_train, y_train)



            from sklearn.metrics import confusion_matrix, accuracy_score
            y_pred = classifier.predict(X_test)
            print(accuracy_score(y_test, y_pred))
            prediciton = classifier.predict([[bpvar, clvar, clcheckvar, bmi, smokevar, strokevar, heartdisvar, physvar, fruitsvar, veggiesvar, alcoholvar, genhealthvar, mentalhealthvar, physhealthvar, diffwalkvar, sexvar, age]])
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
    st.title("HealthMAX")
    st_lottie(url1_json) 
    st.write('''***Welcome to HealthMAX - Your Ultimate Health Companion! 🩺***

At HealthMAX, we're dedicated to empowering you with cutting-edge technology to take control of your health like never before. Our innovative web app offers a suite of powerful features designed to predict, detect, and manage various health conditions, ensuring you stay one step ahead in your wellness journey.

With our advanced algorithms, we provide accurate predictions for diabetes, heart disease, tumor detection, and mental stress. Whether you're looking to assess your risk factors or seeking early detection, HealthMAX equips you with the tools you need to make informed decisions about your health.

Say goodbye to uncertainty and hello to proactive health management with HealthMAX. Try now and embark on a journey towards a healthier, happier you 😊.
''')

elif nav=="General Information":
    st.title("General Information 📖🔬")
    st_lottie(url2_json)
    genopt = st.selectbox("What do you want to know about", ("Diabetes", "Brain Tumor", "Heart Diseases", "Alzheimer", "Mental Stress"))
    if genopt=="Diabetes":
        st.title("1. DIABETES")
        st.write('''Understanding Diabetes: Awareness and Precautions''')
                 
        st.write('''***Risk factors***: Age, family history, ethnicity, weight, physical activity level, and diet can all increase your risk.
    Symptoms: Frequent urination, increased thirst, fatigue, blurred vision, slow healing wounds, and tingling or numbness in the hands or feet can be signs of diabetes.
    Complications: Unmanaged diabetes can lead to serious health problems like heart disease, stroke, kidney disease, nerve damage, and vision loss.
    Early detection and management: Getting diagnosed and starting treatment early can help prevent complications and improve quality of life.
    Precautions:''')
                 
        st.write('''***Lifestyle changes***: Eating a balanced diet, being physically active, and managing stress are crucial for managing diabetes and reducing your risk of complications.
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
    Moderation in alcohol consumption and the decision to quit smoking not only contribute to overall health but are pivotal in diabetes prevention and management. Excessive alcohol intake can disrupt blood sugar levels, while smoking has been linked to insulin resistance. Embracing a smoke-free and moderate lifestyle not only reduces diabetes risk but also enhances the effectiveness of diabetes management strategies, fostering a healthier, more resilient life. ''')  

        st.write('''***Remember:
    This information is for general awareness only and should not be considered a substitute for professional medical advice.
    If you have concerns about your diabetes risk or are experiencing symptoms, please consult a healthcare professional.***''')
        st_lottie(url7_json)
    if genopt=="Brain Tumor":
        st.title("2. Brain Tumor")
        st_lottie(urltum_json)
        st.write('''
    **Understanding Brain Tumors:**

    🧠 **Introduction:**
    A brain tumor is an abnormal growth of cells in the brain. These growths can be benign (non-cancerous) or malignant (cancerous). They can originate in the brain itself (primary tumors) or spread to the brain from other parts of the body (secondary tumors or metastases). Brain tumors can affect various functions of the brain, leading to a wide range of symptoms and complications.

    🔍 **Types of Brain Tumors:**
    1. **Gliomas:** These tumors originate from glial cells, which are supporting cells in the brain. Gliomas can be further classified into subtypes such as astrocytomas, oligodendrogliomas, and ependymomas.
    2. **Meningiomas:** Arising from the meninges, the protective membranes covering the brain and spinal cord, meningiomas are usually benign.
    3. **Metastatic Tumors:** These tumors spread to the brain from other parts of the body, such as the lungs, breast, or colon.
    4. **Pituitary Tumors:** These tumors develop in the pituitary gland, which is located at the base of the brain and controls hormone production.
    5. **Medulloblastomas:** These are fast-growing tumors that mainly occur in the cerebellum, which is responsible for coordination and balance.
    6. **Schwannomas:** These tumors arise from Schwann cells, which produce the insulating myelin sheath around nerves. They often affect the nerves associated with hearing and balance (vestibular schwannomas).

    📑 **Symptoms:**
    The symptoms of a brain tumor can vary depending on its size, location, and rate of growth. Common symptoms include:
    - Headaches
    - Seizures
    - Nausea and vomiting
    - Weakness or numbness in the limbs
    - Changes in vision or hearing
    - Difficulty with balance and coordination
    - Cognitive and personality changes
    - Memory problems

    🩺 **Diagnosis:**
    Diagnosing a brain tumor typically involves a combination of imaging tests such as MRI (Magnetic Resonance Imaging) or CT (Computed Tomography) scans, along with a neurological examination to assess cognitive function, coordination, and reflexes. In some cases, a biopsy may be necessary to determine the type of tumor and its grade (how aggressive it is).

    💉 **Treatment:**
    Treatment options for brain tumors depend on several factors, including the type, location, size, and grade of the tumor, as well as the patient's overall health and preferences. Treatment may involve:
    - **Surgery:** Surgical removal of the tumor is often the first-line treatment if the tumor is accessible and can be safely removed without causing significant damage to surrounding brain tissue.
    - **Radiation Therapy:** High-energy beams are used to target and destroy cancer cells. This can be done externally (external beam radiation therapy) or internally (brachytherapy).
    - **Chemotherapy:** Medications are used to kill cancer cells or stop them from growing. Chemotherapy may be administered orally or intravenously.
    - **Targeted Therapy:** Drugs are designed to target specific abnormalities in cancer cells, such as mutations or proteins that promote growth.

    🧠 **Prognosis:**
    The prognosis for a brain tumor varies widely depending on factors such as the type, grade, and stage of the tumor, as well as the age and overall health of the patient. Some brain tumors, particularly benign ones that can be completely removed with surgery, have a good prognosis with long-term survival. However, malignant brain tumors tend to be more aggressive and may have a poorer prognosis, especially if they are diagnosed at an advanced stage or if they recur after treatment.

    ⚠️ **Complications:**
    Brain tumors can lead to various complications, including:
    - Neurological deficits such as paralysis, speech difficulties, or cognitive impairment
    - Increased intracranial pressure, which can cause symptoms such as headaches, nausea, and vision changes
    - Seizures
    - Hormonal imbalances
    - Side effects of treatment, such as hair loss, fatigue, and increased risk of infections

    👩‍⚕️ **Support and Coping:**
    Living with a brain tumor can be challenging, both physically and emotionally, for patients and their families. It's essential to have a strong support system and access to resources such as counseling, support groups, and palliative care services to help manage symptoms, navigate treatment decisions, and cope with the emotional impact of the diagnosis.

    🔬 **Research and Advances:**
    Ongoing research into the causes, prevention, and treatment of brain tumors continues to advance our understanding of these complex diseases. Emerging therapies, such as immunotherapy and targeted molecular therapies, hold promise for improving outcomes and quality of life for patients with brain tumors.

    In conclusion, brain tumors are a diverse group of conditions that require a multidisciplinary approach to diagnosis and treatment. With advances in medical technology and ongoing research, there is hope for better outcomes and improved quality of life for individuals affected by these challenging conditions.
    ''')
    if genopt=="Heart Diseases":
        st.title("3. Heart Diseases")
        st.write('''
    ### 💔 Comprehensive Guide to Heart Diseases 💔

    Heart diseases, also known as cardiovascular diseases, encompass a range of conditions that affect the heart and blood vessels. These conditions can significantly impact one's health and quality of life, and they are a leading cause of death worldwide. Understanding the different types of heart diseases, their causes, symptoms, and prevention strategies is crucial for maintaining heart health.

    #### Types of Heart Diseases:

    1. **Coronary Artery Disease (CAD):** CAD occurs when the blood vessels that supply blood to the heart muscle become narrowed or blocked due to the buildup of plaque (atherosclerosis). This can lead to chest pain (angina), heart attack, or heart failure.

    2. **Heart Attack:** A heart attack, also known as myocardial infarction, occurs when blood flow to a part of the heart is blocked, leading to the death of heart muscle cells. Symptoms include chest pain, shortness of breath, nausea, and sweating.

    3. **Heart Failure:** Heart failure occurs when the heart cannot pump enough blood to meet the body's needs. It can result from conditions such as CAD, high blood pressure, or heart muscle damage from previous heart attacks. Symptoms include fatigue, shortness of breath, and swelling in the legs and abdomen.

    4. **Arrhythmias:** Arrhythmias are abnormalities in the heart's rhythm, which can cause it to beat too fast (tachycardia), too slow (bradycardia), or irregularly. Symptoms may include palpitations, dizziness, fainting, or chest discomfort.

    5. **Heart Valve Disease:** Heart valve diseases involve damage to or defects in the heart valves, which disrupt the flow of blood through the heart. Symptoms may include chest pain, fatigue, shortness of breath, and swelling of the ankles and feet.

    6. **Congenital Heart Defects:** These are heart abnormalities present at birth, which can affect the heart's structure and function. They range from mild, asymptomatic conditions to severe defects that require surgical intervention.

    #### Causes of Heart Diseases:

    - **Unhealthy Lifestyle:** Factors such as poor diet, lack of exercise, smoking, excessive alcohol consumption, and stress can increase the risk of developing heart diseases.
    
    - **Medical Conditions:** Conditions such as high blood pressure, high cholesterol, diabetes, obesity, and metabolic syndrome can contribute to the development of heart diseases.
    
    - **Genetic Factors:** Some heart diseases have a genetic component, meaning they can run in families and increase the risk of developing cardiovascular conditions.

    #### Symptoms of Heart Diseases:

    Symptoms of heart diseases can vary depending on the specific condition but may include:

    - Chest pain or discomfort
    - Shortness of breath
    - Fatigue
    - Palpitations (irregular heartbeats)
    - Dizziness or lightheadedness
    - Swelling in the legs, ankles, or abdomen

    #### Prevention and Management:

    - **Healthy Lifestyle:** Adopting a healthy diet rich in fruits, vegetables, whole grains, and lean proteins, along with regular exercise and stress management, can significantly reduce the risk of heart diseases.

    - **Regular Check-ups:** Regular medical check-ups, including blood pressure and cholesterol screenings, can help detect and manage risk factors for heart diseases.

    - **Smoking Cessation:** Quitting smoking and avoiding exposure to secondhand smoke is crucial for heart health.

    - **Medication:** In some cases, medications such as statins, blood pressure medications, or antiplatelet drugs may be prescribed to manage risk factors or treat existing heart conditions.

    - **Surgical Interventions:** For severe cases, surgical procedures such as angioplasty, coronary artery bypass grafting (CABG), valve repair or replacement, or implantation of pacemakers or defibrillators may be necessary.

    #### Conclusion:

    Heart diseases are a significant health concern worldwide, but many of them are preventable through lifestyle modifications and proper management of risk factors. By understanding the types, causes, symptoms, and prevention strategies of heart diseases, individuals can take proactive steps to protect their heart health and live longer, healthier lives. Remember, a healthy heart is a happy heart! ❤️
    ''')
    if genopt=="Alzheimer":

        st.title("4. Alzheimer")
        st.write('''
    ### Understanding Alzheimer's Disease 🧠

    Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide, primarily older adults. It is the most common cause of dementia, a syndrome characterized by a decline in cognitive function severe enough to interfere with daily life. Alzheimer's gradually impairs memory, thinking skills, and eventually the ability to carry out even simple tasks. Here's a comprehensive overview of this complex condition:

    #### What Happens in Alzheimer's Disease?

    Alzheimer's disease involves the accumulation of abnormal protein deposits in the brain, primarily beta-amyloid plaques and tau tangles. These deposits disrupt communication between nerve cells (neurons) and ultimately cause cell death. As the disease progresses, brain tissue shrinks, leading to widespread brain damage.

    #### Symptoms and Stages 📉

    1. **Early Stage**: In the early stages, individuals may experience subtle memory lapses and difficulty remembering recent events or conversations. They might misplace items or struggle to find the right words. 🤔

    2. **Middle Stage**: As Alzheimer's advances, symptoms become more pronounced. Memory loss worsens, and individuals may have trouble recognizing family and friends. They might exhibit changes in behavior and personality, becoming more anxious, agitated, or withdrawn. 😟

    3. **Late Stage**: In the late stages, individuals lose the ability to communicate effectively and require assistance with basic activities such as eating, dressing, and toileting. They may also experience significant physical decline and become vulnerable to infections. 😔

    #### Risk Factors 📊

    Several factors can increase the risk of developing Alzheimer's disease:

    - **Age**: Advancing age is the most significant risk factor. The likelihood of developing Alzheimer's doubles approximately every five years after age 65. 🕰️
    
    - **Genetics**: While most cases of Alzheimer's are not directly inherited, having a family history of the disease increases the risk. 👨‍👩‍👦‍👦
    
    - **Lifestyle Factors**: Certain lifestyle choices, such as lack of physical activity, poor diet, smoking, and social isolation, may contribute to the risk of developing Alzheimer's. 🍔🚭

    #### Diagnosis and Treatment 🏥

    Diagnosing Alzheimer's disease involves a thorough assessment of medical history, physical examination, cognitive tests, and sometimes brain imaging studies. While there is currently no cure for Alzheimer's, treatment focuses on managing symptoms and improving quality of life.

    - **Medications**: Cholinesterase inhibitors and memantine are commonly prescribed to help manage cognitive symptoms and behavioral changes. 💊
    
    - **Non-Pharmacological Approaches**: Cognitive stimulation, physical exercise, social engagement, and occupational therapy can all play a role in supporting individuals with Alzheimer's. 🏋️‍♂️🧩

    #### Caregiving Challenges and Support 🤝

    Caring for someone with Alzheimer's disease can be emotionally and physically demanding. Caregivers often face challenges related to behavior management, communication difficulties, and ensuring safety. It's essential for caregivers to seek support from healthcare professionals, support groups, and community resources to prevent burnout and provide the best possible care. 🤲

    #### Research and Hope for the Future 🌟

    Ongoing research into Alzheimer's disease aims to uncover its underlying causes, develop more effective treatments, and ultimately find a cure. Advances in genetics, imaging technology, and biomarker research offer promising avenues for early detection and intervention. While the journey towards a world without Alzheimer's may be long, there is hope on the horizon for better prevention, treatment, and ultimately, a cure. 🧬✨

    In conclusion, Alzheimer's disease is a complex and devastating condition that profoundly impacts individuals, families, and society as a whole. By increasing awareness, supporting research, and providing compassionate care, we can work towards improving the lives of those affected by this challenging disease. 🌈
    ''')
    if genopt=="Mental Stress":

        st.title("5. Mental Stress")
        st_lottie(url6_json)
        st.write('''
    **Navigating the Maze of Mental Stress: Embracing, Coping, and Flourishing!**

    Mental stress 🧠 is an intricate tapestry that entwines itself around individuals from all corners of life. It encompasses a myriad of emotional, psychological, and physiological reactions to external pressures, internal conflicts, or perceived threats. From the whirlwind of daily existence to monumental life transitions, stress can manifest in manifold forms and magnitudes, weaving its intricate web within our thoughts, feelings, and behaviors.

    **Understanding Mental Stress**

    At its core, mental stress is the body's primal response to perceived challenges or threats. When faced with a stressor, the brain orchestrates the body's fight-or-flight response, unleashing a cascade of hormones such as adrenaline and cortisol. These physiological metamorphoses prime us to either confront the stressor head-on or flee from its looming grasp. Nonetheless, chronic or excessive stress can exact a toll on our mental and physical well-being, ushering in a host of health maladies including anxiety, depression, insomnia, and cardiovascular afflictions.

    **Causes of Mental Stress**

    Mental stress is a chameleon, its origins multifarious and labyrinthine. It may emanate from the crucible of work-related pressures, the specter of financial woes, the labyrinth of relationship tumult, the specter of traumatic encounters, or the weight of societal expectations. Additionally, factors such as genetics, personality proclivities, and coping stratagems interlace to shape how individuals perceive and contend with stressors. What may be a surmountable burden for one may burgeon into an overwhelming tempest for another.

    **Effects of Mental Stress**

    The ramifications of mental stress extend far beyond the realm of mere tension or disquietude. Prolonged stress can enfeeble cognitive acuity, memory retention, and discernment faculties. It can also debilitate the immune system, rendering individuals more susceptible to maladies. Furthermore, chronic stress has been correlated with an escalated risk of developing mental health disorders such as depression and anxiety ailments.

    **Coping Strategies**

    Thankfully, there exists a plethora of strategies to adeptly cope with and navigate mental stress:

    1. **Mindfulness and Meditation**: Engaging in practices such as mindfulness meditation cultivates present-moment awareness and diminishes the impact of stress on both mind and body. 🧘‍♂️

    2. **Physical Activity**: Regular exercise releases endorphins, nature's own mood enhancers, while also reducing levels of stress hormones in the body. 🏋️‍♀️

    3. **Healthy Lifestyle Choices**: Consuming a balanced diet, obtaining sufficient sleep, and abstaining from excessive alcohol and caffeine consumption foster better stress management. 🥗

    4. **Social Support**: Nurturing connections with friends, family, or support networks provides emotional succor and practical assistance during tumultuous times. 👫

    5. **Time Management**: Prioritizing tasks, delineating realistic goals, and employing effective time management strategies instill a sense of control and alleviate stress. ⏰

    6. **Professional Help**: Seeking guidance from therapists, counselors, or mental health professionals furnishes invaluable insights and coping mechanisms for traversing stressors. 🤝

    **Thriving Beyond Stress**

    While stress is an inevitable facet of existence, it need not reign supreme over our well-being. By comprehending the essence of stress, implementing wholesome coping mechanisms, and seeking solace when necessary, individuals can navigate life's labyrinthine passages with resilience and flourish amidst adversity. Remember, it's not about vanquishing stress entirely, but rather learning to wield it in a manner that fosters holistic health and well-being. 🌟
    ''')
elif nav=="About":
    st.title("About")
    st.divider()
    st.subheader('Created with ❤️ by Ishaan Verma')
    st.divider()
    st.write("This website is created with the intent of providing a comprehensive place to know about different health-issues, to know about the recommended lifestyle to follow and other preautions. Along with the prediction feature this website is a one stop solution to know about several health related issues")
    st.divider()
    st.write(''' ***Tech stack***: ''')
    st.write('''This website is made completely using python and is made on streamlit framework.''')
    st.write('''The Diabetes prediction model uses K-Nearest Neighbour to predit diabetes. The model has an accuracy rate of 84.94796594134343%. The dataset on which the model is trained on has 253681 patients data.''')
    st.write('''The stress detection through sleep uses a Support Vector Machine Model . The model has an accuracy of 100%''')
    st.write('''The stress detection through physiological data uses a Random Forest model. The model accuracy is 99.8%''')
    st.write('''The heart disease Prediction uses a Random Forest Classifier model. The model as an accuracy of 83.82352941176471%''')
    st.write('''The Alzheimer detection uses a Convolutional Neural Network(CNN). Accuracy on training set is 72.74%''')
    st.write('''The Brain Tumor detection uses a Convolutional Neural Network(CNN). Accuracy on training set is 92.38%''')
