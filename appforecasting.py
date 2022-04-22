import matplotlib.pyplot as plt
import mpld3
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from flask import Markup
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
app = Flask(__name__)
model1 = pickle.load(open('modelheartbeat.pkl', 'rb'))
model2 = pickle.load(open('modeloxygen.pkl', 'rb'))
model3 = pickle.load(open('modelsystolic.pkl', 'rb'))
model4 = pickle.load(open('modeldiastolic.pkl', 'rb'))
model5 = pickle.load(open('modeltemperature.pkl', 'rb'))


@app.route('/')
def home():
    df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20dataset.csv")
    df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
    df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
    df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
    #combining all the  results
    orgdate=list(str(df['DATE'].tail(1)).strip().split(" "))
    date=orgdate[4][:orgdate[4].rfind("\n")]
    orgtime=list(str(df['TIME'].tail(1)).strip().split(" "))
    time=orgtime[4][:orgtime[4].rfind("\n")]
    #temprature
    orgtemp=list(str(df['TEMPERATURE'].tail(1)).strip().split(" "))
    temperature=orgtemp[4][:orgtemp[4].rfind("\n")]
    temperatureresult=model5.predict([[temperature,temperature]])
    #systolic
    orgsys=list(str(df['SYSTOLIC'].tail(1)).strip().split(" "))
    systolic=orgsys[4][:orgsys[4].rfind("\n")]
    print(systolic)
    a=model3.predict([[systolic,systolic]])
    #diastolic
    orgdia=list(str(df['DIASTOLIC'].tail(1)).strip().split(" "))
    diastolic=orgdia[4][:orgdia[4].rfind("\n")] 
    b=model4.predict([[diastolic,diastolic]])
    #blood oxygen
    orgbo=list(str(df['OXYGEN LEVEL'].tail(1)).strip().split(" "))
    bloodoxygen=orgbo[4][:orgbo[4].rfind("\n")]
    bloodoxygenresult=model2.predict([[bloodoxygen,bloodoxygen]])
    #heart beat
    orghb=list(str(df['HEART BEAT'].tail(1)).strip().split(" "))
    heartbeat=orghb[4][:orghb[4].rfind("\n")]
    p=model1.predict([[heartbeat,heartbeat]])
    #condition for printing 
    print(p,a,b,bloodoxygenresult,temperatureresult)
    output ="<br><br><br><p style='padding left:5%;'><center>REPORT</center></p><br><br>Date: "+str(date)+"<br><br>Time:"+str(time)+"<br><br><br>Readings<br><br>HEARTBEAT: "+str(heartbeat)+"<br><br>BLOOD PRESSURE(SYSTOLIC): "+str(systolic)+"<br><br>BLOOD PRESSURE(DIASTOLIC): "+str(diastolic)+"<br><br>BLOOD OXYGEN: "+str(bloodoxygen)+"<br><br>TEMPRATURE: "+str(temperature)+"<br><br>NORMAL RANGE<br><br>HEARTBEAT: 60 to 100 bpm<br><br>BLOOD PRESSURE(SYSTOLIC): 90 to 120<br><br>BLOOD PRESSURE(DIASTOLIC): 60 to 80<br><br>BLOOD OXYGEN: 95.0% to 99.9%<br><br>TEMPRATURE: 97.7 F to 99.0 F<br><br>RESULT<br>"
    fHB,fS,fD,fBO,fT=(float(heartbeat)/100)*10,(float(systolic)/120)*10,(float(diastolic)/80)*5,(float(bloodoxygen)/100)*5,(float(temperature)/97)*5
    if temperatureresult==0.0 and a==0.0 and b==0.0 and bloodoxygenresult==0.0 and p==0.0:
        
        output+="<p style='color:green;background-color:white;'>NORMAL</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR NORMAL RANGE<br><br>Normal HeartBeat:Banana, melons, orange ,sweet potatoes, dairy, whole grains, chicken(8-ounce glass of water)<br><br>Normal BP:egg, chicken, nuts and seeds, fruits and vegetables.<br><br>Normal Temprature:Hot water, water rich foods like cucumber and water melon, green leafy vegetables like spinach, kale, broccoli.<br><br>Normal Blood Oxygen:Beetroot, garlic, leafy greens, pomegranate, cruciferous vegetables, sprouts, meat, nuts, seeds, dates, carrots, banana.<br><br>Medication: Follow Your Regular Medication.<br><br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(int(int(heartbeat)+fHB))+"<br>predicted systolic: "+str(systolic)+" to "+str(int(float(systolic)+fS))+"<br>predicted diastolic: "+str(diastolic)+" to "+str(int(float(diastolic)+fD))+"<br>predicted Blood Oxygen: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br>predicted Temprature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<p style='font-size:100%;color:red;'>These predictions will be apllicable when the above diet followed.</p>"
    if temperatureresult==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL HEARTBEAT</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR ABNORMAL RANGE"
        if heartbeat>100:
            output+="<br><br>Above Normal Range:Omega-3 fatty acids, found in fish, lean meats, nuts, grains and legume. Phenols and tannins found in tea, coffee. and red wine(in moderation).Vitamin A, found in greens. Whole grains. Vitamin C in bean sprouts.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(int(int(heartbeat)-fHB))+" to "+str(heartbeat)+"<br><br>Disease Related to High Pulse<br><br>Tachycardia:<br><br>disease: Stroke, Heart failure, Sudden death, Blood clots<br><br>Symptoms:<br>a fast pulse,chest pain,confusion,dizziness,low blood pressure,lightheadedness,heart palpitations,shortness of breath,sudden weakness,fainting,a loss of consciousness and cardiac arrest, in some cases<br><br>Prevention<br>Eating a heart-healthy diet,Staying physically active and keeping a healthy weight,Avoiding smoking,Limiting or avoiding caffeine and alcohol,Reducing stress, as intense stress and anger can cause heart rhythm problems and Using over-the-counter medications with caution, as some cold and cough medications contain stimulants that may trigger a rapid heartbeat<br><br>Treatment<br>treatment options for tachycardia will depend on various factors like the cause,the age of the person,their overall health like Vagal maneuvers,Medication:amiodarone (Cordarone), sotalol (Betapace), and mexiletine (Mexitil),calcium channel blockers, such as diltiazem (Cardizem) or verapamil (Calan),beta-blockers, such as propranolol (Inderal) or metoprolol (Lopressor)and blood thinners, such as warfarin (Coumadin) or apixaban (Eliquis),Cardioversion and defibrillators(electric shockTrusted Source)Radiofrequency catheter ablation and Surgery"
        if heartbeat<60:
            output+="<br><br>Below Normal Range:Chia seeds, flax seeds, and hemp seeds, greens vegetables, whole grains, berries, avocados, fatty fish and fish oil, walnuts, beans, dark chocolate, tomatoes, almonds, garlic, olive oil.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(heartbeat+fHB)+"<br><br>Disease Related to low Pulse<br><br>Bradycardia<br><br>Symptoms:<br>Fatigue or feeling weak,Dizziness or lightheadedness,Confusion,Fainting (or near-fainting) spells,Shortness of breath,Difficulty when exercising and Cardiac arrest (in extreme cases)<br><br>Diseases:<br>In some cases, slow heartbeat may be a symptom of a serious or life-threatening condition that should be immediately evaluated in an emergency setting. These conditions include:Cardiogenic shock (shock caused by heart damage and ineffective heart function),Congestive heart failure (deterioration of the heart’s ability to pump blood),Dissecting aortic aneurysm (life-threatening bulging and weakening of the aortic artery wall that can burst and cause severe hemorrhage),Myocardial infarction (heart attack),Myocarditis (infection of the middle layer of the heart wall),Overdose, including cumulative overdose, of certain cardiac medications,Pericarditis (infection of the lining that surrounds the heart) and Trauma.<br><br>Treatment:<br>Borderline or occasional bradycardia may not require treatment.,Severe or prolonged bradycardia can be treated in a few ways. For instance, if medication side effects are causing the slow heart rate, then the medication regimen can be adjusted or discontinued andIn many cases, a pacemaker can regulate the heart’s rhythm, speeding up the heart rate as needed."
    if a==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(SYSTOLIC)</p>"
        if systolic>120:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(int(int(systolic)-fS))+" to "+str(systolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if systolic<90:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(systolic)+" to "+str(int(int(systolic)+fS))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if b==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(DIASTOLIC</p>"
        if diastolic>80:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(int(int(diastolic)-fD))+" to "+str(diastolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if diastolic<60:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(diastolic)+" to "+str(int(int(diastolic)+fD))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if bloodoxygenresult==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD OXYGEN</p>"
        if bloodoxygen<90:
            output+="<br><br>Below Normal Range:Cayenna pepper, beets, berries, fatty fish, pomegranates, garlic, walnuts, grapes, turmeric, spinach, citrus fruit , chocolate, ginger.<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br><br>Disease Related to low N=bLood Oxygen<br><br>Symptoms:<br>apid breathing,shortness of breath,fast heart rate,coughing or wheezing,sweating,confusion and changes in the color of your skin<br><br>Treatment:<br>Medication-inhaler,oxygen gas,liquid oxygen,oxygen concentrators and hyperbaric oxygen therapy<br><br>Tips/Prevention:<br>Stop smoking, and avoid secondhand smoke or environmental irritants,Eat foods rich in antioxidants,Get vaccinations like the flu vaccine and the pneumonia vaccine. This can help prevent lung infections and promote lung health,Exercise more frequently, which can help your lungs function properly and Improve indoor air quality. Use tools like indoor air filters and reduce pollutants like artificial fragrances, mold, and dust."
        if bloodoxygen >100:
            output+="<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(round((float(bloodoxygen)-fBO),2))+" to "+str(bloodoxygen)+"Disease Related to High Blood Oxygen<br><br>Oxygen Toxicity<br><br>Symptoms<br>Coughing,Mild throat irritation,Chest pain,Trouble breathing,Muscle twitching in face and hands,Dizziness,Blurred vision,Nausea,A feeling of unease,Confusion and Convulsions (seizure)<br><br>Treatment<br>Your lungs may take weeks or more to recover fully on their own. If you have a collapsed lung, you may need to use a ventilator for a while. Your healthcare provider will tell you more about any other kinds of treatment."

    if p==1.0:
        output="<p style='color:red;background-color:white;'>ABNORMAL TEMPERATURE</p>"
        if temperature>99:
            output+="<br><br>Above Normal Range: Chicken soup, garlic, coconut water, hot tea, honey, ginger, spicy foods, bananas, oatmeal, yogurt, fruits like strawberries, cranberries, blueberries, blackberries, avocados, greeny vegetables, salmon.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<br><br>Hyperpyrexia<br><br>Symptoms<br>increased thirst,extreme sweating,dizziness,muscle cramps,fatigue and weakness,nausea and light-headedness<br><br>Treatement:<br>a cool bath or cold, wet sponges put on the skin,liquid hydration through IV or from drinking and fever-reducing medications, such as dantrolene"
        if temperature<97.7:
            output+="<br><br>Below Normal Range:Hot tea or coffee, soup, roasted veggies, protein and fats like nuts, avocados, seeds ,olives, salmon, hard-boiled eggs, iron like shellfish, red meat, beans, broccoli.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(round((float(temperature)-fT),2))+" to "+str(temperature)+"<br><br>Disease Related to Low temperature<br><br>Hypothermia :<br><br>Symptoms:<br>shivering,slow, shallow breath,slurred or mumbled speech,a weak pulse,poor coordination or clumsiness,low energy or sleepiness,confusion or memory loss and loss of consciousness<br><br>complications:<br>frostbite, or tissue death, which is the most common complication that occurs when body tissues freeze,chilblains, or nerve and blood vessel damage,gangrene, or tissue destruction,trench foot, which is nerve and blood vessel destruction from water immersion and Hypothermia can also cause death.<br><br>Medications:<br> antidepressants, sedatives, and antipsychotic ,warm fluids, often saline, injected into the veins, Airway rewarming.<br><br>Tips/prevention:<br>Handle the person with care,Remove the person’s wet clothing,Apply warm compresses and Monitor the person’s breathing."

    output=Markup(output)

    return render_template('AI FORECASTING.html', prediction_text=output) 
 
@app.route('/predict',methods=['POST'])
def predict():
    output=home1()
    df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20dataset.csv")
    df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
    df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
    df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
    int_features = [x for x in request.form.values()]
    index1=df.DATE[df.DATE == int_features[0]].index.tolist()
    index=df.DATE[df.DATE == int_features[1]].index.tolist()
    print(int_features,index,index1)
    I1,I2=index1[0],index[len(index)-1]
    if(int_features[2]=='HEART BEAT'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['HEART BEAT'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("Heart beat")
        plt.legend(['Heart Beat'])
        plt.title("HEART BEAT")
    elif(int_features[2]=='BLOOD OXYGEN'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['OXYGEN LEVEL'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("Oxygen level")
        plt.legend(['Oxygen level'])
        plt.title("BLOOD OXYGEN")
    elif(int_features[2]=='BLOOD PRESSURE'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['SYSTOLIC'][I1:I2])
        xpoints1 = np.array(df['DATE'][I1:I2])
        ypoints1 = np.array(df['DIASTOLIC'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.plot(xpoints1, ypoints1)
        plt.legend(['systolic','diastolic'])
        plt.xlabel("Date")
        plt.ylabel("Blood Pressure")
        plt.title("BLOOD PRESSURE")
    elif (int_features[2]=='TEMPERATURE'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['TEMPERATURE'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("temperature")
        plt.legend(['Temperature'])
        plt.title("TEMPERATURE")
    elif(int_features[2]=='ALL'):
        figure, axis = plt.subplots(2,2)
        figure.set_figwidth(10)
        figure.set_figheight(12)
        I1,I2=index1[0],index[len(index)-1]
        xpoints = np.array(df['DATE'][I1:I2])       
        ypoints = np.array(df['HEART BEAT'][I1:I2])        
        xpoints1 = np.array(df['DATE'][I1:I2])
        ypoints1= np.array(df['OXYGEN LEVEL'][I1:I2])
        xpoints2 = np.array(df['DATE'][I1:I2])
        ypoints2 = np.array(df['SYSTOLIC'][I1:I2])       
        xpoints3 = np.array(df['DATE'][I1:I2])
        ypoints3 = np.array(df['DIASTOLIC'][I1:I2])       
        xpoints4 = np.array(df['DATE'][I1:I2])
        ypoints4 = np.array(df['TEMPERATURE'][I1:I2])
        axis[0,0].plot(xpoints, ypoints)        
        axis[0,0].legend(['Heart Beat'])
        axis[0,0].set_xlabel("Date")
        axis[0,0].set_ylabel("Heart Beat")
        axis[0,0].set_title("HEART BEAT")
        axis[0,1].plot(xpoints1, ypoints1)
        axis[0,1].set_xlabel("Date")
        axis[0,1].set_ylabel("Oxygen level")
        axis[0,1].legend(['Oxygen level'])
        axis[0,1].set_title("OXYGEN LEVEL")
        axis[1,0].plot(xpoints2, ypoints2)        
        axis[1,0].plot(xpoints3, ypoints3)
        axis[1,0].set_xlabel("Date")
        axis[1,0].set_ylabel("Blood Pressure")
        axis[1,0].legend(['systolic','diastolic'])
        axis[1,0].set_title("BLOOD PRESSURE")
        axis[1,1].plot(xpoints4, ypoints4)
        axis[1,1].set_xlabel("Date")
        axis[1,1].set_ylabel("Temperature")
        axis[1,1].legend(['Temperature']) 
        axis[1,1].set_title("TEMPERATURE")
    plt_html = mpld3.fig_to_html(figure)

    return '''<!DOCTYPE html>
<html >
<head>
   <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="html2pdf.bundle.min.js"></script>
    <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.min.js"></script> 
  <script src="https://unpkg.com/aos@2.3.0/dist/aos.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/aos@2.3.0/dist/aos.css">
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/fullPage.js/3.0.4/vendors/easings.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/fullPage.js/3.0.4/vendors/scrolloverflow.min.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/fullPage.js/3.0.4/fullpage.js"></script>
  <meta name="viewport" content="width=device-width,height=device-height,initial-scale=1.0"/>
  

  <title>AI Predictions</title>
  
</head>
    <style>
    /*BODY Style*/
  @media screen and (min-width: 700px) {
  body {
    font-size: 15px;
  }
}

@media screen and (max-width: 700px) {
  body {
    font-size: 12px;
  }
  #contact
  {
    font-size: 10px;
  }
}
  body
  {
    background-color: rgba(240,240,240);
    display:inherit;
    overflow-x: hidden;
    margin: 0px;
  }
  /*LOGO AND NAVBAR STYLE*/
 .navbar,.navbar1,.desc
  {
    overflow: hidden;
  }
  .navbar a 
  {
    float: left;
    color: rgb(30,144,255);
    text-align: center;
    padding: 14px 16px;
    text-decoration: none;
  }
  .cropped1
  {
    padding-left: 2%;
    vertical-align: middle;
    font-size: 150%;
  }
  .navbar
  {
    background-image: url('https://image.freepik.com/free-photo/shades-blue-white-background_23-2147746645.jpg');
    background-color: rgba(0,0,0, 0.3);
    background-blend-mode: darken;
    color: white;
    padding-left: 2%;
    resize: both;
    overflow: auto;
  }
  .gify:hover 
  {
    color: blue;
    box-shadow: 0 0 40px white;
    text-shadow: 0 0 40px white;
  }
  div.scrollmenu 
  {
    padding-top: 1%;
    overflow: auto;
    white-space: nowrap;
  }
  div.scrollmenu a 
  {
    display: inline-block;
    color: rgb(30,144,255);
    text-align: center;
    padding: 14px;
    text-decoration: none;
  }
  div.scrollmenu a:hover 
  {
  background-color: rgb(30,144,255);
  color: white;
  }
  .scrollmenu 
  {
    background-color: #eee;
    width: 100%;
    height: 100%;
    overflow-y: scroll; 
  }
  .mySlides
  {
    display: none;
  }
  
  /*onloader style*/
  .container 
  {
	  display: flex;
	  justify-content: center;
	  align-items: center;
	  height: 100vh;
	  overflow: hidden;
  }
  .item 
  {
	  width: 20px;
	  height: 20px;
	  margin: 10px;
	  list-style-type: none;
	  transition: 0.5s all ease;
  }
  .item:nth-child(1) {
	  animation: right-1 1s infinite alternate;
	  background-color: #49b8e5;
	  animation-delay: 100ms;
  }
  @keyframes right-1 {
	  0% {
	  	 transform: translateY(-60px);
	  }
	  100% {
		  transform: translateY(60px);
	  }
  }
  .item:nth-child(2) {
	  animation: right-2 1s infinite alternate;
	  background-color: #1e98d4;
	  animation-delay: 200ms;
  }
  @keyframes right-2 {
	  0% {
	  	 transform: translateY(-70px);
	  }
	  100% {
		  transform: translateY(70px);
	  }
  }
  .item:nth-child(3) {
	  animation: right-3 1s infinite alternate;
	  background-color: #2a92d0;
	  animation-delay: 300ms;
  }
  @keyframes right-3 {
	  0% {
	  	 transform: translateY(-80px);
	  }
	  100% {
		  transform: translateY(80px);
	  }
  }
  .item:nth-child(4) {
	  animation: right-4 1s infinite alternate;
	  background-color: #3a88c8;
	  animation-delay: 400ms;
  }
  @keyframes right-4 {
	  0% {
	  	 transform: translateY(-90px);
	  }
	  100% {
		  transform: translateY(90px);
	  }
  }
  .item:nth-child(5) {
	  animation: right-5 1s infinite alternate;
	  background-color: #507cbe;
	  animation-delay: 500ms;
  }
  @keyframes right-5 {
	 0% {
	  	 transform: translateY(-100px);
	  }
	  100% {
		  transform: translateY(100px);
	  }
  }
    </style>
  
<script>
var slideIndex = 0;
showDivs(slideIndex);

function plusDivs(n) {
  showDivs(slideIndex += n);
}

function showDivs(n) {
  var i;
    
  var x = document.getElementsByClassName("mySlides");
  var y = document.getElementsByClassName("mySlides1");
  if (n > x.length) {slideIndex = 1}
  if (n < 1) {slideIndex = x.length}
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";  
  }
  for (i = 0; i < y.length; i++) {
    y[i].style.display = "none";  
  }    
  x[slideIndex-1].style.display = "block";  
}
</script>
     <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.js"></script>
    <script>
        
        window.onload = function () {
    document.getElementById("create_pdf")
        .addEventListener("click", () => {
            const invoice = this.document.getElementById("report");
            console.log(invoice);
            console.log(window);
            var opt = {
                margin: 1,
                filename: 'myreport.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2 },
                jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            html2pdf().from(invoice).set(opt).save();
        })
}
 
  /* onloader function*/
  $(window).on('load', function () {  
           $(".container").fadeOut("slow");  
      }); 
  /*animation function*/    
      $(function() {
          AOS.init();
     });
</script>

<body>
  <!--LOADER CONTENT-->
  <div class="container">
    <div class="item"></div>
    <div class="item"></div>
    <div class="item"></div>
    <div class="item"></div>
    <div class="item"></div>
  </div>
<div class='navbar' id='title'>
    <p style="font-size: 140%"><img class="cropped1" src="https://drive.google.com/thumbnail?id=1JQ6epr36ugrVF7cuTlYMw9kL7J-pZrfd" width=5%; height="5%;">&nbsp HEALTH COMPANION</p></div>
    <div class="scrollmenu">
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/" >HOME</a>
             <a href='https://kruphacm.github.io/health-improviser-and-monitoring-system/Consulting%20doctors.html' >CONSULTING</a>
             <a href='https://aiforecasting.herokuapp.com/' >CHECK MY HEALTH</a>
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/AI%20predictions.html" >AI PREDICTIONS</a>
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/Hospital.html">HOSPITAL CONSULTING</a>      
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/HealthyTips.html">HEALTH TIPS</a>
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/food1-1.html">FOOD INFO</a>
             <a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/Disease%20information.html">DISEASE INFO</a>
             </div>
  <br>
  <br>
  <br>

    <div class="w3-content w3-section" style='background-image: url(https://img.freepik.com/free-photo/abstract-grunge-decorative-relief-navy-blue-stucco-wall-texture-wide-angle-rough-colored-background_1258-28311.jpg?size=626&ext=jpg);background-repeat: no-repeat;background-size: cover;background-color: rgba(0,0,0, 0.6);
    background-blend-mode: darken;color: white;'>
        <br><br>
        <p style="text-align: center;font-family: cursive;"><b>CLICK RIGHT ARROW TO SEE THE STEPS</b></p>
         <!-- Navigation arrows -->  
        <a class="left" onclick="plusDivs(-1)" style="padding-left: 5%;  padding-top: -15; font-size: 150%;"><b>❮</b></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        <a class="right" onclick="plusDivs(1)" style="padding-right: 5%; float: right; padding-top: -15; font-size: 150%;">❯</a>
        <div ><img style="padding-left:15%;"class="mySlides1"  src="https://www.log-hub.com/wp-content/uploads/2017/12/forecating_process.png" width='70%;' ></div>
         
       <div class='id1'><img class="mySlides" style="padding-left:15%;"src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST1.png"  width='70%' ></div>
       <div class='id2'><img class="mySlides"  style="padding-left:15%;"src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST2.png" width='70%'></div>
      <div class='id3'> <img class="mySlides"  style="padding-left:15%;"src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST3.png"width='70%'></div>     
   
      <br><br><br><br>
</div><br><br><br><br>
    <div class='form' style='color: black; background-image:url(https://techchannel.com/getattachment/1c269c15-8656-4e8a-909a-71358ccc76d3/techu-talks.jpg);background-repeat: no-repeat;background-size: cover;background-color: rgba(0,0,0, 0.4);
    background-blend-mode: darken;padding: 10%; '>
        <h2 STYLE="text-align: center;color: white;">REPORT</h2>
         <br><br>

        <div style='padding-left:5%; padding-right: 20%; font-size: 150%;background-color: white;' id="report">
            
     '''+output+'''
         <br>
         <br></div>
      <br><br>
      <br><br>
    <div  style="padding-left: 45%">
    <br><br>
        <input type="button" id="create_pdf" value="GENERATE PDF" > <br><br><br>  <br><br></div> </div>
    <br><br><br>
    <div style="background-image:url(https://www.wallpaperbetter.com/wallpaper/291/641/316/blue-shades-2K-wallpaper.jpg);background-repeat: no-repeat;background-size: cover;background-color: rgba(0,0,0, 0.4);
    background-blend-mode: darken;  padding: 10%; color:white;">
      <form action="{{ url_for('predict')}}"method="post" style='color: white;  text-align:center; font-size: 150%;'>
          <h3>GRAPH REPRESENTATION</h3>
          <p >the Date Startes from 01-01-2021 to 30-04-2021</p>
          <p style="color: red;">enter the starting date and ending date within 20 days limit.</p><br><br>
        <p>Enter Starting Date:<input type="text" name="Enter the Starting Date" placeholder="Enter the Starting Date" required="required" /></p><br><br>
        <p>Enter the Ending Date:<input type="text" name="Enter the Ending Date" placeholder="Enter the Ending Date" required="required" /></p><br><br>
          <label for="cars">Choose a parameter:</label>
          <select id="cars" name="cars">
  <option value="HEART BEAT">HEART BEAT</option>
  <option value="BLOOD OXYGEN">BLOOD OXYGEN</option>
  <option value="BLOOD PRESSURE">BLOOD PRESSURE</option>
  <option value="TEMPERATURE">TEMPERATURE</option>
  <option value="ALL">ALL THE ABOVE</option>
</select><br><br><br><br><br>
        <button type="submit" class="btn btn-primary btn-block btn-large" style='background-color:lightsalmon; color: white; font-size: 100%; padding: 1%;'>&nbsp&nbsp &nbsp SHOW GRAPH&nbsp&nbsp &nbsp  </button><br><br>
    </form>
    </div>
    <br><br>
    <br><br>
    <div style='color: white; background-image: url(https://images.unsplash.com/photo-1568301956237-25a54f5f0d21?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8N3x8fGVufDB8fHx8&w=1000&q=80); text-align: center;background-repeat: no-repeat;background-size: cover;background-color: rgba(0,0,0, 0.4);
    background-blend-mode: darken; font-size: 150%;'>
         <br><br>
     <p>GRAPH</p>
 <div style="padding-left: 10%;padding-right: 10%; ">
        <div><div style="padding-left: 7%; background: white;">'''+plt_html+'''</div></div><br><br>
     <div style="background-color: white;">
     <p style="color:red; ">NOTE: this can be used only once so click the button below  to view the graph again</p></div></div><br><br>
        <a href="https://aiforecasting.herokuapp.com/" style="color:white; background-color:lightsalmon;padding: 2%;text-decoration: none;">CLICK HERE</a><br><br><br>
         <br>
         <br>
     </div>
    <br><br>
    <!--CONTACT AND LINKS SECTION-->
  <div  id="contact" style="background-image: url('https://image.freepik.com/free-photo/shades-blue-white-background_23-2147746645.jpg');background-color: rgba(0,0,0, 0.59);background-blend-mode: darken;color: white;font-size:auto;"><br>
    <table style="width: auto;padding-left: 5%;vertical-align: top;width:100%;">
      <tr>  
        <td >
          <p >Contact<br>Mobile:1234567891<br>Email:abcd123@gmail.com</p>
          <p >Follow us on</p><img src="https://www.beachrealtync.com/sites/default/files/uploads/socials_3.png" width=100px  ><br><br>
        </td>
        <td>
          <p style="margin:0%;">USEFUL LINKS</p>
          <hr style="color: white;width: 50%;float: left;margin:0%;">
          <ul style="text-decoration:none;text-decoration-style: none;">
            <li><a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/HealthyTips.html" style="text-decoration:none;color: white;">diet plan</a></li>
            <li><a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/food1-1.html" style="text-decoration:none;color: white;">food tips</a></li>
            <li><a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/Disease%20information.html" style="text-decoration:none;color: white;">know about disease</a></li>
            <li><a href="https://kruphacm.github.io/health-improviser-and-monitoring-system/Hospital.html" style="text-decoration:none;color: white;">famous hospital</a></li>
          </ul>  
        </td>
        <td>
          <p style="margin:0%;">OFFICIAL WEBSITES</p>
          <hr style="color: white;width: 50%;float: left;margin:0%;">
          <ul style="text-decoration:none;text-decoration-style: none;">
            <li><a href="https://medlineplus.gov/encyclopedia.html" style="text-decoration:none;color: white;">DISEASE ENCYCLOPEDIA</a></li>
            <li><a href="https://www.nal.usda.gov/legacy/fnic/food-dictionaries-and-encyclopedias" style="text-decoration:none;color: white;">FOOD ENCYCLOPEDIA</a></li>
          </ul> 
          <br><br> 
        </td>
      </tr>      
      </table>
      <br>
  </div>
  <!--OWNERS RIGHT-->
  <div style="padding:0.2%;background-color:white;font-size: 100%; ">
    <p style="text-align: center;color: #507cbe">managed by admins of this website</p>
  </div>
    </body>
</html>'''
    
def home1():
    df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20dataset.csv")
    df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
    df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
    df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
    #combining all the  results
    orgdate=list(str(df['DATE'].tail(1)).strip().split(" "))
    date=orgdate[4][:orgdate[4].rfind("\n")]
    orgtime=list(str(df['TIME'].tail(1)).strip().split(" "))
    time=orgtime[4][:orgtime[4].rfind("\n")]
    #tempratue
    orgtemp=list(str(df['TEMPERATURE'].tail(1)).strip().split(" "))
    temperature=orgtemp[4][:orgtemp[4].rfind("\n")]
    temperatureresult=model5.predict([[temperature,temperature]])
    #systolic
    orgsys=list(str(df['SYSTOLIC'].tail(1)).strip().split(" "))
    systolic=orgsys[4][:orgsys[4].rfind("\n")]
    print(systolic)
    a=model3.predict([[systolic,systolic]])
    #diastolic
    orgdia=list(str(df['DIASTOLIC'].tail(1)).strip().split(" "))
    diastolic=orgdia[4][:orgdia[4].rfind("\n")] 
    b=model4.predict([[diastolic,diastolic]])
    #blood oxygen
    orgbo=list(str(df['OXYGEN LEVEL'].tail(1)).strip().split(" "))
    bloodoxygen=orgbo[4][:orgbo[4].rfind("\n")]
    bloodoxygenresult=model2.predict([[bloodoxygen,bloodoxygen]])
    #heart beat
    orghb=list(str(df['HEART BEAT'].tail(1)).strip().split(" "))
    heartbeat=orghb[4][:orghb[4].rfind("\n")]
    p=model1.predict([[heartbeat,heartbeat]])
    #condition for printing 
    print(p,a,b,bloodoxygenresult,temperatureresult)
    output ="<br><br><br><p style='padding left:5%;'><center>REPORT</center></p><br><br>Date: "+str(date)+"<br><br>Time:"+str(time)+"<br><br><br>Readings<br><br>HEARTBEAT: "+str(heartbeat)+"<br><br>BLOOD PRESSURE(SYSTOLIC): "+str(systolic)+"<br><br>BLOOD PRESSURE(DIASTOLIC): "+str(diastolic)+"<br><br>BLOOD OXYGEN: "+str(bloodoxygen)+"<br><br>TEMPRATURE: "+str(temperature)+"<br><br>NORMAL RANGE<br><br>HEARTBEAT: 60 to 100 bpm<br><br>BLOOD PRESSURE(SYSTOLIC): 90 to 120<br><br>BLOOD PRESSURE(DIASTOLIC): 60 to 80<br><br>BLOOD OXYGEN: 95.0% to 99.9%<br><br>TEMPRATURE: 97.7 F to 99.0 F<br><br>RESULT<br>"
    fHB,fS,fD,fBO,fT=(float(heartbeat)/100)*10,(float(systolic)/120)*10,(float(diastolic)/80)*5,(float(bloodoxygen)/100)*5,(float(temperature)/97)*5
    if temperatureresult==0.0 and a==0.0 and b==0.0 and bloodoxygenresult==0.0 and p==0.0:
        
        output+="<p style='color:green;background-color:white;'>NORMAL</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR NORMAL RANGE<br><br>Normal HeartBeat:Banana, melons, orange ,sweet potatoes, dairy, whole grains, chicken(8-ounce glass of water)<br><br>Normal BP:egg, chicken, nuts and seeds, fruits and vegetables.<br><br>Normal Temprature:Hot water, water rich foods like cucumber and water melon, green leafy vegetables like spinach, kale, broccoli.<br><br>Normal Blood Oxygen:Beetroot, garlic, leafy greens, pomegranate, cruciferous vegetables, sprouts, meat, nuts, seeds, dates, carrots, banana.<br><br>Medication: Follow Your Regular Medication.<br><br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(int(int(heartbeat)+fHB))+"<br>predicted systolic: "+str(systolic)+" to "+str(int(float(systolic)+fS))+"<br>predicted diastolic: "+str(diastolic)+" to "+str(int(float(diastolic)+fD))+"<br>predicted Blood Oxygen: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br>predicted Temprature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<p style='font-size:100%;color:red;'>These predictions will be apllicable when the above diet followed.</p>"
    if temperatureresult==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL HEARTBEAT</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR ABNORMAL RANGE"
        if heartbeat>100:
            output+="<br><br>Above Normal Range:Omega-3 fatty acids, found in fish, lean meats, nuts, grains and legume. Phenols and tannins found in tea, coffee. and red wine(in moderation).Vitamin A, found in greens. Whole grains. Vitamin C in bean sprouts.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(int(int(heartbeat)-fHB))+" to "+str(heartbeat)+"<br><br>Disease Related to High Pulse<br><br>Tachycardia:<br><br>disease: Stroke, Heart failure, Sudden death, Blood clots<br><br>Symptoms:<br>a fast pulse,chest pain,confusion,dizziness,low blood pressure,lightheadedness,heart palpitations,shortness of breath,sudden weakness,fainting,a loss of consciousness and cardiac arrest, in some cases<br><br>Prevention<br>Eating a heart-healthy diet,Staying physically active and keeping a healthy weight,Avoiding smoking,Limiting or avoiding caffeine and alcohol,Reducing stress, as intense stress and anger can cause heart rhythm problems and Using over-the-counter medications with caution, as some cold and cough medications contain stimulants that may trigger a rapid heartbeat<br><br>Treatment<br>treatment options for tachycardia will depend on various factors like the cause,the age of the person,their overall health like Vagal maneuvers,Medication:amiodarone (Cordarone), sotalol (Betapace), and mexiletine (Mexitil),calcium channel blockers, such as diltiazem (Cardizem) or verapamil (Calan),beta-blockers, such as propranolol (Inderal) or metoprolol (Lopressor)and blood thinners, such as warfarin (Coumadin) or apixaban (Eliquis),Cardioversion and defibrillators(electric shockTrusted Source)Radiofrequency catheter ablation and Surgery"
        if heartbeat<60:
            output+="<br><br>Below Normal Range:Chia seeds, flax seeds, and hemp seeds, greens vegetables, whole grains, berries, avocados, fatty fish and fish oil, walnuts, beans, dark chocolate, tomatoes, almonds, garlic, olive oil.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(heartbeat+fHB)+"<br><br>Disease Related to low Pulse<br><br>Bradycardia<br><br>Symptoms:<br>Fatigue or feeling weak,Dizziness or lightheadedness,Confusion,Fainting (or near-fainting) spells,Shortness of breath,Difficulty when exercising and Cardiac arrest (in extreme cases)<br><br>Diseases:<br>In some cases, slow heartbeat may be a symptom of a serious or life-threatening condition that should be immediately evaluated in an emergency setting. These conditions include:Cardiogenic shock (shock caused by heart damage and ineffective heart function),Congestive heart failure (deterioration of the heart’s ability to pump blood),Dissecting aortic aneurysm (life-threatening bulging and weakening of the aortic artery wall that can burst and cause severe hemorrhage),Myocardial infarction (heart attack),Myocarditis (infection of the middle layer of the heart wall),Overdose, including cumulative overdose, of certain cardiac medications,Pericarditis (infection of the lining that surrounds the heart) and Trauma.<br><br>Treatment:<br>Borderline or occasional bradycardia may not require treatment.,Severe or prolonged bradycardia can be treated in a few ways. For instance, if medication side effects are causing the slow heart rate, then the medication regimen can be adjusted or discontinued andIn many cases, a pacemaker can regulate the heart’s rhythm, speeding up the heart rate as needed."
    if a==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(SYSTOLIC)</p>"
        if systolic>120:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(int(int(systolic)-fS))+" to "+str(systolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if systolic<90:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(systolic)+" to "+str(int(int(systolic)+fS))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if b==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(DIASTOLIC</p>"
        if diastolic>80:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(int(int(diastolic)-fD))+" to "+str(diastolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if diastolic<60:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(diastolic)+" to "+str(int(int(diastolic)+fD))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if bloodoxygenresult==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD OXYGEN</p>"
        if bloodoxygen<90:
            output+="<br><br>Below Normal Range:Cayenna pepper, beets, berries, fatty fish, pomegranates, garlic, walnuts, grapes, turmeric, spinach, citrus fruit , chocolate, ginger.<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br><br>Disease Related to low N=bLood Oxygen<br><br>Symptoms:<br>apid breathing,shortness of breath,fast heart rate,coughing or wheezing,sweating,confusion and changes in the color of your skin<br><br>Treatment:<br>Medication-inhaler,oxygen gas,liquid oxygen,oxygen concentrators and hyperbaric oxygen therapy<br><br>Tips/Prevention:<br>Stop smoking, and avoid secondhand smoke or environmental irritants,Eat foods rich in antioxidants,Get vaccinations like the flu vaccine and the pneumonia vaccine. This can help prevent lung infections and promote lung health,Exercise more frequently, which can help your lungs function properly and Improve indoor air quality. Use tools like indoor air filters and reduce pollutants like artificial fragrances, mold, and dust."
        if bloodoxygen >100:
            output+="<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(round((float(bloodoxygen)-fBO),2))+" to "+str(bloodoxygen)+"Disease Related to High Blood Oxygen<br><br>Oxygen Toxicity<br><br>Symptoms<br>Coughing,Mild throat irritation,Chest pain,Trouble breathing,Muscle twitching in face and hands,Dizziness,Blurred vision,Nausea,A feeling of unease,Confusion and Convulsions (seizure)<br><br>Treatment<br>Your lungs may take weeks or more to recover fully on their own. If you have a collapsed lung, you may need to use a ventilator for a while. Your healthcare provider will tell you more about any other kinds of treatment."

    if p==1.0:
        output="<p style='color:red;background-color:white;'>ABNORMAL TEMPERATURE</p>"
        if temperature>99:
            output+="<br><br>Above Normal Range: Chicken soup, garlic, coconut water, hot tea, honey, ginger, spicy foods, bananas, oatmeal, yogurt, fruits like strawberries, cranberries, blueberries, blackberries, avocados, greeny vegetables, salmon.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<br><br>Hyperpyrexia<br><br>Symptoms<br>increased thirst,extreme sweating,dizziness,muscle cramps,fatigue and weakness,nausea and light-headedness<br><br>Treatement:<br>a cool bath or cold, wet sponges put on the skin,liquid hydration through IV or from drinking and fever-reducing medications, such as dantrolene"
        if temperature<97.7:
            output+="<br><br>Below Normal Range:Hot tea or coffee, soup, roasted veggies, protein and fats like nuts, avocados, seeds ,olives, salmon, hard-boiled eggs, iron like shellfish, red meat, beans, broccoli.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(round((float(temperature)-fT),2))+" to "+str(temperature)+"<br><br>Disease Related to Low temperature<br><br>Hypothermia :<br><br>Symptoms:<br>shivering,slow, shallow breath,slurred or mumbled speech,a weak pulse,poor coordination or clumsiness,low energy or sleepiness,confusion or memory loss and loss of consciousness<br><br>complications:<br>frostbite, or tissue death, which is the most common complication that occurs when body tissues freeze,chilblains, or nerve and blood vessel damage,gangrene, or tissue destruction,trench foot, which is nerve and blood vessel destruction from water immersion and Hypothermia can also cause death.<br><br>Medications:<br> antidepressants, sedatives, and antipsychotic ,warm fluids, often saline, injected into the veins, Airway rewarming.<br><br>Tips/prevention:<br>Handle the person with care,Remove the person’s wet clothing,Apply warm compresses and Monitor the person’s breathing."


    return output 


if __name__ == "__main__":
    app.run(debug=True)