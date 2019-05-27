# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:55:01 2018

@author: RAHUL
"""
import pandas as pd
from collections import Counter
#file = open("nlp.txt","r")
# 
#lines = file.readlines()[4:] 
#print lines
#print re.match(r'Incident',"".join(lines)).group()
#l=[]
#prev=""
#for line in lines:
#    
#    if line.find('Reported:') != -1:
#        l.append( prev.strip())
#    prev = line

#print len(set(l))    
#print Counter(l).keys()

#df = pd.DataFrame(Counter(l).keys())
#df.to_csv("nlp_keys.csv",index=False)
    
#print Counter(l).values()

L_no = ['OFFICER STATUS Special Detail','SERVICE Court Issued Restraining Order','SERVICE Missing/Found Persons', 'PATROL FIELD INTERVIEW',
        'CHILD NEGLECT Child Abuse', 'SERVICE Commitment To Mental Health For 72 Hrs','ADMINISTRATIVE Report Voided']

L0 = ['PROPERTY Recovered/Impounded Vehicle','PROPERTY Recovered/Impounded Bicycle','PROPERTY Recovered Property Without a Crime','THEFT-GRAND Receiving Stolen Property',
      'HEALTH & SAFETY Possession of Marijuana'
]

L1=[ 'FIRE Alarm Accidental',
    'EH&S Smell of Smoke, Odor, Non-Fire',
    'EH&S Violation of Campus Smoking Policy',
    'TRAFFIC Possession of Stolen Parking Permit',
    'PROPERTY Damaged Property',
    'TRAFFIC Parking Problem',
    'LA MUNICIPAL CODE Loud and Raucous Noise',
    'SERVICE Maintenance Request (Non-Lighting)',
     'DISORDERLY CONDUCT Prowling, Loitering',
     'DISORDERLY CONDUCT Ejection From University Event',
     'ALARM RESPONSE Door Held Open',
     'EH&S Hazardous Materials',
     'DISTURBANCE Disturbing The Peace',
     'DISORDERLY CONDUCT Violation of Campus Bicycle Policies',
     'FIRE Alarm Activation',
     'TRAFFIC Altered Parking Permit',
     'LA MUNICIPAL CODE Illegal Ticket Sale - Scalping',
     'EH&S Slip, Trip & Fall Safety Hazard',
     'INCIDENT Business Dispute',
     'TRAFFIC Parking or Moving Violation',
     'PROPERTY Lost/Missing Property'
     
     
     
    ]
    
L2 = [ 
    'THEFT-PETTY Theft Petty-Plain-Attempt',
    'SERVICE Illness Response', 
    'HARASSMENT Intimidation',
    'FIRE Alarm Smoke',
    'WARRANT Warrant Arrest',
    'ALCOHOL Drinking In Public',
    'VEHICLE CODE Possession Fake Identification',
    'WARRANT Probation/Parole Violation',
    'HARASSMENT Stalking',
    'THEFT-PETTY Theft Petty-Plain',
    'SERVICE Medical Escort',
    'DISORDERLY CONDUCT Interfering/Resisting Arrest',
    'THEFT-TRICK Theft-Trick or Device',
    'SERVICE Emergency Notification',
    'VEHICLE CODE Traffic Device Related Offense',
    'VEHICLE CODE Malicious Mischief to Vehicle',
    'HARASSMENT Harassment',
    'HARASSMENT Restraining Order Violation',
    'THEFT-PETTY Shoplifting',
    'CHILD NEGLECT Juvenile Truancy/Curfew Violation',
    'FRAUD Fraud-General',
    'CRIMINAL THREATS Criminal Threats',
    'INCIDENT Domestic Dispute',
    'BURGLARY Burglary-Hot Prowl',
    'INCIDENT Dispute',
    'HATE INCIDENT Hate/Bias Incident',
    'THEFT-PETTY Theft Bicycle',
    'ROBBERY Robbery-Simulated Weapon',
    'TRAFFIC Traffic Collision Without Injuries',
    'TRESPASS Refusing to Leave Campus Facility',
    'FIRE Fire-Trash',
    'CHILD NEGLECT Contributing to Delinquincy',
    'THEFT-GRAND Theft Grand-Trick or Device'
    'ALCOHOL Drunk In Public',
    'TRAFFIC Traffic Collision-Pedestrian',
    'THEFT-ACCESS Unauthorized Computer Access',
'FIRE Alarm Malfunction',    
'SERVICE Animal Control Problem',
'SERVICE Injury Response',
'THEFT-MOTOR VEHICLE Attempt Theft From Motor Vehicle',
'CRIMINAL THREATS Threatening School Officials',
    'VANDALISM Vandalism-Misdemeanor',
    'OBSCENE ACTIVITY Obscene, Annoying, Threatening Phone Calls',
    'ROBBERY Robbery-Fear-Attempt',
    'VEHICLE CODE Driving Without Consent',
    'TRAFFIC Traffic Collision-Bicycle',
    'OBSCENE ACTIVITY Indecent Exposure',
    'INCIDENT Suspicious Person',
    'THEFT-GRAND Theft Bicycle',
    'FIRE Report of Smoke',
    'THEFT-MOTOR VEHICLE Theft from Motor Vehicle',
    'ALCOHOL Unlawful Possession of Alcohol',
    'SERVICE Suspicious Circumstances',
    'SERVICE Suspicious Social Media Post',
    'THEFT-GRAND Theft Grand-Plain',
    'DOMESTIC VIOLENCE Dating Violence',
    'SERVICE Welfare Check',
    'DISORDERLY CONDUCT Panhandling'
    
    

      ]

L3 = ['BURGLARY Burglary-Attempted-Commercial',
      'ROBBERY Robbery-Strong Arm-Attempt',
      'BATTERY Battery',
      'TRAFFIC Traffic Collision With Injuries',
      'HEALTH & SAFETY Forged or Altered Prescriptions'
      'ASSAULT-OTHER Assault-Other Simple, Not Aggravated',
      'BURGLARY-MOTOR VEHICLE Burglary-Motor Vehicle-Attempt',
      'NARCOTICS Possession of Drug Paraphernalia',
      'THEFT-FRAUD Defrauding An Innkeeper',
      'FIRE Fire-General',
      'ROBBERY Robbery-Fear',
      'ALCOHOL Alcohol/Drug Overdose',
      'ROBBERY Robbery-Carjacking',
      'THEFT-GRAND Theft Grand-Plain-Attempt',
      'ARSON Arson-Residential',
      'THEFT-FRAUD Embezzle Funds, Bad Checks, Forgery',
      'ROBBERY Robbery-Knife or Cutting Instrument-Attempt',
      'EH&S Gas Leak',
      'ROBBERY Robbery-Strong-Arm',
      'EH&S Water Leak',
      'ASSAULT Assault-School Employee',
      'BURGLARY Burglary-Commercial',
      'FIRE Alarm Malicious',
      'FIRE Fire-Commercial',
      'WEAPONS Possession of a Concealed Dirk or Dagger',
      'DISORDERLY CONDUCT Peeping Tom',
      'THEFT-GRAND PERSON Attempt Grand Theft Person',
      'NARCOTICS Possession of a Controlled Substance',
      'OBSCENE ACTIVITY Lewd Conduct',
      'ADMINISTRATIVE Violence in the Work Place',
      'VANDALISM Vandalism-Felony',
      'ALARM RESPONSE Carbon Monoxide Alarm',
      'HARASSMENT Harrasment-Sexual',
      'SEX OFFENSE Oral Copulation',
      'THEFT-GRAND AUTO Attempt Grand Theft Auto',
      'BURGLARY-OTHER Possession of Burglary Tools',
      'VEHICLE CODE Driving Under Influence',
      'IDENTITY THEFT Identity Theft',
      'BURGLARY Burglary-Attempted-Residential',
      'TRAFFIC Reckless Driver',
      'WEAPONS Possession of Metal Knuckles',
      'SERVICE Suspicious Circumstances',
      'VEHICLE CODE Hit & Run - Misdemeanor',
      'EXTORTION Extortion',
      'BURGLARY Burglary-Residential',
      'ALARM RESPONSE Environmental Alarm',
      'DOMESTIC VIOLENCE Willfull Infliction of Corporal Injury'
        'TRESPASS Trespassing'      

      ]

L4 = [
      'VEHICLE CODE Hit & Run - Felony',
      'ASSAULT Assault-Firearm',
      'BURGLARY-MOTOR VEHICLE Burglary-Motor Vehicle',
      'FIRE Fire-Vehicle',
      'ROBBERY Robbery-Other Dangerous Weapon',
      'ROBBERY Robbery-Firearm',
      'THEFT-GRAND AUTO Grand Theft Auto',
      'ARSON Arson-Commercial',
      'WEAPONS Brandishing A Weapon In a Deadly Manner',
      'WEAPONS Possession of a Concealed Firearm',
      'ROBBERY Robbery-Firearm-Attempt',
      'ROBBERY Robbery-Knife or Cutting Instrument',
      'ASSAULT Assault-Other Dangerous Weapon',
      'ARSON Arson-Attempt-Residential',
      'SEX OFFENSE Sodomy',
      'SEX OFFENSE Assault With Intent to Commit Sex Offense'
      'ASSAULT Assault-Knife or Cutting Instrument',
      'BATTERY Battery W/ Serious Injury',
      'SEX OFFENSE Oral Copulation',
      'THEFT-GRAND PERSON Grand Theft Person',
      'SEX OFFENSE Sexual Battery',
      'SEX OFFENSE Undetermined Sexual Assault',
      'KIDNAPPING Kidnapping-Attempt',
      'FIRE Fire-Residential'
      ]

L5 = ['HOMELAND SECURITY Bomb Threat-Commercial',
      'SUICIDE Suicide',
      'SERVICE Person Stuck In Elevator',
      'SEX OFFENSE Rape',
      'SEX OFFENSE Unlawful Sexual Intercourse',
      'SEX OFFENSE Statutory Rape',
      'HOMICIDE Attempt Murder, Non-Neg. Manslaughter',
      'SUICIDE Attempt Suicide',
      'SEX OFFENSE Penetration by Foreign Object',
      'DEATH Death by Undetermined Cause'
      ]

#print L1,L2,L3,L4,L5

df = pd.read_csv('nlp2.csv')
df['label'] = 0

for index,row in df.iterrows():
    if(row[0] in L1):
        df.iloc[index,'label'] = 