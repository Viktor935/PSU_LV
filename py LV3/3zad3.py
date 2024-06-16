import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 

import urllib
import urllib.request
import xml.etree.ElementTree as ET

# 1. Dohvaćanje mjerenja dnevne koncentracije lebdećih čestica PM10 za 2017. godinu za grad Osijek.
url = "http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=5&vrijemeOd=02.01.2017&vrijemeDo=01.01.2018"

airQualityHR = urllib.request.urlopen(url).read()
root = ET.fromstring(airQualityHR)
print(root)

df = pd.DataFrame(columns=('mjerenje', 'vrijeme'))

i = 0
while True:
    
    try:
         obj = root.getchildren()[i].getchildren()
    except:
         break
    
    row = dict(zip(['mjerenje', 'vrijeme'], [obj[0].text, obj[2].text]))
    row_s = pd.Series(row)
    row_s.name = i
    df = df.append(row_s)
    df.mjerenje[i] = float(df.mjerenje[i])
    i = i + 1

df.vrijeme = pd.to_datetime(df.vrijeme, utc=True)


df['month'] = pd.DatetimeIndex(df['vrijeme']).month 
df['dayOfweek'] = pd.DatetimeIndex(df['vrijeme']).dayofweek

 # 2. Ispis tri datuma u godini kada je koncentracija PM10 bila najveća.
topPM10values = df.sort_values(by=['mjerenje'], ascending=False)
print("\nTri datuma kad je koncentracija PM10 u 2017. bila najveca: ")
print(topPM10values['vrijeme'].head(3))