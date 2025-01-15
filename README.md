---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: Import Kebutuhan
``` python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

Dataset yang saya gunakan untuk pelatihan adalah https://www.kaggle.com/mariaren/covid19-healthy-diet-dataset/notebooks
``` python
df_makanan = pd.read_csv('Food_Supply_Quantity_kg_Data.csv')
df_protein = pd.read_csv('Protein_Supply_Quantity_Data.csv')
df_lemak = pd.read_csv('Fat_Supply_Quantity_Data.csv')
df_kategori = pd.read_csv('Supply_Food_Data_Descriptions.csv')
```

``` python
df_makanan.head()
```

::: OUTPUT
Country	Alcoholic Beverages	Animal fats	Animal Products	Aquatic Products, Other	Cereals - Excluding Beer	Eggs	Fish, Seafood	Fruits - Excluding Wine	Meat	...	Vegetables	Vegetal Products	Obesity	Undernourished	Confirmed	Deaths	Recovered	Active	Population	Unit (all except Population)
0	Afghanistan	0.0014	0.1973	9.4341	0.0	24.8097	0.2099	0.0350	5.3495	1.2020	...	6.7642	40.5645	4.5	29.8	0.142134	0.006186	0.123374	0.012574	38928000.0	%
1	Albania	1.6719	0.1357	18.7684	0.0	5.7817	0.5815	0.2126	6.7861	1.8845	...	11.7753	31.2304	22.3	6.2	2.967301	0.050951	1.792636	1.123714	2838000.0	%
2	Algeria	0.2711	0.0282	9.6334	0.0	13.6816	0.5277	0.2416	6.3801	1.1305	...	11.6484	40.3651	26.6	3.9	0.244897	0.006558	0.167572	0.070767	44357000.0	%
3	Angola	5.8087	0.0560	4.9278	0.0	9.1085	0.0587	1.7707	6.0005	2.0571	...	2.3041	45.0722	6.8	25	0.061687	0.001461	0.056808	0.003419	32522000.0	%
4	Antigua and Barbuda	3.5764	0.0087	16.6613	0.0	5.9960	0.2274	4.1489	10.7451	5.6888	...	5.4495	33.3233	19.1	NaN	0.293878	0.007143	0.190816	0.095918	98000.0	%
5 rows × 32 columns


``` python
df_protein.head()
```

::: OUTPUT

Country	Alcoholic Beverages	Animal Products	Animal fats	Aquatic Products, Other	Cereals - Excluding Beer	Eggs	Fish, Seafood	Fruits - Excluding Wine	Meat	...	Vegetable Oils	Vegetables	Obesity	Undernourished	Confirmed	Deaths	Recovered	Active	Population	Unit (all except Population)
0	Afghanistan	0.0	21.6397	6.2224	0.0	8.0353	0.6859	0.0327	0.4246	6.1244	...	17.0831	0.3593	4.5	29.8	0.142134	0.006186	0.123374	0.012574	38928000.0	%
1	Albania	0.0	32.0002	3.4172	0.0	2.6734	1.6448	0.1445	0.6418	8.7428	...	9.2443	0.6503	22.3	6.2	2.967301	0.050951	1.792636	1.123714	2838000.0	%
2	Algeria	0.0	14.4175	0.8972	0.0	4.2035	1.2171	0.2008	0.5772	3.8961	...	27.3606	0.5145	26.6	3.9	0.244897	0.006558	0.167572	0.070767	44357000.0	%
3	Angola	0.0	15.3041	1.3130	0.0	6.5545	0.1539	1.4155	0.3488	11.0268	...	22.4638	0.1231	6.8	25	0.061687	0.001461	0.056808	0.003419	32522000.0	%
4	Antigua and Barbuda	0.0	27.7033	4.6686	0.0	3.2153	0.3872	1.5263	1.2177	14.3202	...	14.4436	0.2469	19.1	NaN	0.293878	0.007143	0.190816	0.095918	98000.0	%
5 rows × 32 columns


``` python
df_lemak.head()
```

::: OUTPUT

Country	Alcoholic Beverages	Animal Products	Animal fats	Aquatic Products, Other	Cereals - Excluding Beer	Eggs	Fish, Seafood	Fruits - Excluding Wine	Meat	...	Vegetable Oils	Vegetables	Obesity	Undernourished	Confirmed	Deaths	Recovered	Active	Population	Unit (all except Population)
0	Afghanistan	0.0	21.6397	6.2224	0.0	8.0353	0.6859	0.0327	0.4246	6.1244	...	17.0831	0.3593	4.5	29.8	0.142134	0.006186	0.123374	0.012574	38928000.0	%
1	Albania	0.0	32.0002	3.4172	0.0	2.6734	1.6448	0.1445	0.6418	8.7428	...	9.2443	0.6503	22.3	6.2	2.967301	0.050951	1.792636	1.123714	2838000.0	%
2	Algeria	0.0	14.4175	0.8972	0.0	4.2035	1.2171	0.2008	0.5772	3.8961	...	27.3606	0.5145	26.6	3.9	0.244897	0.006558	0.167572	0.070767	44357000.0	%
3	Angola	0.0	15.3041	1.3130	0.0	6.5545	0.1539	1.4155	0.3488	11.0268	...	22.4638	0.1231	6.8	25	0.061687	0.001461	0.056808	0.003419	32522000.0	%
4	Antigua and Barbuda	0.0	27.7033	4.6686	0.0	3.2153	0.3872	1.5263	1.2177	14.3202	...	14.4436	0.2469	19.1	NaN	0.293878	0.007143	0.190816	0.095918	98000.0	%
5 rows × 32 columns


``` python
df_kategori.head()
```

::: OUTPUT
``` json
Categories	Items
0	Alcoholic Beverages	Alcohol, Non-Food; Beer; Beverages, Alcoholic;...
1	Animal fats	Butter, Ghee; Cream; Fats, Animals, Raw; Fish,...
2	Animal Products	Aquatic Animals, Others; Aquatic Plants; Bovin...
3	Aquatic Products, Other	Aquatic Animals, Others; Aquatic Plants; Meat,...
4	Cereals - Excluding Beer	Barley and products; Cereals, Other; Maize and...
```

Memisahkan kolom antara makanan dan bukan makanan 
``` python
semua_kolom = df_makanan.columns.tolist()
print(f'Semua Kolom : {semua_kolom} \n\n')

kolom_supplier = df_kategori['Categories'].tolist()
print(f'Makanan : {kolom_supplier} \n\n')

kolom_non_supplier = [i for i in semua_kolom if i not in kolom_supplier]
print(f'Bukan Makanan  : {kolom_non_supplier}')
```

OUTPUT
    Semua Kolom : ['Country', 'Alcoholic Beverages', 'Animal fats', 'Animal Products', 'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs', 'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat', 'Milk - Excluding Butter', 'Miscellaneous', 'Offals', 'Oilcrops', 'Pulses', 'Spices', 'Starchy Roots', 'Stimulants', 'Sugar & Sweeteners', 'Sugar Crops', 'Treenuts', 'Vegetable Oils', 'Vegetables', 'Vegetal Products', 'Obesity', 'Undernourished', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population', 'Unit (all except Population)'] 


    Makanan : ['Alcoholic Beverages', 'Animal fats', 'Animal Products', 'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs', 'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat', 'Milk - Excluding Butter', 'Miscellaneous', 'Offals', 'Oilcrops', 'Pulses', 'Spices', 'Starchy Roots', 'Stimulants', 'Sugar & Sweeteners', 'Sugar Crops', 'Treenuts', 'Vegetable Oils', 'Vegetables', 'Vegetal Products'] 


    Bukan Makanan  : ['Country', 'Obesity', 'Undernourished', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Population', 'Unit (all except Population)']


Menggabungkan DataFrame dengan menyatukan data kategori non-food/non-supply dengan kategori supply, di mana data kategori supply dari setiap DataFrame diberi label berdasarkan nama DataFrame-nya, seperti 'Meat_food', 'Meat_fat', dan 'Meat_protein'.
``` python
df_non_supplier_baru = df_makanan[kolom_non_supplier]
df_makanan_baru = df_makanan[['Country']+kolom_supplier]
df_protein_baru = df_protein[['Country']+kolom_supplier]
df_lemak_baru = df_lemak[['Country']+kolom_supplier]
```

``` python
df1=pd.merge(df_makanan_baru, df_protein_baru, on="Country", suffixes=('_food', '_protein'), how = 'outer')
df_lemak_baru_colm = [i + '_fat' if i != 'Country' else i for i in df_lemak_baru.columns]
df_lemak_baru.columns = df_lemak_baru_colm

df2=pd.merge(df_lemak_baru, df_non_supplier_baru, how='outer', on='Country')
df3 = pd.merge(df1, df2, left_index=True, right_index=True, how='left')
df3 = pd.merge(df1, df2, on='Country', how='left')
df3.columns.tolist()
```

::: OUTPUT

    ['Country',
     'Alcoholic Beverages_food',
     'Animal fats_food',
     'Animal Products_food',
     'Aquatic Products, Other_food',
     'Cereals - Excluding Beer_food',
     'Eggs_food',
     'Fish, Seafood_food',
     'Fruits - Excluding Wine_food',
     'Meat_food',
     'Milk - Excluding Butter_food',
     'Miscellaneous_food',
     'Offals_food',
     'Oilcrops_food',
     'Pulses_food',
     'Spices_food',
     'Starchy Roots_food',
     'Stimulants_food',
     'Sugar & Sweeteners_food',
     'Sugar Crops_food',
     'Treenuts_food',
     'Vegetable Oils_food',
     'Vegetables_food',
     'Vegetal Products_food',
     'Alcoholic Beverages_protein',
     'Animal fats_protein',
     'Animal Products_protein',
     'Aquatic Products, Other_protein',
     'Cereals - Excluding Beer_protein',
     'Eggs_protein',
     'Fish, Seafood_protein',
     'Fruits - Excluding Wine_protein',
     'Meat_protein',
     'Milk - Excluding Butter_protein',
     'Miscellaneous_protein',
     'Offals_protein',
     'Oilcrops_protein',
     'Pulses_protein',
     'Spices_protein',
     'Starchy Roots_protein',
     'Stimulants_protein',
     'Sugar & Sweeteners_protein',
     'Sugar Crops_protein',
     'Treenuts_protein',
     'Vegetable Oils_protein',
     'Vegetables_protein',
     'Vegetal Products_protein',
     'Alcoholic Beverages_fat_fat_fat_fat',
     'Animal fats_fat_fat_fat_fat',
     'Animal Products_fat_fat_fat_fat',
     'Aquatic Products, Other_fat_fat_fat_fat',
     'Cereals - Excluding Beer_fat_fat_fat_fat',
     'Eggs_fat_fat_fat_fat',
     'Fish, Seafood_fat_fat_fat_fat',
     'Fruits - Excluding Wine_fat_fat_fat_fat',
     'Meat_fat_fat_fat_fat',
     'Milk - Excluding Butter_fat_fat_fat_fat',
     'Miscellaneous_fat_fat_fat_fat',
     'Offals_fat_fat_fat_fat',
     'Oilcrops_fat_fat_fat_fat',
     'Pulses_fat_fat_fat_fat',
     'Spices_fat_fat_fat_fat',
     'Starchy Roots_fat_fat_fat_fat',
     'Stimulants_fat_fat_fat_fat',
     'Sugar & Sweeteners_fat_fat_fat_fat',
     'Sugar Crops_fat_fat_fat_fat',
     'Treenuts_fat_fat_fat_fat',
     'Vegetable Oils_fat_fat_fat_fat',
     'Vegetables_fat_fat_fat_fat',
     'Vegetal Products_fat_fat_fat_fat',
     'Obesity',
     'Undernourished',
     'Confirmed',
     'Deaths',
     'Recovered',
     'Active',
     'Population',
     'Unit (all except Population)']


``` python
df3
```

::: OUTPUT
Country	Alcoholic Beverages_food	Animal fats_food	Animal Products_food	Aquatic Products, Other_food	Cereals - Excluding Beer_food	Eggs_food	Fish, Seafood_food	Fruits - Excluding Wine_food	Meat_food	...	Vegetables_fat_fat_fat_fat	Vegetal Products_fat_fat_fat_fat	Obesity	Undernourished	Confirmed	Deaths	Recovered	Active	Population	Unit (all except Population)
0	Afghanistan	0.0014	0.1973	9.4341	0.0000	24.8097	0.2099	0.0350	5.3495	1.2020	...	0.3593	28.3684	4.5	29.8	0.142134	0.006186	0.123374	0.012574	38928000.0	%
1	Albania	1.6719	0.1357	18.7684	0.0000	5.7817	0.5815	0.2126	6.7861	1.8845	...	0.6503	17.9998	22.3	6.2	2.967301	0.050951	1.792636	1.123714	2838000.0	%
2	Algeria	0.2711	0.0282	9.6334	0.0000	13.6816	0.5277	0.2416	6.3801	1.1305	...	0.5145	35.5857	26.6	3.9	0.244897	0.006558	0.167572	0.070767	44357000.0	%
3	Angola	5.8087	0.0560	4.9278	0.0000	9.1085	0.0587	1.7707	6.0005	2.0571	...	0.1231	34.7010	6.8	25	0.061687	0.001461	0.056808	0.003419	32522000.0	%
4	Antigua and Barbuda	3.5764	0.0087	16.6613	0.0000	5.9960	0.2274	4.1489	10.7451	5.6888	...	0.2469	22.2995	19.1	NaN	0.293878	0.007143	0.190816	0.095918	98000.0	%
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
165	Venezuela (Bolivarian Republic of)	2.5952	0.0403	14.7565	0.0000	12.9253	0.3389	0.9456	7.6460	3.8328	...	0.1851	33.6855	25.2	21.2	0.452585	0.004287	0.424399	0.023899	28645000.0	%
166	Vietnam	1.4591	0.1640	8.5765	0.0042	16.8740	0.3077	2.6392	5.9029	4.4382	...	0.6373	16.7548	2.1	9.3	0.002063	0.000036	0.001526	0.000501	96209000.0	%
167	Yemen	0.0364	0.0446	5.7874	0.0000	27.2077	0.2579	0.5240	5.1344	2.7871	...	0.1667	37.4535	14.1	38.9	0.007131	0.002062	0.004788	0.000282	29826000.0	%
168	Zambia	5.7360	0.0829	6.0197	0.0000	21.1938	0.3399	1.6924	1.0183	1.8427	...	0.1567	40.3939	6.5	46.7	0.334133	0.004564	0.290524	0.039045	18384000.0	%
169	Zimbabwe	4.0552	0.0755	8.1489	0.0000	22.6240	0.2678	0.5518	2.2000	2.6142	...	0.0789	39.6248	12.3	51.3	0.232033	0.008854	0.190964	0.032214	14863000.0	%
170 rows × 78 columns

SORT & FILTER
``` python
df_sort = df3.sort_values(by=['Deaths'], ascending=False, ignore_index=True)
df_sort
```

::: OUTPUT

Country	Alcoholic Beverages_food	Animal fats_food	Animal Products_food	Aquatic Products, Other_food	Cereals - Excluding Beer_food	Eggs_food	Fish, Seafood_food	Fruits - Excluding Wine_food	Meat_food	...	Vegetables_fat_fat_fat_fat	Vegetal Products_fat_fat_fat_fat	Obesity	Undernourished	Confirmed	Deaths	Recovered	Active	Population	Unit (all except Population)
0	Belgium	5.3730	0.8559	17.7279	0.0010	6.6704	0.6487	1.1325	4.1623	3.2370	...	0.2982	23.2622	24.5	<2.5	6.286322	0.185428	0.000000	6.100894	11515000.0	%
1	Slovenia	4.9933	1.1248	18.9196	0.0005	7.6345	0.5376	0.6515	6.4367	4.1610	...	0.2697	23.3878	22.5	<2.5	8.235901	0.171755	7.312934	0.751213	2103000.0	%
2	United Kingdom	5.2632	0.2754	18.8798	0.0006	6.5412	0.6210	1.0911	4.9551	4.4181	...	0.2127	24.1332	29.5	<2.5	5.868483	0.167220	0.015161	5.686102	67160000.0	%
3	Czechia	9.8498	0.8945	17.8065	0.0006	5.6937	0.4964	0.5355	3.3962	4.7618	...	0.1729	24.8333	28.5	<2.5	9.612841	0.159845	8.555328	0.897667	10716000.0	%
4	Italy	3.1892	0.2834	19.0329	0.0005	8.5417	0.6247	1.5816	6.0207	4.2963	...	0.2277	28.1306	22.9	<2.5	4.353685	0.150927	3.494529	0.708229	60296000.0	%
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
165	Kiribati	0.2970	0.0486	8.8958	0.0000	8.2747	0.1640	6.1065	5.8907	2.3856	...	0.1397	39.4325	45.6	2.7	NaN	NaN	NaN	NaN	125000.0	%
166	Korea, North	0.8981	0.0010	3.3933	0.0010	17.9378	0.4668	1.1568	6.1383	1.3872	...	1.0357	38.5160	7.1	47.8	NaN	NaN	NaN	NaN	25779000.0	%
167	Myanmar	0.2195	0.1751	13.5188	0.0034	16.7608	0.4613	4.0424	4.0287	4.9753	...	0.2938	23.9530	5.7	10.6	NaN	NaN	NaN	NaN	54704000.0	%
168	New Caledonia	5.0363	0.0821	11.7818	0.0804	8.5840	0.7494	2.0361	5.5855	5.8687	...	0.1887	27.7282	NaN	7.1	NaN	NaN	NaN	NaN	295000.0	%
169	Turkmenistan	0.5038	0.3373	16.0611	0.0000	14.8029	0.5139	0.2137	4.1441	4.3767	...	0.4459	17.3085	17.5	5.4	NaN	NaN	NaN	NaN	6031000.0	%
170 rows × 78 columns


TESTING
``` python
negara = list(df_sort[df_sort['Country']== 'Indonesia'].index+1)[0]
print('Indonesia urutan : {}'.format(negara))
```

::: OUTPUT
    Indonesia urutan : 83
:::
:::

``` python
meninggal_persen = list(df_sort['Deaths'][df_sort['Country']=='Indonesia'])[0]
meninggal = list(df_sort['Deaths'][df_sort['Country']=='Indonesia']*df_sort['Population'][df_sort['Country']=='Indonesia']*0.01)[0]

print(f'Jumlah yang meninggal di Indonesia ada {round(meninggal_persen, 10)}% dari jumlah populasi.') # 10 digit
print(f'Jumlah yang meninggal di Indonesia ada {round(meninggal)} orang.')
```

::: OUTPUT
    Jumlah yang meninggal di Indonesia ada 0.0115526295% dari jumlah populasi.
    Jumlah yang meninggal di Indonesia ada 31393 orang.

Binning, Grouping, dan Aggregating data
``` python
labels = ['Sangat Tidak Banyak', 'Tidak Banyak', 'Sedang', 'Cukup Banyak', 'Sangat Banyak']
df_lemak['Confirmed Category'] = pd.qcut((df_lemak['Confirmed']), 5, labels=labels)
```

``` python
cols = ['Obesity', 'Meat', 'Fruits - Excluding Wine', 'Fish, Seafood', 'Animal Products', 'Vegetal Products']

groupby_confirmed = df_lemak.groupby('Confirmed Category')[cols].mean()
groupby_confirmed
```

::: OUTPUT
``` json
{"summary":"{\n  \"name\": \"groupby_confirmed\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"Confirmed Category\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Tidak Banyak\",\n          \"Sangat Banyak\",\n          \"Sedang\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Obesity\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6.382033532010137,\n        \"min\": 12.136363636363637,\n        \"max\": 24.903030303030302,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          12.136363636363637,\n          24.903030303030302,\n          19.24375\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Meat\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.4379305356348795,\n        \"min\": 6.856963636363636,\n        \"max\": 10.415409090909092,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          6.856963636363636,\n          10.415409090909092,\n          9.743353125\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fruits - Excluding Wine\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.18627634088335995,\n        \"min\": 0.4112939393939394,\n        \"max\": 0.865030303030303,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.865030303030303,\n          0.4112939393939394,\n          0.5181125\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fish, Seafood\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.1665288460625058,\n        \"min\": 0.5589242424242423,\n        \"max\": 0.9674787878787878,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.9608242424242425,\n          0.5589242424242423,\n          0.800271875\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Animal Products\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4.832726472976786,\n        \"min\": 15.073281818181817,\n        \"max\": 26.164790909090907,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          15.073281818181817,\n          26.164790909090907,\n          20.31525\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Vegetal Products\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4.832999170973168,\n        \"min\": 23.83529393939394,\n        \"max\": 34.92687575757576,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          34.92687575757576,\n          23.83529393939394,\n          29.684996875\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"groupby_confirmed"}
```