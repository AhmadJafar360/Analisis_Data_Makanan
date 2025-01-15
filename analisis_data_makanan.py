import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_makanan = pd.read_csv('Data_Kuantitas_Pasokan_Makanan_KG.csv')
df_protein = pd.read_csv('Data_Kuantitas_Pasokan_Protein.csv')
df_lemak = pd.read_csv('Data_Kuantitas_Pasokan_Lemak.csv')
df_kategori = pd.read_csv('Deskripsi_Data_Pasokan_Makanan.csv')

df_makanan.head()

df_protein.head()

df_lemak.head()

df_kategori.head()

# menempatkan semua data kedalam all_columns
semua_kolom = df_makanan.columns.tolist()
print(f'Semua Kolom : {semua_kolom} \n\n')

# menempatkan data yang merupakan kategori food dalam supply_columns
kolom_supplier = df_kategori['Categories'].tolist()
print(f'Kolom Pemasok : {kolom_supplier} \n\n')

# menempatkan data yang merupakan kategori non food dalam non_supply_columns
kolom_non_supplier = [i for i in semua_kolom if i not in kolom_supplier]
print(f'Kolom Non-Supplier  : {kolom_non_supplier}')

# menyimpan data yang merupakan non supply
df_non_supplier_baru = df_makanan[kolom_non_supplier]
# menyimpan data food yang merupakan supply dan ditambahkan kolom country
df_makanan_baru = df_makanan[['Country']+kolom_supplier]
# menyimpan data protein yang merupakan supply dan ditambahkan kolom country
df_protein_baru = df_protein[['Country']+kolom_supplier]
# menyimpan data fat yang merupakan supply dan ditambahkan kolom country
df_lemak_baru = df_lemak[['Country']+kolom_supplier]

df1=pd.merge(df_makanan_baru, df_protein_baru, on="Country", suffixes=('_food', '_protein'), how = 'outer')
# Menambahkan _fat
df_lemak_baru_colm = [i + '_fat' if i != 'Country' else i for i in df_lemak_baru.columns]
df_lemak_baru.columns = df_lemak_baru_colm

df2=pd.merge(df_lemak_baru, df_non_supplier_baru, how='outer', on='Country')
df3 = pd.merge(df1, df2, left_index=True, right_index=True, how='left')
df3 = pd.merge(df1, df2, on='Country', how='left')
df3.columns.tolist()

df3

df_sort = df3.sort_values(by=['Deaths'], ascending=False, ignore_index=True)
df_sort

negara = list(df_sort[df_sort['Country']== 'Indonesia'].index+1)[0]
print('Indonesia urutan : {}'.format(negara))

meninggal_persen = list(df_sort['Deaths'][df_sort['Country']=='Indonesia'])[0]
meninggal = list(df_sort['Deaths'][df_sort['Country']=='Indonesia']*df_sort['Population'][df_sort['Country']=='Indonesia']*0.01)[0]

print(f'Jumlah yang meninggal di Indonesia ada {round(meninggal_persen, 10)}% dari jumlah populasi.') # 10 digit
print(f'Jumlah yang meninggal di Indonesia ada {round(meninggal)} orang.')

labels = ['Sangat Tidak Banyak', 'Tidak Banyak', 'Sedang', 'Cukup Banyak', 'Sangat Banyak']
df_lemak['Confirmed Category'] = pd.qcut((df_lemak['Confirmed']), 5, labels=labels)

cols = ['Obesity', 'Meat', 'Fruits - Excluding Wine', 'Fish, Seafood', 'Animal Products', 'Vegetal Products']

groupby_confirmed = df_lemak.groupby('Confirmed Category')[cols].mean()
groupby_confirmed