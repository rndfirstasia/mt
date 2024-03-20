import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_gsheets import GSheetsConnection

# CORE
#df = pd.read_excel('data.xlsx')
st.set_page_config(page_title="Management Trainee Profile", page_icon="icon.png")
st.set_option('deprecation.showPyplotGlobalUse', False)

conn = st.connection("gsheets", type=GSheetsConnection)

url = "https://docs.google.com/spreadsheets/d/1Rd__ozy2kEuNRJImq_wciBRBGKnCWSoP53fqD3I6dB8/edit#gid=0"
df = conn.read(spreadsheet=url, worksheet="0")
#-----

#Sidebar
st.sidebar.title('Filter')

#Filter Job Family
labels_fmly = ['L', 'P']
selected_jobfam = st.sidebar.multiselect('Jenis Kelamin', labels_fmly, default=labels_fmly)
df = df[df['J/K'].isin(selected_jobfam)]
#Filter
labels_mng = ['DANAMON', 'PT. Bela Parahiyangan Investindo', 'PT. Mitra Akademi Perkasa (MAP Group)', 'PT. Quadra Dinamika Internasional', 'PT. Satya Djaya Raya Trading Coy', 'PT. Satya Langgeng Sentosa', 'Solusi Bangun Indonesia, PT.']
selected_managerial = st.sidebar.multiselect('Klien', labels_mng, default=labels_mng)
df = df[df['KLIEN'].isin(selected_managerial)]

st.title('Management Trainee')

st.subheader('Jenis Kelamin')
distribusi = df['J/K'].value_counts().rename(index={'L': 'Laki-laki', 'P': 'Perempuan'})

def func(pct, allvalues):
    absolute = int(pct/100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

plt.figure(figsize=(6, 4))
plt.pie(distribusi, labels=distribusi.index, autopct=lambda pct: func(pct, distribusi), startangle=140)
st.pyplot(plt)

st.subheader('STAGE Distribution')
data_stage = df [['S','T','A','G','E']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_stage)
plt.show()
st.pyplot(plt)

data_stage_melted = pd.melt(data_stage, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_stage_melted)
plt.show()
st.pyplot(plt)

st.subheader('Stability Distribution')
data_trait_s = df[['S1','S2','S3','S4']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_trait_s)
plt.show()
st.pyplot(plt)

data_trait_s_melted = pd.melt(data_trait_s, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_trait_s_melted)
plt.show()
st.pyplot(plt)

st.subheader('Tenacity Distribution')
data_trait_t = df[['T1','T2','T3','T4', 'T5']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_trait_t)
plt.show()
st.pyplot(plt)

data_trait_t_melted = pd.melt(data_trait_t, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_trait_t_melted)
plt.show()
st.pyplot(plt)

st.subheader('Adaptability Distribution')
data_trait_a = df[['A1','A2','A3','A4']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_trait_a)
plt.show()
st.pyplot(plt)

data_trait_a_melted = pd.melt(data_trait_a, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_trait_a_melted)
plt.show()
st.pyplot(plt)

st.subheader('Genuineness Distribution')
data_trait_g = df[['G1','G2','G3','G4']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_trait_g)
plt.show()
st.pyplot(plt)

data_trait_g_melted = pd.melt(data_trait_g, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_trait_g_melted)
plt.show()
st.pyplot(plt)

st.subheader('Extrovert Distribution')
data_trait_e = df[['E1','E2','E3','E4','E5','E6']]
plt.figure(figsize=(10,8))
sns.kdeplot(data=data_trait_e)
plt.show()
st.pyplot(plt)

data_trait_e_melted = pd.melt(data_trait_e, var_name='Kolom', value_name='Nilai')
plt.figure(figsize=(10,8))
sns.boxplot(x='Kolom', y='Nilai', data=data_trait_e_melted)
plt.show()
st.pyplot(plt)

#Profil
st.subheader('Profil')
df_stage_profil = df[['S_Trait', 'T_Trait', 'A_Trait', 'G_Trait', 'E_Trait']]
df_stage_profil = df_stage_profil.rename(columns={'S_Trait': 'Stability', 'T_Trait':'Tenacity', 'A_Trait':'Adaptability', 'G_Trait':'Genuineness', 'E_Trait':'Extrovert'})

# Conversion to percentage and raw count
freq_dfs = []
total_counts = df_stage_profil.count()  # Total counts for each trait for raw count calculation
for trait in df_stage_profil.columns:
    freq_series = df_stage_profil[trait].value_counts(normalize=True) * 100
    raw_counts = df_stage_profil[trait].value_counts()  # Raw counts
    freq_df = freq_series.reset_index()
    freq_df['Count'] = freq_df['index'].apply(lambda x: raw_counts[x])  
    freq_df.columns = ['Value', 'Percentage', 'Count']
    freq_df['Trait'] = trait
    freq_dfs.append(freq_df)

final_df = pd.concat(freq_dfs)

traits = df_stage_profil.columns.tolist()
values = final_df['Value'].unique()
colors = ['#AEC6CF', '#B6D0E2', '#89CFF0']

fig, ax = plt.subplots(figsize=(10, 8))

starts = {trait: 0 for trait in traits}

for index, value in enumerate(values):
    percentages = []
    raw_counts = []
    for trait in traits:
        df_subset = final_df[(final_df['Trait'] == trait) & (final_df['Value'] == value)]
        if df_subset.empty:
            percentages.append(0)
            raw_counts.append(0)
        else:
            percentages.append(df_subset['Percentage'].values[0])
            raw_counts.append(df_subset['Count'].values[0])

    color = colors[index % len(colors)]
    bars = ax.barh(traits, percentages, left=[starts[trait] for trait in traits], color=color, label=value if index < len(colors) else "")

    # Update the starts for the next segment
    for i, trait in enumerate(traits):
        starts[trait] += percentages[i]
        # Annotate bars with percentage and count if space allows
        if percentages[i] > 0:  # Avoid cluttering the plot with annotations for very small or zero percentages
            ax.annotate(f"{values[index]}\n{percentages[i]:.2f}%\n({raw_counts[i]})",
                        xy=(starts[trait] - percentages[i] / 2, i),
                        xytext=(0, 0),  
                        textcoords="offset points",
                        ha='center', va='center',
                        color='black', fontsize=10)

ax.set_xlabel('Prosentase')
ax.set_title('Profil Management Trainee')

plt.tight_layout()
plt.show()
st.pyplot(plt)
