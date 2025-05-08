# Impor pustaka yang diperlukan
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings

# Suppress FutureWarning dari pandas untuk Plotly
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*When grouping with a length-1 list-like, you will need to pass a length-1 tuple.*"
)

# Muat dataset dari CSV
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQE3hZAiVnfnsbcd_FROCasP1h2dCS_woQe9ZEhuNGc294mIbhFjd_VAfeNSiav_HIn3vcd8oBuV8rQ/pub?gid=165225167&single=true&output=csv"
data = pd.read_csv(url)

# Rename kolom agar sesuai
data = data.rename(columns={
    'housing_median_age': 'HouseAge',
    'total_rooms': 'TotalRooms',
    'total_bedrooms': 'TotalBedrooms',
    'population': 'Population',
    'households': 'Households',
    'median_income': 'MedInc',
    'median_house_value': 'HARGA',
    'longitude': 'Longitude',
    'latitude': 'Latitude'
})

# Hitung AveRooms, hindari pembagian dengan nol
data['AveRooms'] = data.apply(
    lambda row: row['TotalRooms'] / row['Households'] if row['Households'] > 0 else np.nan, axis=1
)

# Tangani NaN di AveRooms dengan rata-rata
ave_rooms_mean = data['AveRooms'].mean(skipna=True)
data['AveRooms'] = data['AveRooms'].fillna(ave_rooms_mean)

# Tangani NaN di fitur lain dengan menghapus baris
data = data.dropna(subset=['Longitude', 'Latitude', 'HouseAge', 'TotalRooms', 'TotalBedrooms', 'Population', 'Households', 'MedInc', 'HARGA', 'ocean_proximity'])

# Skala HARGA ke satuan $100,000
data['HARGA'] = data['HARGA'] / 100000

# Encode ocean_proximity untuk model
data_encoded = pd.get_dummies(data, columns=['ocean_proximity'], prefix='ocean_proximity')

# Definisikan fitur numerik untuk model (termasuk encoded columns)
fitur_numerik = ['Longitude', 'Latitude', 'HouseAge', 'TotalRooms', 'TotalBedrooms', 'Population', 'Households', 'MedInc', 'AveRooms']
fitur_model = fitur_numerik + [col for col in data_encoded.columns if col.startswith('ocean_proximity_')]

# Latih model regresi linear
X = data_encoded[fitur_model]
y = data_encoded['HARGA']
model = LinearRegression()
model.fit(X, y)

# Inisialisasi aplikasi Dash
app = dash.Dash(__name__)

# Definisikan opsi fitur untuk dropdown
opsi_fitur = [
    {'label': 'Pendapatan Median (MedInc)', 'value': 'MedInc'},
    {'label': 'Usia Rumah (HouseAge)', 'value': 'HouseAge'},
    {'label': 'Rata-rata Kamar (AveRooms)', 'value': 'AveRooms'},
    {'label': 'Populasi', 'value': 'Population'},
    {'label': 'Jumlah Kamar (TotalRooms)', 'value': 'TotalRooms'},
    {'label': 'Jumlah Kamar Tidur (TotalBedrooms)', 'value': 'TotalBedrooms'},
    {'label': 'Jumlah Rumah Tangga (Households)', 'value': 'Households'},
    {'label': 'Lintang (Latitude)', 'value': 'Latitude'},
    {'label': 'Bujur (Longitude)', 'value': 'Longitude'},
    {'label': 'Kedekatan dengan Laut (Ocean Proximity)', 'value': 'ocean_proximity'}
]

# Definisikan fitur untuk slider
fitur_slider = ['MedInc', 'HouseAge', 'AveRooms', 'Population']

# Definisikan rentang slider
rentang_slider = {
    'MedInc': (max(data['MedInc'].min(), 0), max(data['MedInc'].max(), 1)),
    'HouseAge': (1, 52),
    'AveRooms': (1, 10),
    'Population': (max(data['Population'].min(), 1), max(data['Population'].max(), 2))
}

# Fungsi untuk menghasilkan marks dengan langkah aman
def generate_marks(min_val, max_val):
    range_val = max_val - min_val
    if range_val <= 1:
        step = 0.1
    else:
        step = max(range_val / 10, 0.1)
    return {float(i): f'{i:.1f}' for i in np.arange(min_val, max_val + step, step)}

# Definisikan tata letak aplikasi
app.layout = html.Div([
    html.H1("Dashboard Prediksi Harga Rumah California (Dataset Baru)", style={'textAlign': 'center'}),
    
    # Dropdown untuk memilih fitur visualisasi
    html.Label("Pilih Fitur untuk Visualisasi:"),
    dcc.Dropdown(
        id='dropdown-fitur',
        options=opsi_fitur,
        value='MedInc',
        style={'width': '50%'}
    ),
    
    # Visualisasi
    html.Div([
        # Scatter Plot
        html.Div([
            html.H3("Fitur vs Harga"),
            dcc.Graph(id='scatter-plot')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Histogram atau Box Plot
        html.Div([
            html.H3("Distribusi Fitur"),
            dcc.Graph(id='dist-plot')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Heatmap Korelasi
    html.H3("Peta Korelasi Fitur Numerik dan Harga"),
    dcc.Graph(id='heatmap'),
    
    # Slider untuk prediksi
    html.H3("Atur Nilai Fitur untuk Prediksi"),
    html.Div([
        html.Div([
            html.Label(f"{fitur}:"),
            dcc.Slider(
                id=f'slider-{fitur}',
                min=rentang_slider[fitur][0],
                max=rentang_slider[fitur][1],
                step=0.1,
                value=data[fitur].mean() if fitur not in ['HouseAge', 'AveRooms'] else (rentang_slider[fitur][0] + rentang_slider[fitur][1]) / 2,
                marks=generate_marks(rentang_slider[fitur][0], rentang_slider[fitur][1]),
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin': '10px'}) for fitur in fitur_slider
    ], style={'margin': '20px'}),
    
    # Tampilkan prediksi
    html.H3(id='output-prediksi', style={'textAlign': 'center'})
])

# Callback untuk memperbarui scatter plot dan distribusi plot
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('dist-plot', 'figure')],
    Input('dropdown-fitur', 'value')
)
def perbarui_visualisasi(fitur_dipilih):
    scatter_fig = px.scatter(data, x=fitur_dipilih, y='HARGA',
                             title=f'Harga vs {fitur_dipilih}',
                             labels={fitur_dipilih: fitur_dipilih, 'HARGA': 'Harga Rumah ($100,000)'},
                             color='ocean_proximity' if fitur_dipilih != 'ocean_proximity' else None)
    
    if fitur_dipilih == 'ocean_proximity':
        dist_fig = px.box(data, x=fitur_dipilih, y='HARGA',
                          title=f'Distribusi Harga berdasarkan {fitur_dipilih}',
                          labels={fitur_dipilih: fitur_dipilih, 'HARGA': 'Harga Rumah ($100,000)'})
    else:
        dist_fig = px.histogram(data, x=fitur_dipilih,
                                title=f'Distribusi {fitur_dipilih}',
                                labels={fitur_dipilih: fitur_dipilih})
    
    return scatter_fig, dist_fig

# Callback untuk memperbarui heatmap korelasi
@app.callback(
    Output('heatmap', 'figure'),
    Input('dropdown-fitur', 'value')
)
def perbarui_heatmap(_):
    corr_matrix = data[fitur_numerik + ['HARGA']].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(title="Peta Korelasi Fitur Numerik dan Harga")
    return fig

# Callback untuk memperbarui prediksi berdasarkan slider
@app.callback(
    Output('output-prediksi', 'children'),
    [Input(f'slider-{fitur}', 'value') for fitur in fitur_slider]
)
def perbarui_prediksi(*nilai_slider):
    input_data = data_encoded[fitur_model].mean().to_frame().T
    for fitur, nilai in zip(fitur_slider, nilai_slider):
        input_data[fitur] = nilai
    for col in [col for col in fitur_model if col.startswith('ocean_proximity_')]:
        input_data[col] = 1 if col == 'ocean_proximity_NEAR_BAY' else 0
    
    harga_prediksi = model.predict(input_data)[0]
    return f"Harga Rumah yang Diprediksi: ${harga_prediksi*100:.2f} Ribu"

# # Jalankan aplikasi
# if __name__ == '__main__':
#     app.run_server(debug=True)
