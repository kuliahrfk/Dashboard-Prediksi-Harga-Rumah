# Impor pustaka yang diperlukan
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np

# Muat dataset California Housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['HARGA'] = california.target  # Harga dalam $100,000

# Latih model regresi linear
X = data.drop('HARGA', axis=1)
y = data['HARGA']
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
    {'label': 'Rata-rata Penghuni (AveOccup)', 'value': 'AveOccup'},
    {'label': 'Lintang (Latitude)', 'value': 'Latitude'},
    {'label': 'Bujur (Longitude)', 'value': 'Longitude'}
]

# Definisikan fitur untuk slider
fitur_slider = ['MedInc', 'HouseAge', 'AveRooms', 'Population']

# Definisikan rentang slider
rentang_slider = {
    'MedInc': (data['MedInc'].min(), data['MedInc'].max()),
    'HouseAge': (1, 52),  # Dibatasi sesuai permintaan
    'AveRooms': (1, 10),  # Dibatasi sesuai permintaan
    'Population': (data['Population'].min(), data['Population'].max())
}

# Definisikan tata letak aplikasi
app.layout = html.Div([
    html.H1("Dashboard Prediksi Harga Rumah California", style={'textAlign': 'center'}),
    
    # Dropdown untuk memilih fitur visualisasi
    html.Label("Pilih Fitur untuk Visualisasi:"),
    dcc.Dropdown(
        id='dropdown-fitur',
        options=opsi_fitur,
        value='MedInc',  # Fitur default
        style={'width': '50%'}
    ),
    
    # Visualisasi
    html.Div([
        # Scatter Plot
        html.Div([
            html.H3("Fitur vs Harga"),
            dcc.Graph(id='scatter-plot')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Histogram
        html.Div([
            html.H3("Distribusi Fitur"),
            dcc.Graph(id='histogram')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Heatmap Korelasi
    html.H3("Peta Korelasi Fitur dan Harga"),
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
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin': '10px'}) for fitur in fitur_slider
    ], style={'margin': '20px'}),
    
    # Tampilkan prediksi
    html.H3(id='output-prediksi', style={'textAlign': 'center'})
])

# Callback untuk memperbarui scatter plot dan histogram
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('histogram', 'figure')],
    Input('dropdown-fitur', 'value')
)
def perbarui_visualisasi(fitur_dipilih):
    # Scatter Plot
    scatter_fig = px.scatter(data, x=fitur_dipilih, y='HARGA',
                             title=f'Harga vs {fitur_dipilih}',
                             labels={fitur_dipilih: fitur_dipilih, 'HARGA': 'Harga Rumah ($100,000)'})
    
    # Histogram
    hist_fig = px.histogram(data, x=fitur_dipilih,
                            title=f'Distribusi {fitur_dipilih}',
                            labels={fitur_dipilih: fitur_dipilih})
    
    return scatter_fig, hist_fig

# Callback untuk memperbarui heatmap korelasi
@app.callback(
    Output('heatmap', 'figure'),
    Input('dropdown-fitur', 'value')  # Input dummy untuk memicu callback
)
def perbarui_heatmap(_):
    corr_matrix = data.corr()
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
    fig.update_layout(title="Peta Korelasi Fitur dan Harga")
    return fig

# Callback untuk memperbarui prediksi berdasarkan slider
@app.callback(
    Output('output-prediksi', 'children'),
    [Input(f'slider-{fitur}', 'value') for fitur in fitur_slider]
)
def perbarui_prediksi(*nilai_slider):
    # Siapkan input untuk prediksi
    input_data = data.drop('HARGA', axis=1).mean().to_frame().T
    for fitur, nilai in zip(fitur_slider, nilai_slider):
        input_data[fitur] = nilai
    
    # Prediksi harga
    harga_prediksi = model.predict(input_data)[0]
    
    # Format output prediksi
    return f"Harga Rumah yang Diprediksi: ${harga_prediksi*100:.2f} Ribu"

# Jalankan aplikasi
#if __name__ == '__main__':
#    app.run_server(debug=True)
