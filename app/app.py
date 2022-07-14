# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from datetime import date
import utils as ut
import visualize as vis
from notebooks.utils import pickler,unpickler
import json
import plotly.graph_objects as go


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


colors = {
    'background': '#111111',
    'text': '#fefefa',
    'font-family': 'San Serif'
}

# -------------
# DATA
# -------------
patents = ut.unpickler(f'data/patents_dataframe.pkl')
cordis = ut.unpickler(f'data/cordis_dataframe.pkl')
semantic_scholar = ut.unpickler(f'data/semantic_scholar_dataframe.pkl')

patents_tsne = ut.unpickler(f'data/patents_tsne.pkl')
cordis_tsne = ut.unpickler(f'data/cordis_tsne.pkl')
semantic_scholar_tsne = ut.unpickler(f'data/semantic_scholar_tsne.pkl')

# -------------
# LAYOUT
# -------------

app.layout = html.Div([
    html.H1(
        children='Thematic Trends in the Biomedical Field',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='A web application framework for visualizing topic analysis in the biological and medical fields.',
             style={
                'textAlign': 'center',
                'color': colors['text'],
                'margin':20,
    }),

    dbc.Row([
        dbc.Col(html.Div([html.Div(children=[
        html.Label('Corpus'),
        dcc.Dropdown(['Publications', 'Projects', 'Patents',], 'Patents', id='corpus-input', style={ 'color': '#212121',}),
        ], style={'padding': 10, 'margin':20})],), width=2),

    dbc.Col(dbc.Row([
        dcc.Graph(id='topic-pca-graph'),
    ]), width = 8),

    dbc.Col(
        html.Div(children=[
            dcc.Markdown("""
                            **Topic WordCloud**

                            Click on a topic in the graph to view its WordCloud.
                        """,),
            html.Br(),
            html.Img(id='topic-wordcloud', src='assets/cordis_topic_0.png', style={'margin-top':20}, height=200),
        ], style={'margin':20}),width=2),
    ],),
    dbc.Row([
        dbc.Col(html.Div([
            # Patents DIV
            html.Div(children=[
            html.Label('Year'),
            dcc.RangeSlider(1960, 2022, 1, value=[1960, 2022],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            id='patent-year-input'),
            html.P('Time range of of IPC filing'), html.Br(),

            html.Label('Country'),
            dcc.Dropdown(patents['Author'].unique(), multi=True, id='patent-author-input',
                         style={'color': '#212121', }),
            html.P('Country of origin of IPC filing'), html.Br(),

            html.Label('IPC Class'),
            dcc.Dropdown(patents['Class Symbol (1)'].unique(), multi=True, id='patent-class-input',
                         style={'color': '#212121', }),
            html.P("Class Symbol for IPC's patent classification hierarchy"), html.Br(),

            html.Label('Family ID'),
            dcc.Dropdown(patents['Family ID'].unique(), multi=True, id='patent-family-input',
                         style={'color': '#212121', }),
            html.P('DOCDB simple patent family code'), html.Br(),

            html.Label('Information Value'),
            dcc.Dropdown(patents['Value'].unique(), multi=True, id='patent-value-input',
                         style={'color': '#212121', }),
            html.P('I: invention information or N: non-invention information'), html.Br(),

            ], style={'padding': 10, 'display':'block', 'visibility':'visible'}, id='patents-div'),

            # Cordis DIV
            html.Div(children=[
            html.Label('Year'),
            dcc.RangeSlider(1960, 2022, 1, value=[1960, 2022],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            id='cordis-year-input'),
            html.P('Starting year of the project'), html.Br(),

            html.Label('Status'),
            dcc.Dropdown(cordis['Status'].unique(), multi=True, id='cordis-status-input',
                         style={'color': '#212121', }), html.Br(),

            html.Label('Coordinator Country'),
            dcc.Dropdown(cordis['Coordinator Country'].unique(), multi=True, id='cordis-coor-count-input',
                         style={'color': '#212121', }),
            html.P('Coordinator Country of the Project'), html.Br(),

            html.Label('Cost'),
            dcc.RangeSlider(0, 2238016300, 1, value=[0, 2238016300],
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='cordis-cost-input'), html.Br(),

            html.Label('DOI'),
            dcc.Dropdown(cordis['RCN'].unique(), multi=True, id='cordis-doi-input',
                         style={'color': '#212121', }), html.Br(),

            ], style={'padding': 10, 'display':'none', 'visibility':'hidden'}, id='cordis-div'),

            # Semantic Scholar DIV
            html.Div(children=[
            html.Label('Year'),
            dcc.RangeSlider(1960, 2022, 1, value=[1960, 2022],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            id='semantic-scholar-year-input'),html.Br(),

            html.Label('Journal'),
            dcc.Dropdown(semantic_scholar['Journal'].unique(), multi=True, id='semantic-scholar-journal-input',
                         style={'color': '#212121', }), html.Br(),

            html.Label('PMID'),
            dcc.Dropdown(semantic_scholar['PMID'].unique(), multi=True, id='semantic-scholar-pmid-input',
                         style={'color': '#212121', }),
            html.P('PubMed ID of the article'), html.Br(),

            html.Label('Venue'),
            dcc.Dropdown(semantic_scholar['Venue'].unique(), multi=True, id='semantic-scholar-venue-input',
                         style={'color': '#212121', }), html.Br(),

            html.Label('DOI'),
            dcc.Dropdown(semantic_scholar['DOI'].unique(), multi=True, id='semantic-scholar-doi-input',
                         style={'color': '#212121', }), html.Br(),

            ], style={'padding': 10, 'display':'none', 'visibility':'hidden'}, id='semantic-scholar-div')

        ], style={'padding': 10, 'margin':20}), width=2),

        dbc.Col(dbc.Row([
            dcc.Graph(id='tsne-graph', style={'margin-top': 20, 'height': '50vh'}),
        ]), width=8),

        dbc.Col(dbc.Row([
        dcc.Markdown("""
                            **Document's Topic Distribution**

                            Click on a document in the graph to view its topic distribution.
                        """,),
            html.Br(),
            dcc.Graph(id='pie-chart-graph'),
        ],style={'margin':20}), width=2)
    ],)
],)

# -------------
# CONTROLS
# -------------

@app.callback(
    Output('topic-wordcloud', 'src'),
    Input('corpus-input', 'value'),
    Input('topic-pca-graph', 'clickData'))

def update_wordcloud(selected_corpus, clickData):
    corpus = { 'Publications':'semantic_scholar','Projects': 'cordis','Patents': 'patents' }
    if clickData:
        topic_n = clickData['points'][0]['curveNumber']
    else:
        topic_n = 0
    src = f'{corpus[selected_corpus]}_topic_{topic_n}.png'
    return app.get_asset_url(src)


@app.callback(
    Output('tsne-graph', 'figure'),
    Input('corpus-input', 'value'),

    # Filter Inputs
    Input('patent-year-input', 'value'),
    Input('patent-author-input', 'value'),
    Input('patent-class-input', 'value'),
    Input('patent-family-input', 'value'),
    Input('patent-value-input', 'value'),

    Input('semantic-scholar-year-input', 'value'),
    Input('semantic-scholar-journal-input', 'value'),
    Input('semantic-scholar-pmid-input', 'value'),
    Input('semantic-scholar-venue-input', 'value'),
    Input('semantic-scholar-doi-input', 'value'),

    Input('cordis-year-input', 'value'),
    Input('cordis-status-input', 'value'),
    Input('cordis-coor-count-input', 'value'),
    Input('cordis-doi-input', 'value'),
    Input('cordis-cost-input', 'value'),

)
def update_figure_docs(selected_corpus,
                    patent_year, patent_author, patent_class, patent_family, patent_value,
                    ss_year, ss_journal, ss_pmid, ss_venue, ss_doi,
                    cordis_year, cordis_status, cordis_country, cordis_doi,cordis_cost
):
    corpus = { 'Publications':'semantic_scholar','Projects': 'cordis','Patents': 'patents' }
    # Create hovertext
    if corpus[selected_corpus] == 'patents':
        # Join both dataframes
        patents.drop_duplicates('Patent ID', inplace=True)
        unfiltered_df = patents_tsne.join(patents.set_index('Patent ID', drop=False))
        # Generate hovertext
        HOVER_TEXT = vis.create_hovertexts(unfiltered_df, corpus[selected_corpus])
        n_topics = 20
        filtered = unfiltered_df

        # Filters
        if patent_year:
            filtered = filtered[filtered['Year'].apply(lambda x: str(x).isdigit())]
            filtered = filtered.loc[filtered['Year'].astype('int').between(patent_year[0], patent_year[1])]
        if patent_author:
            filtered = filtered.loc[filtered['Author'].isin(patent_author)]
        if patent_class:
            filtered = filtered.loc[filtered['Class Symbol (1)'].isin(patent_class)]
        if patent_family:
            filtered = filtered.loc[filtered['Family ID'].isin(patent_family)]
        if patent_value:
            filtered = filtered.loc[filtered['Value'].isin(patent_value)]

        # Visualize
        return vis.display_documents_tsme(filtered, n_topics, HOVER_TEXT)

    elif corpus[selected_corpus] == 'cordis':
        # Generate hovertext
        HOVER_TEXT = vis.create_hovertexts(cordis_tsne, corpus[selected_corpus])
        n_topics = 15
        filtered = cordis_tsne

        # Filters
        if cordis_year:
            # Remove non-digits
            filtered = filtered[filtered['Year'].apply(lambda x: str(x).isdigit())]
            filtered = filtered.loc[filtered['Year'].astype('int').between(cordis_year[0], cordis_year[1])]
        if cordis_status:
            filtered = filtered.loc[filtered['Status'].isin(cordis_status)]
        if cordis_country:
            filtered = filtered.loc[filtered['Coordinator Country'].isin(cordis_country)]
        if cordis_doi:
            filtered = filtered.loc[filtered['DOI'].isin(cordis_doi)]
        if cordis_cost:
            filtered = filtered[filtered['Cost'].apply(lambda x: str(x).isdigit())]
            filtered = filtered.loc[filtered['Cost'].astype('int').between(cordis_cost[0], cordis_cost[1])]

        # Visualize
        return vis.display_documents_tsme(filtered, n_topics, HOVER_TEXT)

    elif corpus[selected_corpus] == 'semantic_scholar':
        # Generate hovertext
        HOVER_TEXT = vis.create_hovertexts(semantic_scholar_tsne, corpus[selected_corpus])
        n_topics = 20
        filtered = semantic_scholar_tsne.head(50000)

        # Filters
        if ss_year:
            filtered = filtered[filtered['Year'].apply(lambda x: str(x).isdigit())]
            filtered = filtered.loc[filtered['Year'].astype('int').between(ss_year[0], ss_year[1])]
        if ss_journal:
            filtered = filtered.loc[filtered['Journal'].isin(ss_journal)]
        if ss_pmid:
            filtered = filtered.loc[filtered['PMID'].isin(ss_pmid)]
        if ss_venue:
            filtered = filtered.loc[filtered['Venue'].isin(ss_venue)]
        if ss_doi:
            filtered = filtered.loc[filtered['DOI'].isin(ss_doi)]

        # Visualize
        return vis.display_documents_tsme(filtered, n_topics, HOVER_TEXT)



@app.callback(
    Output('topic-pca-graph', 'figure'),
    Input('corpus-input', 'value'))
def update_figure_pca(selected_corpus):
    corpus = { 'Publications':'semantic_scholar','Projects': 'cordis','Patents': 'patents' }
    topic_report = unpickler(f'data/{corpus[selected_corpus]}_topic_report.pkl')
    df = ut.run_PCA(topic_report)
    fig = vis.display_PCA(df)
    return fig

@app.callback(
    Output('patents-div', 'style'),
    Output('cordis-div', 'style'),
    Output('semantic-scholar-div', 'style'),
    Input('corpus-input', 'value'))
def update_menu(selected_corpus):
    corpus = { 'Publications':'semantic_scholar','Projects': 'cordis','Patents': 'patents' }
    if corpus[selected_corpus] == 'semantic_scholar':
        return {'visibility': 'hidden', 'display': 'none'}, {'visibility': 'hidden', 'display': 'none'},{'display':'block', 'visibility':'visible'}
    elif corpus[selected_corpus] == 'patents':
        return {'display':'block', 'visibility':'visible'}, {'visibility': 'hidden', 'display': 'none'},{'visibility': 'hidden', 'display': 'none'}
    elif corpus[selected_corpus] == 'cordis':
        return {'visibility': 'hidden', 'display': 'none'}, {'display':'block', 'visibility':'visible'},{'visibility': 'hidden', 'display': 'none'}


@app.callback(
    Output('pie-chart-graph', 'figure'),
    Input('corpus-input', 'value'),
    Input('tsne-graph', 'clickData')
)
def upload_piehart(selected_corpus,clickData):
    corpus = { 'Publications':'semantic_scholar','Projects': 'cordis','Patents': 'patents' }
    if clickData:
        doc_id = clickData['points'][0]['hovertemplate'].split('<br>')[0].split(': ')[1]
        clickData = None
        if corpus[selected_corpus] == 'patents':
            n_topics = [i for i in range(20)]
            values = patents_tsne.loc[[str(doc_id)]][n_topics].values.tolist()[0]
        elif corpus[selected_corpus] == 'cordis':
            n_topics = [i for i in range(15)]
            values = cordis_tsne.loc[[str(doc_id)]][n_topics].values.tolist()[0]
        elif corpus[selected_corpus] == 'semantic_scholar':
            n_topics = [i for i in range(20)]
            values = semantic_scholar_tsne.loc[[str(doc_id)]][n_topics].values.tolist()[0]

        # Plot figure
        fig = go.Figure(data=[go.Pie(labels=n_topics, values=values)])
        fig.update_traces(textposition='inside', hoverinfo='percent', textinfo='label')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', template='plotly_dark', showlegend=False)
        return fig
    else:
        return go.Figure(data=[go.Pie(labels=[1], values=[""])])

if __name__ == '__main__':
    app.run_server(debug=True)