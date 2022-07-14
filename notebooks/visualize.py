from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from notebooks.utils import unpickler


def display_coherence(coherence: dict):
	"""Display linegraph for topic coherence"""
	# Make n_topics and tc lists
	n_topics = list(sorted(coherence.keys()))
	tc = [0 if np.isnan(coherence[x]) else coherence[x] for x in n_topics]

	# Plot
	fig = px.line(x=n_topics, y=tc, labels={'x': 'Number of Topics', 'y': 'Average Coherence'})
	fig.update_traces(mode="markers+lines")
	fig.update_xaxes(showspikes=True)
	fig.update_yaxes(showspikes=True)
	fig.update_layout(title_text=f"Coherence in n-Topic Models",
					  showlegend=False,
					  font=dict(size=10),
					  template='plotly_dark')

	return fig


def n_token_histogram(file: str):
	"""Display histogram of number of tokens per document"""
	# Open the file
	mycorpus = unpickler(file)
	# Calculate tokens
	ntokens = [len(el) for el in mycorpus]
	# Create the bins
	counts, bins = np.histogram(ntokens, bins=range(0, 200, 2))
	bins = 0.5 * (bins[:-1] + bins[1:])

	fig = px.bar(x=bins, y=counts, labels={'x': 'Number of Tokens', 'y': 'Number of Documents'})
	fig.update_layout(title_text=f"Tokens per Document",
					  showlegend=False,
					  font=dict(size=10),
					  template='plotly_dark')
	# Add text with mean in upper right corner
	fig.add_annotation(xref="paper", yref="paper", x=1, y=1, showarrow=False,
					   text=f"Average number of tokens per document: {round(np.mean(ntokens), 2)}")

	return fig


def display_topics(topic_report: dict, n: int):
	"""Display barchart of most frequent words for each topic"""
	# Creating two subplots
	fig = make_subplots(rows=int(n / 5), cols=5, start_cell="top-left",
						subplot_titles=([f"Topic {i}" for i in range(1, 26)]),
						horizontal_spacing=0.15,
						vertical_spacing=0.05
						)
	k = 1
	i = 1

	for topic in topic_report:
		# Assign possition in subplot grid
		if i > 5:
			k += 1
			i = i - 5

		# Extract data into lists
		words = [each['#text'] for each in topic['word']]
		count = [each['@count'] for each in topic['word']]

		# Add subplot
		fig.add_trace(
			go.Bar(
				x=count,
				y=words,
				orientation='h'),
			row=k, col=i
		)
		i += 1

	# Layout
	fig.update_xaxes(showgrid=False, showticklabels=False)
	fig.update_yaxes(showgrid=False, dtick=1)

	fig.update_layout(height=n * 75, title_text=f"Topic Models: {n} Topics",
					  showlegend=False,
					  font=dict(size=10),
					  template='plotly_dark')

	return fig


def display_documents(df1: pd.DataFrame, i: int = 0, j: int = 20) -> go.Figure:
	"""Display documents from corpus as BarPlot"""
	fig = go.Figure(layout=go.Layout(
		title="Topic Representation in Documents",
		yaxis=dict(
			title=None,
			showticklabels=True,
			dtick=1, )
	))
	for c in df1.columns:
		fig.add_trace(go.Bar(
			x=df1[c],
			y=df1.index[i:j],
			orientation='h',
			name=f'Topic {c}',
			text=[c for _ in df1.index[i:j]],
			hovertemplate="ID: %{y}<br>Topic %{text} (%{x:.2f}/1)",

		))
	# Add template
	fig.update_layout(template='plotly_dark', barmode='stack')
	# Remove x axes
	fig.update_xaxes(visible=False)

	return fig


def generate_wordcloud(topic_report, k, corpus):
	"""Display word clouds that make up a topic"""
	list_words = {word.get('#text'): float(word.get('@count')) for word in topic_report[k].get('word')}
	wc = WordCloud(mode="RGBA", background_color=None, width=300, height=300, margin=2, colormap='tab20')
	wc.fit_words(list_words)
	wc.to_file(f'app_data/img/{corpus}_topic_{k}.png')



def display_PCA(dataframe:pd.DataFrame) -> go.Figure:
	"""Displays PCA Bubble plot from a Topic Report's df"""
	# Define bubble size and hovertext
	hover_text = []
	bubble_size = []

	df = dataframe
	for index, row in dataframe.iterrows():
		hover_text.append((f'Tokens: {row["Tokens"]}'))
		bubble_size.append(row['Tokens'])

	df['text'] = hover_text
	df['size'] = bubble_size
	sizeref = 2.*max(df['size'])/(50**2)

	# Dictionary with dataframes for each topic
	topic_names = [f'Topic {i}' for i in range(len(hover_text))]
	topic_data = {topic : df.query("topic == '%s'" %i)
								  for i,topic in enumerate(topic_names)}

	# Create figure
	fig = go.Figure()

	for topic_name, topic in topic_data.items():
		fig.add_trace(go.Scatter(
			x=topic['PC 1'], y=topic['PC 2'],
			name=topic_name, text=topic['text'],
			marker_size=topic['size'],
			))

	# Tune marker appearance and layout
	fig.update_traces(mode='markers', marker=dict(sizemode='area',
												  sizeref=sizeref, line_width=2))

	fig.update_layout(
		title='Topic Models Visualization',
		xaxis=dict(title='PC1',),
		yaxis=dict(title='PC2',),
		template='plotly_dark',
	)
	return fig


def display_documents_tsme(data,n:int, hover_text:list):
	"""Display filtered document dataframe in TSME Scatter Plot """
	# Create figure
	fig = go.Figure()
	# Dictionary with dataframes for each topic
	topics_n = [ i for i in range(n) ]
	topic_data = {topic:data.query(f"topic == {int(topic)}")
								  for topic in topics_n}
	for topic_name, topic in topic_data.items():
		fig.add_trace(go.Scatter(
			x=topic['x'], y=topic['y'],
			name='Topic '+str(topic_name),
			hovertemplate=hover_text,
			hoverinfo="text"
			))
	fig.update_traces(mode="markers")
	fig.update_layout(
		title='Documents by Topic',
		template='plotly_dark',)

	return fig


def create_hovertexts(df, corpus):
	"""Create Hover Data for TSNE Graphs"""
	HOVER_TEXT  = []
	if corpus == 'patents':
		for index, row in df.iterrows():
			HOVER_TEXT.append(
				f'ID: {index}<br>' +
				f'Title: {row["Title"][:15]}[...]<br>' +
				f'Author: {row["Author"]}<br>' +
				f'Family ID: {row["Family ID"]}<br>' +
				f'Abstract: {row["Abstract"][:15]}[...]<br>' +
				f'Position: {row["Position"]}<br>' +
				f'Value: {row["Value"]}<br>' +
				f'IPC: {row["Class Symbol (1)"]}{row["Class Symbol (2)"]}<br>'
				+ f'Year: {row["Year"]}'
			)
		return HOVER_TEXT

	elif corpus == 'cordis':
		for index, row in df.iterrows():
			HOVER_TEXT.append(
				f'ID: {index}<br>' +
				f'Title: {row["Title"][:15]}[...]<br>' +
				f'Country: {row["Country"]}<br>' +
				f'Start Year: {row["Start Year"]}<br>' +
				f'End Year: {row["End Year"]}<br>' +
				f'Acronym: {row["Acronym"]}<br>' +
				f'Status: {row["Status"]}<br>' +
				f'Coordinator Country: {row["Coordinator Country"]}<br>' +
				f'Status: {row["Status"]}<br>' +
				f'Cost: {row["Cost"]}<br>' +
				f'Abstract: {row["Abstract"][:15]}[...]<br>' +
				f'RCN: {row["RCN"]}<br>' +
				f'DOI: {row["DOI"]}<br>' +
				f'Topic: {row["Topic"]} ({row["Topic Title"]})<br>'
				+ f'ESV Codes: {row["ESV Codes"]}'
			)
		return HOVER_TEXT

	elif corpus == 'semantic_scholar':
		for index, row in df.iterrows():
			HOVER_TEXT.append(
				f'ID: {index}<br>' +
				f'Title: {row["Title"][:15]}[...]<br>' +
				f'Journal: {row["Journal"]}<br>' +
				f'Pages: {row["Pages"]}<br>' +
				f'Volume: {row["Volume"]}<br>' +
				f'PMID: {row["PMID"]}<br>' +
				f'DOI: {row["DOI"]}<br>' +
				f'Abstract: {row["Abstract"][:15]}[...]<br>' +
				f'Venue: {row["Venue"]}<br>' +
				f'Lemmas: {row["Lemmas"]}<br>'
				+ f'Year: {row["Year"]}'
			)
		return HOVER_TEXT

