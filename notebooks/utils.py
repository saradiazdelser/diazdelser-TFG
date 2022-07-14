import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xmltodict
from sklearn.manifold import TSNE


def unpickler(file:str):
	"""Unpickle file"""
	with open(file, 'rb') as f:
		return pickle.load(f)


def pickler(file:str, ob):
	"""Pickle object to file"""
	with open(file, 'wb') as f:
		pickle.dump(ob, f)
	return


def read_report(file: str) -> (dict, int):
	with open(file, 'r') as f:
		topic_report = xmltodict.parse(f.read())

	topic_report = topic_report['topicModel']['topic']
	n = len(topic_report)
	return topic_report, n


def read_doc_topics(file: str) -> pd.DataFrame:
	"""Convert Document Topic file into dataframe"""
	df_docs = pd.read_csv(file, sep="\t", header=None, index_col=0)
	df_docs = df_docs.rename(columns={0: 'document', 1: 'id'})
	df_docs = df_docs.rename(columns={i + 2: i for i in range(len(df_docs.columns) - 1)})

	df_docs['id'] = df_docs['id'].astype('string')
	df_docs = df_docs.set_index(['id'], drop=True)
	return df_docs


def run_PCA(topic_report:list[dict]) -> pd.DataFrame:
	"""Run a PCA on a Topic Report and return results"""
	report = pd.DataFrame(topic_report)
	totalTokens = {}

	# Create an empty dictionary
	zero_list = [float(0) for _ in range(len(topic_report))]
	dictionary = {}

	for i, topic in enumerate(topic_report):
		# Save totalToken in dict for later
		totalTokens[topic['@id']] = topic['@totalTokens']
		# Extract data into lists
		for each in topic['word']:
			if not dictionary.get(each['#text']):
				dictionary[each['#text']] = [float(0) for _ in range(len(topic_report))]
			# Assign the frequency to the word
			dictionary[each['#text']][i] =+ float(each['@count'])

	new = pd.DataFrame.from_dict(dictionary)

	features = new.columns

	# Standarize data
	x = new.loc[:, features].values
	x = StandardScaler().fit_transform(x)

	# Run PCA
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents
				 , columns = ['PC 1', 'PC 2'])

	# Add topic and tokens colums
	new['topic'] = new.index.astype('int').astype('string')
	finalDf = pd.concat([principalDf, new[['topic']]], axis = 1)
	finalDf['Tokens'] = finalDf.apply(lambda x: totalTokens[x.topic], axis=1).astype('int')

	return finalDf



def tsne_model(df):
	"""Generate TSNE model from documents"""
	tsne_model = TSNE(n_components=2, verbose=1, random_state=7, angle=.99, init='pca') # 13-D -> 2-D
	tsne_lda = tsne_model.fit_transform(df)

	df['topic'] = df.apply( lambda x: df.columns[x == x.max()][0], axis=1)
	df['x'] = tsne_lda[:,0]
	df['y'] = tsne_lda[:,1]

	return df
