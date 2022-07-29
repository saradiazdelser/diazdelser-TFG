# Application of Natural Language Processing for Analysis of Science and Technology in the Biomedical Field

## Project Data
For the topic modeling visualization data, please refer to the following Zenodo dataset:

> Diaz del Ser, Sara. (2022). Visualization Data for Topic Modeling in the Field of Biotechnology (Bachelor's Thesis) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6835651



## Project Structure

```
diazdelser-TFG
├───README.md
│
├───dockerization
|
├───notebooks
│   ├   coherence.ipynb
│   ├   legend.md
│   ├   preprocessing.py
│   ├   pre_processing.ipynb
│   ├   preprocessing.py
│   ├   read_datasets.ipynb
│   ├   selecting_cordis.ipynb
│   ├   selecting_patents.ipynb
│   ├   selecting_semantic_scholar.ipynb
│   ├   stop_words.ipynb
│   ├   stopwords.pkl
│   ├   tsne_reduction.ipynb
│   ├   utils.py
│   └   visualize.py
│   
└───app
    ├────data
    │   ├   cordis_dataframe.pkl
    │   ├   cordis_topic_report.pkl
    │   ├   cordis_tsne.pkl
    │   ├   patents_dataframe.pkl
    │   ├   patents_topic_report.pkl
    │   ├   patents_tsne.pkl
    │   ├   semantic_scholar_dataframe.pkl
    │   ├   semantic_scholar_topic_report.pkl
    │   └   semantic_scholar_tsne.pkl
    │ 
    ├────assets
    │   ├   cordis_topic_0.png
    │   ├   ...
    │ 
    ├────visualize.py
    ├────utils.py
    └────app.py



```

