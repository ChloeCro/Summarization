# Summarization of Dutch Legal Cases (Rechtspraak)
Master thesis by Chloe Crombach

## Installation
### Environment
Create a new environment with conda:
```
conda create -n LegNLSum
conda activate LegNLSum
```

### Requirements
Install the requirements:

```
pip install -r requirements.txt
```

### Data
Get the Rechtspraak cases zip file [here](http://static.rechtspraak.nl/PI/OpenDataUitspraken.zip). Then run the unzip file to acquire folders for each year in a data folder.
```
python utils/unzip.py
```

## Run the code

### Available summarization methods
The summarization methods implemented are numbered and ordered in extractive and abstractive methods.

<b> Extractive Methods </b>

<b> Abstractive Methods </b>