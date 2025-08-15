# Summer project 2025: Decoding of IQ-data from the repetition code with GNN+RNN
by Vidar Petersson

This Git repo is a fork from the project and master's thesis "Sequential Graph-Based Decoding of the Surface Code using a Hybrid Graph and Recurrent Neural Network Model".

Under projektets gång så har jag fokuserat på och undersök följande områden
 - Avkodning av IQ-data 
 - Turning the knob
 - Train-at-all-times
 - Utvärdera och optimera GNN+RNN
 - Threshold och maximum likelihood  analys

## Projektets struktur

### Data pipeline
Generera data:
1. repetition_code_execute.py. Detta kan göras via IBM hårdvara (IBM Pittsburgh har använts för det mesta) eller den lokala Aer-simulatorn. Observera att Aer inte kan generera IQ-data. Man kan också använda examples\batch_execution_rep_code.py för att köra många jobb samtidigt och svepa över olika parametrar. En diskpresans är att filnamnet innehåller hur många faktiska tidsrepetitioner som gjordes, till skillnad från namningskonvetionen i projektet där t är antalet detektorer längs tidsaxeln. Detta innebär att man får +1 på t i filens namn för att få det riktiga t. 

Avkodare:
1. dataloader_ibm.py
2. graph_creator.py
3. gru_decoder.py: Contains our decoder, and methods to train and evaluate it.
körs genom att speca i args och exekveras med filerna i /examples

1. mwpm_decoder_ibm.py: Decodes jobdata with the mwpm algoritm. Can be used as a benchmark when evaluating our decoder. Can not currently utlizse the soft-info from iq-data

### project misc files

 - args.py: Contains the Args dataclass that is used to set various parameters for decoder training and inference.

 - training_utils.py: Contains some helper methods.

 - models: Directory containing weights and biases for models trained on jobdata files.

 - Examples: Directory containing some code examples showing how to load, train, and test our decoder.

## Fokusområden

### Avkodning av IQ-data 
### Turning the knob
### Train-at-all-times
### Utvärdera och optimera GNN+RNN
GNN+RNN
Uppdatering av hyperparametrar
### Threshold och maximum likelihood  analys

## Train-at-all-times
Tanken är att mäta det logiska tillståndet i varje tidsrepetition för att 



## Installation 
Första gångs installation:
```
git clone https://github.com/Vidar-Petersson/QEC_GNN-RNN_IBM.git
cd QEC_GNN_RNN_IBM
pip3 install -r requirements.txt
```

Make sure att ladda ner jobdata-filerna från projektets gemensamma lagringsenehet: /mimer/NOBACKUP/groups/snic2021-23-319/vidar_petersson_IBM_data/jobdata till ./jobdata mappen

Logga in på wandb med
```
wandb login
```

Kör sedan exempelvis träningen eller testning på jobdata-filerna
```
python3 examples/test_nn.py
```

## How to run on Alvis cluster
To execute ./examples/train_nn.py and verify that Alvis has received the job on currently running

Första gången för att skapa env
```
./create_container
```
Köra jobb
```
sbatch run_jobscript.sh
squeue --me 
```

För att avrbyta påbörjade jobb:
scancel *JOBID*


run python scripts: apptainer exec bash_container.sif python file.py