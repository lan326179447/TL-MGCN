# TL-MGCN
Reference: Transfer learning with molecular graph convolutional networks for accurate modelling and representation of bioactivities of ligands targeting G protein-coupled receptors without sufficient data

The web server of TL-MGCN, as well as the source codes and datasets is available at http://www.noveldelta.com/TL_MGCN for academic purposes.
Follow the instructions below, or consult the knowledge base for more information. If you have any questions, please contact us via live chat or email.

# Inroduction:
TL-MGCN, using transfer learning with molecular graph convolutional networks, for accurate modelling and representation of bioactivities of ligands targeting GPCR proteins without sufficient data. The pipeline of TL-MGCN consists of three steps: (i) To pretrain the molecular graph convolutional network model on the source domain dataset; (ii) To finetune the model on the target domain dataset; and (iii) To predict bioactivities of ligands from the target domain by random forest.

# Code usage:
All codes are used in the python3 version of the windows10 operating system.

Necessary packages: rdkit(rdkit.org), wdl_rf(pip install wdl_rf), and other basic packages

(1) demo_new: The program is a generic framework of TL-MGCN for ligand-based virtual screening. Users can develop their own virtual screening tools for drug targets in interest based on our codes. Input: Datasets in the format of SMILES strings with their corresponding p-bioactivity values. Output: Model performance (RMSE, r2). The steps are as follows: To input the datasets in the format of SMILES with their corresponding p-bioactivity values→To train the molecular graph convolutional network(MGCN) on the source domain dataset→To fine-tune the MGCN model on the target domain dataset→To obtain the molecular fingerprints→To build a random forest regression model→To get model performance(RMSE, r2)on the target domain dataset.
In this demo code, "A1.csv" is target domain dataset, "trained_weights4AS1.pkl" is the molecular graph convolutional network(MGCN) on the source domain dataset, i.e. AS1, which we have provided.

(2)demo_activity: The program can be used to predict the bioactivities of new compounds interacting with a given GPCR target. Input: Compounds in the format of SMILES strings. Output: Bioactivity values. The process is as follows: To input compounds in the format of SMILES strings→To obtain the molecular fingerprints through the given MGCN model→To get the bioactivities by the random forest model.
turn to:
trans("P47900",["CC1=CC=CC(=C1)C2=NOC(=N2)CN(C(C)C)C(=O)C3=CC(=CC(=C3)OC)C","C1=CC=C2C=C(C=CC2=C1)C=CC(=O)CCC(=O)O"])
transfile("P30939","data.csv")
where "P47900" is the given target, "["CC1=CC=CC(=C1)C2=NOC(=N2)CN(C(C)C)C(=O)C3=CC(=CC(=C3)OC)C","C1=CC=C2C=C(C=CC2=C1)C=CC(=O)CCC(=O)O"]" is the list of the SMILES strings. You can also upload your csv file to predict the bioactivities. "data.csv" is the demo file we provided.

(3)demo_fp: The program is used to generate various kinds of molecular fingerprints for given compounds. Input: Compounds in the format of SMILES strings. Output: Molecular fingerprints. The steps are as follows: To input compounds in the format of SMILES strings→To obtain molecular fingerprints based on the trained MGCN model for a given GPCR target.
turn to:
trans("P47900",["CC1=CC=CC(=C1)C2=NOC(=N2)CN(C(C)C)C(=O)C3=CC(=CC(=C3)OC)C","C1=CC=C2C=C(C=CC2=C1)C=CC(=O)CCC(=O)O"])
transfile("P30939","data.csv")
where "P47900" is the given target, "["CC1=CC=CC(=C1)C2=NOC(=N2)CN(C(C)C)C(=O)C3=CC(=CC(=C3)OC)C","C1=CC=C2C=C(C=CC2=C1)C=CC(=O)CCC(=O)O"]" is the list of the SMILES strings. You can also upload your csv file to generate the fingerprints. "data.csv" is the demo file we provided.

If you have any questions, please contact us by email: 1019172220@njupt.edu.cn
