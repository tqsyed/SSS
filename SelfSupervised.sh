##----------------------------Pen digits
#python main.py -s=True -save='A_' -type="noise" -path='/content/pen_digits.csv' -data='pen' -e=150 -fully=0 -semi=1 -p=1
#python main.py -s=False -load='A_153' -save='/content/pen_Classifier_NoiseSigma_' -path='/content/pen_digits.csv' -data='pen' -e=100 -fully=0 -semi=0
#---rot
#python main.py -s=True -save='SavedModel/pen_Self_Block_rot_' -type="rot" -path='pen_digits.csv' -data='pen' -e=150 -fully=0 -semi=1 -p=1
#python main.py -s=False -load='SavedModel/pen_Self_Block_rot_150' -save='SavedModel/pen_Classifier_RotSigma_' -path='pen_digits.csv' -data='pen' -e=100 -fully=0 -semi=0

##----------------------------------EEG-EYEMovement
python main.py -s=True -save='SavedModel_pen_Self_Block_Noise_' -type="noise" -path='eeg-eye-state_csv.csv' -data='pen' -e=50 -fully=0 -semi=1 -p=1
#python main.py -s=False -load='SavedModel_pen_Self_Block_Noise_150' -save='SavedModel/pen_Classifier_NoiseSigma_' -path='eeg-eye-state_csv.csv' -data='pen' -e=100 -fully=0 -semi=0
