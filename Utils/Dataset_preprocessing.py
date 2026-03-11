import pandas as pd

GK=pd.read_csv('Train.csv')
IT=pd.read_csv('IT_QA_Dataset_50k.csv')
HR=pd.read_csv('HR_QA_Dataset_70k.csv')
SCI=pd.read_csv('Science_QA_Dataset_100k.csv')


# Dictionary mapping the specific index to the correct answer
answers = {
    6188: "50.8 cm",
    11881: "269",
    11951: "x = 0.10 * 5,000,000",
    13898: "1200 seconds",
    19231: "40",
    19832: "Requires a visual diagram of input, process, and output blocks.",
    19892: "192.168.1.1",
    24339: "1.35 tablespoons",
    24816: "21, 34",
    26905: "1876 (Bell Patent), 1973 (First Mobile Call), 2007 (iPhone).",
    32935: "2016",
    35102: "Japan GDP (2020) was approx. $5.06 trillion."
}

# Apply the answers to the 'Answer' column at the specific indices
for idx, val in answers.items():
    GK.at[idx, 'Answer'] = val




Questions_IT=IT['question']
Questions_GK=GK['Question']
Answers_GK=GK['Answer']
Answers_IT=IT['answer']
Question_HR=HR['question']
Answer_HR=HR['answer']
Questions_SCI=SCI['question']
Answers_SCI=SCI['answer']


Questions = pd.concat([Questions_IT, Questions_GK, Question_HR,Questions_SCI], axis=0, ignore_index=True)

Answers = pd.concat([Answers_IT, Answers_GK, Answer_HR,Answers_SCI], axis=0, ignore_index=True)


DT=pd.concat([Questions,Answers],axis=1)


DT=pd.concat([Questions,Answers],axis=1)

DT.columns=['Questions','Answers']

DT.to_csv('FYP1Data',index=False)

