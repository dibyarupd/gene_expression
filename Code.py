import streamlit as st

st.title("Cancer prediction using 'Gene Expression Monitoring'")

# st.write("\n\n\n\n\n\n")

'''
    \n\nHi! My name is Rup! I love tech and technical advancements. So lately I have started to learn 'Machine Learning'.
    \nThis project is something that can be used to classify patients with 'acute myeloid leukemia' (AML) or 'acute lymphoblastic leukemia' (ALL). The dataset is from Kaggle (originally it comes from a proof-of-concept study published in 1999 by Golub et al).
'''

st.markdown("Click [here](https://www.kaggle.com/datasets/crawford/gene-expression) to download the data.")

"Now, lets jump on the code!"

with st.echo():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
   
'''   
    First, I have imported all the necessary libaries / dependancies for the whole code to run in an smooth and easy manner. My Python knowledge is mostly limited to libraries like 'pandas', 'NumPy', 'Matplotlib' and very few things from 'seaborn' and 'Streamlit'. In this project till now I have unleashed the power of data analysis libary pandas and machine learning library scikit learn. It's not like I know every single thing from this modules rather I am constantly improving my knowledge to develop this project to perform better as well as to make this notebook more aesthetic in nature.
'''

with st.echo():
    train = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\Rup's Palace\GeneExpression\data\data_set_ALL_AML_train.csv")
    test = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\Rup's Palace\GeneExpression\data\data_set_ALL_AML_independent.csv")
    labels = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\Rup's Palace\GeneExpression\data\actual.csv")

'''   
    I have loaded all the data into pandas dataframes for further processing. This data definitely need some cleaning. The row indexes of the data consist of 'Gene Accession Numbers' and the column indexes consist of 'samples' (or people on whom the dataset is made).
'''

with st.echo():
    cols = [col for col in train.columns if "call" not in col]
    train = train[cols]
    cols = [col for col in test.columns if "call" not in col]
    test = test[cols]
    labels = labels.replace({"ALL" : 0, "AML" : 1})
    label_names = ["ALL", "AML"]

'''
    The columns starting with 'call' are mainly useless for our work. So we have decided to drop those columns from both the training as well as test dataframes. Also machine learning algorithms don't work well with categorical data, so I have changed all the labels to numerical alias in the labels dataframe.
'''

with st.echo():
    train = train.T
    test = test.T
    train.columns = train.loc["Gene Accession Number"]
    test.columns = test.loc["Gene Accession Number"]
    train = train.drop(labels = ["Gene Description", "Gene Accession Number"])
    test = test.drop(labels = ["Gene Description", "Gene Accession Number"])

'''
    I transposed the dataframes to make them more related to the conventions.
'''

train.index = train.index.astype(int)
test.index = test.index.astype(int)
train = train.astype(int)
test = test.astype(int)

train = train.sort_index()
test = test.sort_index()

with st.echo():
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

'''
    Since the data is mostly imbalanced, we need to standardize the data. So I have performed standardization (another option could have been normalization) which makes the mean 0 and variance 1 for each distribution.
'''

y_train = labels[labels['patient'] <= 38]['cancer']
y_test = labels[labels['patient'] > 38]['cancer']
rcf = RandomForestClassifier()
rcf.fit(train, y_train)
prediction = rcf.predict(test)

'''
    I used random forest algorithm on the training dataset for it to predict correctly the labels for the test set.
'''
with st.echo():
    score = accuracy_score(y_test, prediction)
    st.write("Accuracy Score:", score)

'''
    As you can see the score is roughly 76 percent which means our algorithms predicted the labels with 73 percent accuracy. There's definitely room for some improvements.
'''