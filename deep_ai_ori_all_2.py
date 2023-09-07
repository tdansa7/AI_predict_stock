import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Concatenate, Dropout


#listcode = ["8591"]
#listcode = ["8306","7203","6098","8002","9104"]
#listcode = ["2579","2212","4004","9531","2002","4151","4502"]
listcode = ["2579"]
#listcode = ["9048","9502","2579","7203","7267","2212","6301","4004","2002","2229","6098","3231","7011","9531"]
days_a = 5
epochcaa=200
batch_sizeaa=32
clayer = 2
nodec = 30 
sum = 0
for codem in listcode:
    # データを読み込む
    path="stock_data\\list_"
    df = pd.read_csv(path+codem+'_day_com_hou.csv')

    # 特徴量を作成する
    df['PriceDiff'] = (df['Close'] - df['Open'])/df['Open']
    df['Up'] = (df['PriceDiff'] > 0.01).astype(int)
    df['Opendiv'] = df['Open']/df['Close'].shift(1)

    for i in range(1, 6):
        df[f'hour{i}'] = (df[f'hour{i}']-df['Open'])/df['Open']
    sumopen = 0
    for i in range(1, 6):
        sumopen = sumopen + df['Close'].shift(i)
    sumopen = sumopen*0.2
    df['avediv5'] = (df['Open']- sumopen)/sumopen
    sumopen = 0
    for i in range(1, 26):
        sumopen = sumopen + df['Close'].shift(i)
    sumopen = sumopen*0.04
    df['avediv'] = (df['Open']- sumopen)/sumopen


    # 過去5日間のデータを使用して特徴量を作成する
    for i in range(1, days_a+1):
        df[f'PriceDiff_{i}'] = (df['Close'].shift(i) - df['Open'].shift(i))/df['Open'].shift(i)
        df[f'Up_{i}'] = (df[f'PriceDiff_{i}'] > 0.01).astype(int)
        #df[f'DOUWN_{i}'] = (df[f'PriceDiff_{i}'] < 0.01).astype(int)
        #df[f'high{i}'] = df['high'].shift(i)/df['Open'].shift(i)
        #df[f'low{i}'] = df['low'].shift(i)/df['Open'].shift(i)
        df[f'high{i}'] = df['high'].shift(i)/df['Open']
        df[f'low{i}'] = df['low'].shift(i)/df['Open']
        df[f'volume{i}'] = df['volume'].shift(i)/df['volume'].shift(1)



    # NaNを含む行を削除する
    df.dropna(inplace=True)
    df = df.drop(['timestamp','datetime'], axis=1)
    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: -0.02 if x <= -0.02 else x)
    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: 0.02 if x >= 0.02 else x)
    #df = df.reset_index(drop=True)
    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: -0.02 if x <= -0.02 else x)
    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: 0.02 if x >= 0.02 else x)

    dfo = df.copy()
    df.to_csv('output'+codem+'.csv')
    
    for ROWNAME, ROWDATA in df.iteritems():
        AVEROW = ROWDATA.mean()
        STDROW = 2*ROWDATA.std()
        max_value = AVEROW + STDROW
        min_value = AVEROW - STDROW
        #STDROW = ROWDATA.std()/(max_value-min_value)
        
        df[ROWNAME] = df[ROWNAME].apply(lambda x: AVEROW - STDROW if x < AVEROW - STDROW else x)
        df[ROWNAME] = df[ROWNAME].apply(lambda x: AVEROW + STDROW if x > AVEROW + STDROW else x)

        STDROW = ROWDATA.std()/(max_value-min_value)

        df[ROWNAME] = (2*STDROW-1)*df[ROWNAME]/(min_value-max_value) + (min_value-min_value*STDROW-STDROW*max_value)/(min_value-max_value)
        
        #df[ROWNAME] = df[ROWNAME].apply(lambda x: AVEROW - STDROW if x < AVEROW - STDROW else x)
        #df[ROWNAME] = df[ROWNAME].apply(lambda x: AVEROW + STDROW if x > AVEROW + STDROW else x)

    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: -0.02 if x <= -0.02 else x)
    #df['PriceDiff'] = df['PriceDiff'].apply(lambda x: 0.02 if x >= 0.02 else x)
    


    #df.to_csv('output'+codem+'.csv')
    df['volume1'] = 1

    #max_value = df['PriceDiff'].max()
    #min_value = df['PriceDiff'].min()
    #max_value = dfo['PriceDiff'].max()
    #min_value = dfo['PriceDiff'].min()
    #STDROW    = dfo['PriceDiff'].std()/(max_value-min_value)
    AVEROW = dfo['PriceDiff'].mean()
    STDROW = 2*dfo['PriceDiff'].std()
    max_value = AVEROW + STDROW
    min_value = AVEROW - STDROW
    STDROW    = dfo['PriceDiff'].std()/(max_value-min_value)



    
    #scaler = MinMaxScaler()
    #scaled_data = scaler.fit_transform(df)
    #df = pd.DataFrame(scaled_data, columns=df.columns)

    df.to_csv('output2'+codem+'.csv')

    X = df.drop(['Up','PriceDiff',"Close","high","low","volume","Open"], axis=1)
    #X = df.drop(['Up','timestamp','datetime','PriceDiff',"Close","high","low","volume","Open"], axis=1)
    y = df['PriceDiff']
    for i in reversed(range(1, 6)):
        # 特徴量とラベルを分割する
        X = X.drop([f'hour{i}'], axis=1)
     

        # 訓練データとテストデータに分割する
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_list = [X_train,X_test]
        data_dict = {}
        j=0
        for X_data in X_list:
            name = f'X_data{2*j+1}_'
            X1 = X_data
            X1_len = len(X1) 
            selected_columns = []
            for k in range(1, days_a+1):
                new_elements = [f'PriceDiff_{k}',f'Up_{k}', f'high{k}', f'low{k}', f'volume{k}']
                #new_elements = [f'PriceDiff_{k}', f'high{k}', f'low{k}', f'volume{k}']
                selected_columns.extend(new_elements)
            selected_data1 = X1[selected_columns].values
            
            data_dict[name] = selected_data1.reshape(X1_len, days_a,5)
            #data_dict[name] = np.transpose(data_dict[name], (0, 2, 1))
            #print(data_dict[name].shape)
            #print(X_data)
            #print(data_dict[name])
            name = f'X_data{2*j+2}_'
            X1 = X1.drop(selected_columns, axis=1)

            data_dict[name] = np.array(X1)

            j = j + 1

        
        # モデルの定義
        print(days_a)
        input1 = Input(shape=(days_a,5))
        input2 = Input(shape=(2+i,))
        #input1 = Input(shape=(days_a*5+2+i))
        

        #conv1 = Conv1D(1, kernel_size=5, activation='leaky_relu')(input1)
        #pool1 = MaxPooling1D(pool_size=2)(conv1)
        #conv2 = Conv1D(1, kernel_size=2, activation='sigmoid')(conv1)
        #pool2 = MaxPooling1D(pool_size=2)(conv2)
        #flatten = Flatten()(conv2)
        #concat = Concatenate()([flatten, input2])
        #dense1 = Dense(128, activation='leaky_relu')(concat)
        #dense2 = Dense(64, activation='leaky_relu')(dense1)
        #dense3 = Dense(64, activation='tanh')(dense2)
        #dense4 = Dense(64, activation='leaky_relu')(dense3)
        #dropout1 = Dropout(0.1)(dense2)  # ドロップアウト率を0.2に設定
        #dense = Dense(32, activation='leaky_relu')(dense4)
        #dropout2 = Dropout(0.1)(dense)  # ドロップアウト率を0.2に設定
        #output = Dense(1, activation='leaky_relu')(dense)


        flatten = Flatten()(input1)
        concat = Concatenate()([flatten, input2])
        dense = Dense(128)(concat)
        #dense1 = Dense(128, activation='leaky_relu')(dense)
        dense2 = Dense(64)(dense)
        dropout2 = Dropout(0.1)(dense2)
        #dense3 = Dense(64, activation='tanh')(dense2)
        output = Dense(1)(dropout2)


        model = tf.keras.Model(inputs=[input1, input2], outputs=output)

        # モデルのコンパイル
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.optimizer.lr = 0.000001

        # モデルのサマリを表示
        model.summary()

        # モデルを訓練する
        # モデルの訓練
        model.fit([data_dict['X_data1_'], data_dict['X_data2_']], y_train, epochs=epochcaa, batch_size=batch_sizeaa)
        with open(f'model{i}_'+codem+'.pkl', mode='wb') as f:
            pickle.dump(model, f)
        # モデルの評価
        loss = model.evaluate([data_dict['X_data3_'], data_dict['X_data4_']], y_test)
        print('Test Loss:', loss)
        #print('Test Accuracy:', accuracy)

        #y_test = (max_value-min_value)*y_test+min_value
        y_test = (min_value-max_value)*(y_test-(min_value-min_value*STDROW-STDROW*max_value)/(min_value-max_value))/(2*STDROW-1)



        y_test.to_csv('output2'+codem+'.csv')

        y_pred = model.predict([data_dict['X_data3_'], data_dict['X_data4_']])
        with open('output3'+codem+'.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                k=0
                for item in y_pred:
                    item = (min_value-max_value)*(item-(min_value-min_value*STDROW-STDROW*max_value)/(min_value-max_value))/(2*STDROW-1)
                    y_pred[k] = item

                    writer.writerow(item)
                    k=k+1
        if i == 2:
            y_pred2 = y_pred
        listdiv = []
        k=0
        if i==1:
            for j in X_test.index.tolist():
                if y_pred[k] > 0.01:
                    if y_pred2[k] > 0.01:
                        sum = sum + dfo.loc[j,'Close'] - dfo.loc[j,'Open']
                        listdiv.append(dfo.loc[j,'Close'] - dfo.loc[j,'Open'])
                    else:
                        sum = sum + dfo.loc[j,'hour1'] * dfo.loc[j,'Open']
    
                        listdiv.append(dfo.loc[j,'hour1'] * dfo.loc[j,'Open'])
                if y_pred[k] < -0.01:
                    if y_pred2[k] < -0.01:
                        sum = sum - dfo.loc[j,'Close'] + dfo.loc[j,'Open']
                    
                        listdiv.append(-dfo.loc[j,'Close'] + dfo.loc[j,'Open'])
                    else:
                        sum = sum - dfo.loc[j,'hour1'] * dfo.loc[j,'Open']
                    
                        listdiv.append(-dfo.loc[j,'hour1'] * dfo.loc[j,'Open'])

                k=k+1
            
            with open('output4'+codem+'.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for item in listdiv:
                    writer.writerow([item])
        

    # 特徴量とラベルを分割する
    X = df.drop(['Up','PriceDiff',"Close","high","low","volume","Open"], axis=1)
    #X = df.drop(['Up','timestamp','datetime','PriceDiff',"Close","high","low","volume","Open"], axis=1)
    y = df['PriceDiff']

    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_list = [X_train,X_test]
    data_dict = {}
    j=0
    for X_data in X_list:
        name = f'X_data{2*j+1}_'
        X1 = X_data
        X1_len = len(X1) 
        selected_columns = []
        for k in range(1, days_a+1):
            new_elements = [f'PriceDiff_{k}',f'Up_{k}', f'high{k}', f'low{k}', f'volume{k}']
            #new_elements = [f'PriceDiff_{k}', f'high{k}', f'low{k}', f'volume{k}']
            selected_columns.extend(new_elements)
        selected_data1 = X1[selected_columns].values
        
        data_dict[name] = selected_data1.reshape(X1_len, days_a, 5)
        print(data_dict[name].shape)
        name = f'X_data{2*j+2}_'
        X1 = X1.drop(selected_columns, axis=1)
        data_dict[name] = np.array(X1)
        print(1+i)
        j = j + 1

    
    # モデルの定義
    input1 = Input(shape=(days_a, 5))
    input2 = Input(shape=(8,))

    #conv = Conv1D(32, kernel_size=3, activation='leaky_relu')(input1)
    #pool = MaxPooling1D(pool_size=2)(conv)
    #conv = Conv1D(64, kernel_size=3, activation='leaky_relu')(pool)
    #pool = MaxPooling1D(pool_size=2)(conv)
    flatten = Flatten()(input1)
    concat = Concatenate()([flatten, input2])
    dense = Dense(128, activation='leaky_relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    # モデルのコンパイル
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # モデルのサマリを表示
    model.summary()

    # モデルを訓練する
    # モデルの訓練
    model.fit([data_dict['X_data1_'], data_dict['X_data2_']], y_train, epochs=epochcaa, batch_size=batch_sizeaa)
    with open(f'model6_'+codem+'.pkl', mode='wb') as f:
        pickle.dump(model, f)

    loss = model.evaluate([data_dict['X_data3_'], data_dict['X_data4_']], y_test)
    print('Test Loss:', loss)
    y_pred = model.predict([data_dict['X_data3_'], data_dict['X_data4_']])
    


print(sum)
