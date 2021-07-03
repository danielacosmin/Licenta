#!/usr/bin/python3
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
from numpy.lib.index_tricks import diag_indices
from pandas.io.parsers import read_csv
from seaborn.distributions import ecdfplot
from sklearn.pipeline import Pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, url_for, session, request, redirect, render_template
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn import metrics
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import artist, pyplot
from plotly import graph_objs as go
import plotly
from sklearn.decomposition import PCA
import json
from bs4 import BeautifulSoup as bs
import io
import zlib
import pprint
import plotly.express as px
import scipy
import matplotlib
matplotlib.use('Agg')
# App config
app = Flask(__name__)
app.secret_key = 'widuhw77t72yg'
app.config['SESSION_COOKIE_NAME'] = 'Dani cookie'
recomm_sad=[]
recomm_happy=[]
recomm_calm=[]
recomm_energetic=[]


@app.route('/')
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/choose-mood', methods=['GET','POST'])
def choose_mood():
    
    return render_template("choose_mood.html",data=[{'name':'Happy'}, {'name':'Sad'}, {'name':'Energetic'},{'name':'Calm'}])


@app.route('/action-mood', methods=['GET','POST'])
def action_mood():
    if request.method == 'POST':
        
            select = request.form.get('comp_select')
            print(select)
            if(select == 'Sad'):
                return redirect('/my-sad-songs')
            if(select == 'Happy'):
                return redirect('/my-happy-songs')
            if(select == 'Energetic'):
                return redirect('/my-energetic-songs')
            if(select == 'Calm'):
                return redirect('/my-calm-songs')
            
@app.route('/my-happy-songs', methods=['GET','POST'])
def my_happy_songs():
    

    df_happy = pd.read_csv('happy_songs.csv')
    dict_happy = df_happy.to_dict()

    res = []
    artists =[]
    songs =[]
    ids =[]
    photos =[]
    previews=[]
    for k, v in dict_happy.items():
        if k == 'artist':
            for value in v.items():
                artists.append(value[1])
        if k == 'name':
            for value in v.items():
                songs.append(value[1])
        if k == 'id':
            for value in v.items():
                ids.append(value[1])
        if k == 'images':
            for value in v.items():
                photos.append(value[1])
        if k == 'preview':
            for value in v.items():
                previews.append(value[1])

    res =list(zip(songs,artists,ids,photos,previews))
    
    if request.method == "POST":
         recomm_happy.append(request.form.get("submit_song"))

    print(recomm_happy)
    # urls = list(df_sad['images'])
    return render_template("happy_songs.html",data=res,enumerate=enumerate,recomm_happy=recomm_happy)

@app.route('/my-energetic-songs', methods=['GET','POST'])
def my_energetic_songs():
    

    df_energetic = pd.read_csv('energetic_songs.csv')
    dict_energetic = df_energetic.to_dict()

    res = []
    artists =[]
    songs =[]
    ids =[]
    photos =[]
    previews=[]
    for k, v in dict_energetic.items():
        if k == 'artist':
            for value in v.items():
                artists.append(value[1])
        if k == 'name':
            for value in v.items():
                songs.append(value[1])
        if k == 'id':
            for value in v.items():
                ids.append(value[1])
        if k == 'images':
            for value in v.items():
                photos.append(value[1])
        if k == 'preview':
            for value in v.items():
                previews.append(value[1])

    res =list(zip(songs,artists,ids,photos,previews))
    
    if request.method == "POST":
         recomm_energetic.append(request.form.get("submit_song"))

    print(recomm_energetic)
    # urls = list(df_sad['images'])
    return render_template("energetic_songs.html",data=res,enumerate=enumerate,recomm_energetic=recomm_energetic)

@app.route('/my-sad-songs', methods=['GET','POST'])
def my_sad_songs():
    

    df_sad = pd.read_csv('sad_songs.csv')
    dict_sad = df_sad.to_dict()

    res = []
    artists =[]
    songs =[]
    ids =[]
    photos =[]
    previews=[]
    for k, v in dict_sad.items():
        if k == 'artist':
            for value in v.items():
                artists.append(value[1])
        if k == 'name':
            for value in v.items():
                songs.append(value[1])
        if k == 'id':
            for value in v.items():
                ids.append(value[1])
        if k == 'images':
            for value in v.items():
                photos.append(value[1])
        if k == 'preview':
            for value in v.items():
                previews.append(value[1])

    res =list(zip(songs,artists,ids,photos,previews))
    
    if request.method == "POST":
         recomm_sad.append(request.form.get("submit_song"))

    print(recomm_sad)
    # urls = list(df_sad['images'])
    return render_template("sad_songs.html",data=res,enumerate=enumerate,recomm_sad=recomm_sad)

@app.route('/my-calm-songs', methods=['GET','POST'])
def my_calm_songs():
    

    df_calm = pd.read_csv('calm_songs.csv')
    dict_calm = df_calm.to_dict()

    res = []
    artists =[]
    songs =[]
    ids =[]
    photos =[]
    previews=[]
    for k, v in dict_calm.items():
        if k == 'artist':
            for value in v.items():
                artists.append(value[1])
        if k == 'name':
            for value in v.items():
                songs.append(value[1])
        if k == 'id':
            for value in v.items():
                ids.append(value[1])
        if k == 'images':
            for value in v.items():
                photos.append(value[1])
        if k == 'preview':
            for value in v.items():
                previews.append(value[1])

    res =list(zip(songs,artists,ids,photos,previews))
    
    if request.method == "POST":
         recomm_calm.append(request.form.get("submit_song"))

    print(res)
    # urls = list(df_sad['images'])
    return render_template("calm_songs.html",data=res,enumerate=enumerate,recomm_calm=recomm_calm)

@app.route('/recommend-sad',methods=['GET','POST'])
def recommend_sad():

    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')


    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    df = read_csv('D1.csv')
    encoder = LabelEncoder()
    Y = df['mood']
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    col_features = df.columns[6:-3]
    X2 = np.array(df[col_features])
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)

    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                            batch_size=50,verbose=0))])
    pip = pip.fit(X2,encoded_y)

    images =[]
    previews =[]
    names=[]
    artists =[]
    names =[]
    popularities =[]
    dances =[]
    accs =[]
    energies =[]
    instrs =[]
    lives =[]
    valences =[]
    loudnesses =[]
    speech=[]
    tempos=[]

    if request.method == 'POST':
        print(recomm_energetic)
        results_energetic = sp.recommendations(seed_tracks=recomm_sad,limit=20)
    for value in results_energetic['tracks']:
    
        preview = value['preview_url']
        preds = get_songs_features(value['id'])
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        results = pip.predict(preds_features)
        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        #Add only if the recommended song is sad also
        if mood[0] == 'Sad':

            names.append(preds[0][0])
            artists.append(preds[0][2])
            previews.append(preview)
            images.append(value['album']['images'][0]['url'])
            popularities.append(preds[0][5])
            dances.append(preds[0][7])
            accs.append(preds[0][8])
            energies.append(preds[0][9])
            instrs.append(preds[0][10])
            lives.append(preds[0][11])
            valences.append(preds[0][12])
            loudnesses.append(preds[0][13])
            speech.append(preds[0][14])
            tempos.append(preds[0][15]) 

    final_recomm =list(zip(names,artists,images,previews,popularities,dances,accs,energies,instrs,lives,valences,loudnesses,speech,tempos))
    print(final_recomm)
    
    return render_template("recommend_page.html",mood="Sad",r=final_recomm,enumerate=enumerate)

@app.route('/recommend-happy',methods=['GET','POST'])
def recommend_happy():
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    df = read_csv('D1.csv')
    encoder = LabelEncoder()
    Y = df['mood']
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    col_features = df.columns[6:-3]
    X2 = np.array(df[col_features])
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                            batch_size=50,verbose=0))])
    pip = pip.fit(X2,encoded_y)

    
    images =[]
    previews =[]
    names=[]
    artists =[]
    names =[]
    popularities =[]
    dances =[]
    accs =[]
    energies =[]
    instrs =[]
    lives =[]
    valences =[]
    loudnesses =[]
    speech=[]
    tempos=[]

    if request.method == 'POST':
        
        results_energetic = sp.recommendations(seed_tracks=recomm_happy,limit=20)
    for value in results_energetic['tracks']:
    
        preview = value['preview_url']
        preds = get_songs_features(value['id'])
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        results = pip.predict(preds_features)
        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        #Add only if the recommended song is sad also
        if mood[0] == 'Happy':

            names.append(preds[0][0])
            artists.append(preds[0][2])
            previews.append(preview)
            images.append(value['album']['images'][0]['url'])
            popularities.append(preds[0][5])
            dances.append(preds[0][7])
            accs.append(preds[0][8])
            energies.append(preds[0][9])
            instrs.append(preds[0][10])
            lives.append(preds[0][11])
            valences.append(preds[0][12])
            loudnesses.append(preds[0][13])
            speech.append(preds[0][14])
            tempos.append(preds[0][15]) 

    final_recomm =list(zip(names,artists,images,previews,popularities,dances,accs,energies,instrs,lives,valences,loudnesses,speech,tempos))
    print(final_recomm)
    return render_template("recommend_page.html",mood="Happy",r=final_recomm,enumerate=enumerate)

@app.route('/recommend-energetic',methods=['GET','POST'])
def recommend_energetic():
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    df = read_csv('D1.csv')
    encoder = LabelEncoder()
    Y = df['mood']
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    col_features = df.columns[6:-3]
    X2 = np.array(df[col_features])
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                            batch_size=50,verbose=0))])
    pip = pip.fit(X2,encoded_y)

    images =[]
    previews =[]
    names=[]
    artists =[]
    names =[]
    popularities =[]
    dances =[]
    accs =[]
    energies =[]
    instrs =[]
    lives =[]
    valences =[]
    loudnesses =[]
    speech=[]
    tempos=[]

    if request.method == 'POST':
        print(recomm_energetic)
        results_energetic = sp.recommendations(seed_tracks=recomm_energetic,limit=20)
    for value in results_energetic['tracks']:
    
        preview = value['preview_url']
        preds = get_songs_features(value['id'])
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        results = pip.predict(preds_features)
        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        #Add only if the recommended song is sad also
        if mood[0] == 'Energetic':

            names.append(preds[0][0])
            artists.append(preds[0][2])
            previews.append(preview)
            images.append(value['album']['images'][0]['url'])
            popularities.append(preds[0][5])
            dances.append(preds[0][7])
            accs.append(preds[0][8])
            energies.append(preds[0][9])
            instrs.append(preds[0][10])
            lives.append(preds[0][11])
            valences.append(preds[0][12])
            loudnesses.append(preds[0][13])
            speech.append(preds[0][14])
            tempos.append(preds[0][15]) 

    final_recomm =list(zip(names,artists,images,previews,popularities,dances,accs,energies,instrs,lives,valences,loudnesses,speech,tempos))
    print(final_recomm)
    return render_template("recommend_page.html",mood="Energetic",r=final_recomm,enumerate=enumerate)

@app.route('/recommend-calm',methods=['GET','POST'])
def recommend_calm():
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    df = read_csv('D1.csv')
    encoder = LabelEncoder()
    Y = df['mood']
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    col_features = df.columns[6:-3]
    X2 = np.array(df[col_features])
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                            batch_size=50,verbose=0))])
    pip = pip.fit(X2,encoded_y)

    
    images =[]
    previews =[]
    names=[]
    artists =[]
    names =[]
    popularities =[]
    dances =[]
    accs =[]
    energies =[]
    instrs =[]
    lives =[]
    valences =[]
    loudnesses =[]
    speech=[]
    tempos=[]

    if request.method == 'POST':
      
        results_energetic = sp.recommendations(seed_tracks=recomm_calm,limit=50)
    for value in results_energetic['tracks']:
    
        preview = value['preview_url']
        preds = get_songs_features(value['id'])
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        results = pip.predict(preds_features)
        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        #Add only if the recommended song is sad also
        if mood[0] == 'Calm':

            names.append(preds[0][0])
            artists.append(preds[0][2])
            previews.append(preview)
            images.append(value['album']['images'][0]['url'])
            popularities.append(preds[0][5])
            dances.append(preds[0][7])
            accs.append(preds[0][8])
            energies.append(preds[0][9])
            instrs.append(preds[0][10])
            lives.append(preds[0][11])
            valences.append(preds[0][12])
            loudnesses.append(preds[0][13])
            speech.append(preds[0][14])
            tempos.append(preds[0][15]) 

    final_recomm =list(zip(names,artists,images,previews,popularities,dances,accs,energies,instrs,lives,valences,loudnesses,speech,tempos))
    print(final_recomm)
    return render_template("recommend_page.html",mood="Calm",r=final_recomm,enumerate=enumerate)


@app.route('/prediction', methods=['GET','POST'])
def prediction():

    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    listen = ''
    pred_song =''
    pred_artist = ''
    pred_mood = ''
    image_url=''

    similar_songs = []
    similar_artists =[]
    similars=[]
    if request.method == 'POST':
        form_data  = request.form
        data = dict(form_data)
        df = read_csv('D1.csv')
        encoder = LabelEncoder()
        Y = df['mood']
        encoder.fit(Y)
        encoded_y = encoder.transform(Y)
        col_features = df.columns[6:-3]
        X2 = np.array(df[col_features])
        target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
        #Join the model and the scaler in a Pipeline
        pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                                batch_size=50,verbose=0))])                                                             
        #Fit the Pipeline
        pip = pip.fit(X2,encoded_y)

        song_name = data['song']
        artist_name = data['artist']

        query = "artist:%{artist_name} track:%{song_name}".format(artist_name = artist_name, song_name=song_name)
        response = sp.search(q=query, type="track", limit=1)

        id = response['tracks']['items'][0]['id']
        listen = response['tracks']['items'][0]['preview_url']
        #Obtain the features of the song
        preds = get_songs_features(id)
        #Pre-process the features to input the Model
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        #Predict the features of the song
        results = pip.predict(preds_features)

        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        song_name = preds[0][0]
        artist_name = preds[0][2]

        # prediction =  "Artist: {artist_name} track:{song_name} is a {mood} song".
        # format(artist_name = artist_name, song_name=song_name, mood = mood[0])
        pred_artist = "{artist_name}".format(artist_name=artist_name)
        pred_song = "{song_name}".format(song_name=song_name)
        pred_mood = mood[0]
        image_url = response['tracks']['items'][0]['album']['images'][1]['url']

        #Calculate similar songs from the dataset
        df_song = pd.DataFrame(data=preds,columns=preds[1])
        df_song = df_song[:-1]
        df_source = df.iloc[:, :-1]
        ary = scipy.spatial.distance.cdist(df.loc[:,'danceability':'tempo'], df_song.loc[:,'danceability':'tempo'], metric='euclidean')

        df_res = pd.DataFrame()
        closest = np.array(sorted(ary)[0:5])
        for dist in closest:
            df_res = df_res.append(df_source[ary ==dist.min()])

        similar_songs = df_res["name"].values
        similar_artists = df_res["artist"].values
        similars = zip(similar_songs,similar_artists)

    return render_template("predict.html", similars = similars, listen = listen,image_url=image_url ,pred_artist = pred_artist, pred_song = pred_song,pred_mood = pred_mood)

@app.route('/stats', methods = ['GET','POST'])  
def success():  
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))

    if request.method == 'POST':  
        
        f = request.files['file']  
        f.save(f.filename)  
        
        #Begin ML part
        df = pd.read_csv(f.filename)
        score1 = randomf_model(df)
        score2, X2, encoded_y,target = neural_model(df)

       
        get_all_tracks()
        chosen = ["energy", "liveness", "tempo", "valence", "loudness", "speechiness", "acousticness", "danceability", "instrumentalness"]
        text1 = df["artist"] + " - " + df["name"] + "(" + df["release_date"] + ")" + "Mood : " + df["mood"]
        text2 = text1.values

        X = df[chosen].values

        Y = df['mood']
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_y = encoder.transform(Y)
        # y = df["danceability"].values
        print(encoded_y)

        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        pca = PCA(n_components=3)
        pca.fit(X)

        X = pca.transform(X)

        trace = go.Scatter3d(
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
            text=text2,
            mode="markers",
            marker=dict(
                size=4,
                color=encoded_y,
                colorscale='tropic'
            )
        )
        
        data = [trace]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)


        # chosen = ["danceability","valence"]
        # text1 = df["artist"] + " - " + data_frame["song_title"]
        # text1 = df["artist"] + " - " + df["name"] + "(" + df["release_date"] + ")" + "Mood : " + df["mood"]
        # text2 = text1.values

        # X = data_frame.drop(droppable, axis=1).values
        # X = df[chosen].values
        # y = df["valence"].values

        # min_max_scaler = MinMaxScaler()
        # X = min_max_scaler.fit_transform(X)

        # pca = PCA(n_components=2)
        # pca.fit(X)

        # X = pca.transform(X)

        # fig = {
        #     "data": [
        #         {
        #             "x": X[:, 0],
        #             "y": X[:, 1],
        #             "text": text2,
        #             "mode": "markers",
        #             "marker": dict(size=8, color=encoded_y),
                    
        #         }        
        #     ],
        #     "layout": {
        #         "xaxis": {"title": "How hard is this to dance to?"},
        #         "yaxis": {"title": "Valence of a song"}
                
        #     }
        # }
        fig = px.scatter(df, x="acousticness", y="energy", color="mood",
                  hover_data=["name","artist","release_date"])
                  
        fig.update_layout(
            title={
                'text': "Energy and Accousticness importance in classification",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        

        return render_template("stats.html", name = f.filename, score2=score2,score1=score1,graphJSON=graphJSON, graphJSON2=graphJSON2)
        

@app.route('/home')
def home():

    return render_template("home.html")


@app.route('/authorize')
def authorize():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return redirect("/home")



@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')

def get_all_tracks():
    
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    df = read_csv('D1.csv')
    encoder = LabelEncoder()
    Y = df['mood']
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    col_features = df.columns[6:-3]
    X2 = np.array(df[col_features])
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=150,
                                                                            batch_size=50,verbose=0))])
    pip = pip.fit(X2,encoded_y)

    ids = []
    results = []

    names =[]
    artists =[]
    moods = []
    albums = []
    release_dates = []
    popularities =[]
    lengths =[]
    dances =[]
    accs =[]
    energies =[]
    instrs =[]
    lives =[]
    valences =[]
    loudnesses =[]
    speech=[]
    tempos=[]
    images=[]
    previews=[]
    iter = 0
    # while True:
    #     offset = iter * 50
    #     iter += 1
    curGroup = sp.current_user_saved_tracks(limit=50, offset=0)['items']
    for idx, item in enumerate(curGroup):
        track = item['track']
        print(track)
        # val = track['name'] + " - " + track['artists'][0]['name']
        val = track['id']
        image = track['album']['images'][0]['url']
        preview = track['preview_url']
        ids.append(val)
        #Obtain the features of the song
        preds = get_songs_features(val)
        #Pre-process the features to input the Model
        preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
        #Predict the features of the song
        results = pip.predict(preds_features)
        #Calculated mood
        mood = np.array(target['mood'][target['encode']==int(results)])
        moods.append(mood[0])

        name_song = preds[0][0]
        names.append(name_song)
        
        artist = preds[0][2]
        artists.append(artist)

        albums.append(preds[0][1])
        release_dates.append(preds[0][4])
        popularities.append(preds[0][5])
        lengths.append(preds[0][6])
        dances.append(preds[0][7])
        accs.append(preds[0][8])
        energies.append(preds[0][9])
        instrs.append(preds[0][10])
        lives.append(preds[0][11])
        valences.append(preds[0][12])
        loudnesses.append(preds[0][13])
        speech.append(preds[0][14])
        tempos.append(preds[0][15]) 
        images.append(image)
        previews.append(preview)
        # if (len(curGroup) < 50):
        #     break

    df = pd.DataFrame()
    df['name'] = names
    df['album'] = albums
    df['artist'] = artists
    df['id'] = ids
    df['release_date'] = release_dates
    df['popularity'] = popularities
    df['duration_ms'] = lengths
    df['danceability'] = dances
    df['acousticness'] = accs
    df['energy'] = energies
    df['instrumentalness'] = instrs
    df['liveness'] = lives
    df['valence'] = valences
    df['loudness'] = loudnesses
    df['speechiness'] = speech
    df['tempo'] = tempos
    df['mood'] = moods 
    df['images'] = images
    df['preview'] = previews
    df.to_csv('songs.csv', index=False)

    df_sad = df[(df.mood == "Sad")]
    df_sad.to_csv('sad_songs.csv', index=False)

    df_happy = df[(df.mood == "Happy")]
    df_happy.to_csv('happy_songs.csv', index=False)

    df_calm = df[(df.mood == "Calm")]
    df_calm.to_csv('calm_songs.csv', index=False)

    df_energetic = df[(df.mood == "Energetic")]
    df_energetic.to_csv('energetic_songs.csv', index=False)

    return "done"

def randomf_model(df):
 
    col_features = df.columns[6:-3]
    X = MinMaxScaler().fit_transform(df[col_features])
    encoded_y = pd.get_dummies(df.mood)
    X_train,X_test,y_train,y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
    
    clf=RandomForestClassifier(bootstrap=False, criterion='gini',
                max_depth=90, min_samples_leaf=2, n_estimators=185, 
                n_jobs=-1, verbose=0)

    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    #Calculate accuracy
    plt.clf()
    score = metrics.accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred,output_dict=True,zero_division="warn")
    ax = plt.axes()
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True,ax=ax)
    ax.set_title('Classification Report Random Forest model')
    pyplot.savefig("static/report-randomf-small.png",bbox_inches='tight')
    plt.clf()

    #Calculate confusion matrix
    cm = confusion_matrix(y_test.values.argmax(axis=1),y_pred.argmax(axis=1))
    fig, ax = plt.subplots()
    sns.heatmap(cm,annot=True,ax=ax,fmt='d')
    

    labels = ['Calm','Energetic','Happy','Sad']

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix Random Forest model')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    pyplot.savefig("static/matrix-randomf-small.png")
    plt.clf()
    return score

def neural_model(df):

    col_features = df.columns[6:-3]
    X= MinMaxScaler().fit_transform(df[col_features])
    X2 = np.array(df[col_features])
    Y = df['mood']

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)

    X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)
    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)

    estimator = KerasClassifier(build_fn=base_model,epochs=125,batch_size=50,verbose=0)
    estimator.fit(X_train,Y_train)
    y_preds = estimator.predict(X_test)


    # kfold = KFold(n_splits=10,shuffle=True)
    # results = cross_val_score(estimator,X,encoded_y,cv=kfold)

    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))
    #Classification report
    plt.clf()
    cr = classification_report(Y_test, y_preds,output_dict=True,zero_division="warn")
    ax = plt.axes()
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True,ax=ax)
    ax.set_title('Classification Report Neural Network model')
    pyplot.savefig("static/report-neural-small.png", bbox_inches='tight')
    plt.clf()

    #Calculate confusion matrix
    cm = confusion_matrix(Y_test,y_preds)

    ax = plt.subplot()
    sns.heatmap(cm,annot=True,ax=ax)

    labels = target['mood']
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix Neural Network Model')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    pyplot.savefig("static/matrix-neural-small.png")
    plt.clf()
    #Calculate accuracy
    score = accuracy_score(Y_test,y_preds)
    return score, X2, encoded_y,target


def base_model():

    model = Sequential()
    model.add(Dense(8,input_dim=10,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='softmax'))
  
    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
   
    return model

# Checks to see if token is valid and gets a new token if not
def get_token():
    token_valid = False
    token_info = session.get("token_info", {})

    # Checking if the session already has a token stored
    if not (session.get('token_info', False)):
        token_valid = False
        return token_info, token_valid

    # Checking if token has expired
    now = int(time.time())
    is_token_expired = session.get('token_info').get('expires_at') - now < 60

    # Refreshing token if it has expired
    if (is_token_expired):
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(session.get('token_info').get('refresh_token'))

    token_valid = True
    return token_info, token_valid

def get_songs_features(ids):

    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')

    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
            energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature']
    return track,columns

def create_spotify_oauth():
    return SpotifyOAuth(
            client_id="d3184248655247a59b76ce5a0820c3fc",
            client_secret="31f613eabd1347dabba3bc3add64236d",
            redirect_uri=url_for('authorize', _external=True),
            scope="user-library-read",
            show_dialog=True)

if __name__ == "__main__":
    app.run(debug=True)