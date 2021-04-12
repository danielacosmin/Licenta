import spotipy
import spotipy.util as util
import pandas as pd
import pprint

CLIENT_ID = "d3184248655247a59b76ce5a0820c3fc"
CLIENT_SECRET = "31f613eabd1347dabba3bc3add64236d"


token = spotipy.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
cache_token = token.get_access_token(as_dict=False)
sp = spotipy.Spotify(cache_token)

playlist_dict = {
    
    #happy playlists
    # "Happy one song" :("Daniela Cosmin","1Rs3zqk0PCTjD26iUNt8k1"),
     "Happy Hits!" :("spotify", "37i9dQZF1DXdPec7aLTmlC"),
    "Happy Songs(80s-2020s)":("indiemono", "4AnAUkQNrLKlJCInZGSXRO"),
    "Happy Tunes":("spotify",'37i9dQZF1DX9u7XXOp0l5L'),
    "Wake up Happy":("spotify",'37i9dQZF1DX0UrRvztWcAU'),

    # # # # #sad playlists
    "Sad songs":("spotify", "37i9dQZF1DX7qK8ma5wgG1"),
    "Sad music for crying":("JudasVargas","44tRfteJJzAmUONSiA56bQ"),
    "Sad beats -electronic melancholy":("spotify","37i9dQZF1DWVrtsSlLKzro"),
    "Sad 00's":("spotify", "37i9dQZF1DXa39zZwdBPSN"),
    "Sad 10's":("spotify", "37i9dQZF1DX8Vz2ROLXhTT"),
    "Sad songs 2021":("heartbreak songs","3c0Nv5CY6TIaRszlTZbUFk"),

    # # # # #calm/relaxing songs
    "Chill Hits":("spotify","37i9dQZF1DX4WYpdgoIcn6"),
     "Chill Vibes":("spotify","37i9dQZF1DX889U0CL85jj"),
    "Chill Tracks":("spotify","37i9dQZF1DX6VdMW310YC7"),

    # # # # #energetic songs
     "Energy Booster:Dance":("spotify","37i9dQZF1DX35X4JNyBWtb"),
     "Energy Booster:Rock":("spotify","37i9dQZF1DWZVAVMhIe3pV"),
     "Energy Booster:HipHop":("spotify","37i9dQZF1DWZixSclZdoFE"),
    "Energy Booster:Pop":("spotify","37i9dQZF1DX0vHZ8elq0UK"),
    "Energy Boost":("Jeffry Harrison","0Vjhah37el0Aq5yoRaujBz"),
     "Energy EDM/Rave":("Logan Weeb","2AMSyBkXMTXMhIs7Co5MPq")
     


}
#Custom method for extracting playlist details - more than 100 per playlisy
def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    # pprint.pprint(tracks)
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    return tracks
def analyze_playlist(creator, playlist_id):
    
    # Create empty dataframe
    # playlist_features_list = ["artist", "album", "track_name", "track_id", 
    #                          "danceability", "energy", "key", "loudness", "mode", "speechiness",
    #                          "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]
    playlist_features_list = ["name", "album", "artist", "id", "release_date","popularity","length",
                                "danceability","acousticness","energy","instrumentalness", "liveness",
                                "valence", "loudness", "speechiness", "tempo", "key", "time_signature"]
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Create empty dict
    playlist_features = {}
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    playlist = get_playlist_tracks(creator, playlist_id)
    for track in playlist:
        # Get metadata
        playlist_features["name"] = track["track"]["name"]
        # print(playlist_features["name"])
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["id"] = track["track"]["id"]
        playlist_features["release_date"] = track["track"]["album"]["release_date"]
        playlist_features["popularity"]=track["track"]["popularity"]
        playlist_features["length"]=track["track"]["duration_ms"]
        # print(playlist_features["length"])
        # Get audio features
        audio_features = sp.audio_features(playlist_features["id"])[0]
        for feature in playlist_features_list[7:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
        # print(playlist_df)
    return playlist_df

def analyze_playlist_dict(playlist_dict):
    
    # Loop through every playlist in the dict and analyze it
    for i, (key, val) in enumerate(playlist_dict.items()):
        playlist_df = analyze_playlist(*val)
        # Add a playlist column so that we can see which playlist a track belongs too
        playlist_df["mood"] = key


        # Create or concat df
        if i == 0:
            playlist_dict_df = playlist_df
        else:
            playlist_dict_df = pd.concat([playlist_dict_df, playlist_df], ignore_index = True)

    #Place the mood for each playlist
    playlist_name=playlist_dict_df["mood"]
    h = playlist_name[playlist_name.str.contains('happy|Happy',na=False)]
    s = playlist_name[playlist_name.str.contains('sad|Sad',na=False)]
    c = playlist_name[playlist_name.str.contains('Chill',na=False)]
    e = playlist_name[playlist_name.str.contains('Energy|Energetic',na=False)]
    # playlist_dict_df["mood"] = ""
    playlist_dict_df["mood"].loc[playlist_name.isin(h)] = "Happy"
    playlist_dict_df["mood"].loc[playlist_name.isin(s)] = "Sad"
    playlist_dict_df["mood"].loc[playlist_name.isin(e)] = "Energetic"
    playlist_dict_df["mood"].loc[playlist_name.isin(c)] = "Calm"
    # print(playlist_dict_df)
    return playlist_dict_df

multiple_playlist_df = analyze_playlist_dict(playlist_dict)
# multiple_playlist_df.drop_duplicates(subset='id')
# multiple_playlist_df.head()

print(multiple_playlist_df["mood"].value_counts())
# print(multiple_playlist_df["mood"].value_counts())
multiple_playlist_df.to_csv("dataset2.csv", index = False)



