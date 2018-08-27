from time import sleep
import pandas as pd
import numpy as np
from data_spotify_key import *
from ast import literal_eval
import spotipy
from sklearn.model_selection import train_test_split
import os

# PLAN:
# 1: Get 'limit' tracks for each of the 126 genres. (limit <= 100)
# 2: Gather all the information for each track into table form
# 3: Upload this table into BigQuery (transfer script to DataLab, and move to BigQuery)
# 4: Write script to pull specific data from BigQuery for experiment (this is where the classes come in)

# This code assumes we're executing it in the U.S.

token, cred, sp = getuserauth()
limit = 15   # For CoDaS project, you may want (max) 80 tracks for all 126 genres (so, no maxseeds)
dataset = 'music'
projectID = 'coral-theme-198721'
current = True  # Only for the first 5 genres


def alldata(limit, sp=sp, maxseeds=None, must='play'):
    '''
    Collects (virtually) all track and analysis data for `limit` tracks from all 126 genres Spotify offers.
    :param limit: int. max number of tracks you'd like to have from each genre. Remember, the `must` parameter may
    change this. I.e., if only 5/8 tracks from a genre are playable (say), then only 5 will be returned for that genre.
    :param sp: Spotipy. This is the Spotipy session. Use data_spotipy_key.loadsp().
    :param maxseeds: int, (default=None). This is the maximum number of genres you'd like to return data for.
    This function will only take the top `maxseeds` genres, in alphabetical order.
    :param must: str='play30', or 'play'. If 'play30', will only return the tracks with a 30 second preview. If
    'play' will only return tracks that will play the track in the Spotipy app.
    :param minlen: float, (default=20.0). Minimum length of track in seconds.
    :return: pd.DataFrame. The whole data set.
    '''

    seeds = sp.recommendation_genre_seeds()['genres'][:maxseeds]    # if you don't want to use too much data (for testing)

    inds = []

    # First, we get all data

    # Column names for these data: Track Info, Album, Features, and Analysis (meta-data, track, analysis)
    cols_tr = ['genre',
               'title',
               'artists',
               'artists_ids',
               'is_playable',
               'explicit',
               'track_url',
               'popularity',
               'sample_30sec']
    cols_al = ['album',
               'album_url']
    cols_fe = ['danceability',
                 'energy',
                 'speechiness',
                 'acousticness',
                 'instrumentalness',
                 'liveness',
                 'valence',
                 'duration_ms']
    cols_an_m = ['analyzer_version',
                 'platform',
                 'input_process']
    cols_an_t = ['num_samples',
                 'duration',
                 'offset_seconds',
                 'window_seconds',
                 'analysis_sample_rate',
                 'analysis_channels',
                 'end_of_fade_in',
                 'start_of_fade_out',
                 'loudness',
                 'tempo',
                 'tempo_confidence',
                 'time_signature',
                 'time_signature_confidence',
                 'key',
                 'key_confidence',
                 'mode',
                 'mode_confidence',
                 'code_version',
                 'echoprint_version',
                 'synch_version',
                 'rhythm_version']
    cols_an_a = ['bars',
                 'beats',
                 'tatums',
                 'sections',
                 'segments_meta',
                 'segments_pitches',
                 'segments_timbre']

    cols = cols_tr + cols_al + cols_fe + cols_an_m + cols_an_t + cols_an_a
    data = []

    for seed in seeds:

        print(f'Genre: {seed}. That\'s {seeds.index(seed) + 1} / {len(seeds)} genres.')  # updates and debugging

        results = sp.recommendations(seed_genres=[seed], limit=limit, country='US')
        tracks = results['tracks']

        if must == 'play30':
            tracks = [track for track in tracks if isinstance(track['preview_url'], str)]

        elif must == 'play':
            tracks = [track for track in tracks if track['is_playable']]

        limit = min(len(tracks), limit)

        ids = [tracks[i]['id'] for i in range(limit)]

        features = sp.audio_features(ids)

        for i, id in enumerate(ids):

            inds.append(id)

            track = tracks[i]
            features_ = features[i]

            try:
                analysis = sp.audio_analysis(id)

            except:
                X_all = pd.DataFrame(data=data, index=inds, columns=cols)
                return X_all

            data_tr = [seed,
                       track['name'],
                       [artist['name'] for artist in track['artists']],
                       [artist['id'] for artist in track['artists']],
                       track['is_playable'],
                       track['explicit'],
                       track['external_urls']['spotify'],
                       track['popularity'],
                       track['preview_url']]
            data_al = [track['album']['name'],
                       track['album']['external_urls']['spotify']]
            data_fe = [features_[key] for key in cols_fe]
            data_an_m = [analysis['meta'][key] for key in cols_an_m]
            data_an_t = [analysis['track'][key] for key in cols_an_t]

            bars = [[bar['start'], bar['duration'], bar['confidence']] for bar in analysis['bars']]
            beats = [[beat['start'], beat['duration'], beat['confidence']] for beat in analysis['beats']]
            tatums = [[tatum['start'], tatum['duration'], tatum['confidence']] for tatum in analysis['tatums']]

            sectionkeys = ['start',
                             'duration',
                             'confidence',
                             'loudness',
                             'tempo',
                             'tempo_confidence',
                             'key',
                             'key_confidence',
                             'mode',
                             'mode_confidence',
                             'time_signature',
                             'time_signature_confidence']

            sections = [[section[key] for key in sectionkeys] for section in analysis['sections']]

            # These are the titles for the segments_meta data
            segmentkeys = ['start',
                           'duration',
                           'confidence',
                           'loudness_start',
                           'loudness_max_time',
                           'loudness_max']

            # Meta-data for segments (in documentation).
            segments_meta = [[segment[key] for key in segmentkeys] for segment in analysis['segments']]
            segments_pitches = [segment['pitches'] for segment in analysis['segments']]
            segments_timbre = [segment['timbre'] for segment in analysis['segments']]

            data.append(data_tr + data_al + data_fe + data_an_m + data_an_t + [bars] + [beats] + [tatums] + [sections] +
                        [segments_meta] + [segments_pitches] + [segments_timbre])

            print(f'Track {i + 1} / {len(ids)} loaded.')    # Just to give an update

        if maxseeds is None or maxseeds * limit > 500:

            sleep(10)   # For rate limiting

        sp = loadsp()

    X_all = pd.DataFrame(data=data, index=inds, columns=cols)

    # X_all.to_gbq(dataset + '.allitems', projectID)

    return X_all  # print('The data has been added to ' + dataset + ' in ' + projectID + '.')


def trackdata(X_all):

    inds = list(X_all.index)

    # In general, these all use the same audio analysis type (see Spotify API for more info)

    X_info = X_all[['genre',
                    'title',
                    'artists',
                    'artists_ids',
                    'is_playable',
                    'duration',
                    'track_url',
                    'sample_30sec',
                    'album',
                    'album_url']]

    # Get numerical information data (loudness, instrumentalness, etc.)

    X_meta = X_all[['popularity',
                    'danceability',
                    'energy',
                    'speechiness',
                    'acousticness',
                    'instrumentalness',
                    'liveness',
                    'valence',
                    'loudness',
                    'tempo',
                    'tempo_confidence',
                    'time_signature',
                    'time_signature_confidence',
                    'key',
                    'key_confidence',
                    'mode',
                    'mode_confidence',
                    'explicit'
                    ]]
    X_meta_np = np.array(X_meta)

    # Set up numpy arrays for training: bars, beats, tatums, sections, and segments

    minbars = min([len(r) for r in X_all['bars']])
    minbeats = min([len(r) for r in X_all['beats']])
    mintatums = min([len(r) for r in X_all['tatums']])
    minsecs = min([len(r) for r in X_all['sections']])
    minsegs = min([len(r) for r in X_all['segments_meta']])

    X_bars = np.array([r[:minbars] for r in X_all['bars']])
    X_beats = np.array([r[:minbeats] for r in X_all['beats']])
    X_tatums = np.array([r[:mintatums] for r in X_all['tatums']])
    X_secs = np.array([r[:minsecs] for r in X_all['sections']])

    X_bars = X_bars.reshape((X_bars.shape[0], X_bars.shape[1] * X_bars.shape[2]))
    X_beats = X_beats.reshape((X_beats.shape[0], X_beats.shape[1] * X_beats.shape[2]))
    X_tatums = X_tatums.reshape((X_tatums.shape[0], X_tatums.shape[1] * X_tatums.shape[2]))
    X_secs = X_secs.reshape((X_secs.shape[0], X_secs.shape[1] * X_secs.shape[2]))

    # Break up the data so each segment (for each track) contains the metadata, pitches, and timbre (flattened)
    segs_meta = np.array([r[:minsegs] for r in X_all['segments_meta']])
    segs_pitches = np.array([r[:minsegs] for r in X_all['segments_pitches']])
    segs_timbre = np.array([r[:minsegs] for r in X_all['segments_timbre']])

    X_segs = np.concatenate((segs_meta, segs_pitches, segs_timbre), axis=2)
    X_segs = X_segs.reshape((X_segs.shape[0], X_segs.shape[1] * X_segs.shape[2]))

    return inds, X_info, X_all, X_meta, X_meta_np, X_bars, X_beats, X_tatums, X_secs, X_segs


def strtoarr(X_csv):

    for col in ['bars', 'beats', 'tatums', 'sections', 'segments_meta', 'segments_pitches', 'segments_timbre']:

        X_csv[col] = X_csv[col].apply(lambda r: literal_eval(r))

    X_csv.set_index('Unnamed: 0', inplace=True)

    return X_csv


def codas_prep(X_all, playdur=20, start=0, info=('title', 'artists', 'album', 'sample_30sec')):

    musicData = []

    for i in range(X_all.shape[0]):

        if len(X_all.iloc[i]['segments_meta']) <= 1 or len(X_all.iloc[i]['sections']) <= 1:
            continue

        track_segs_1 = np.array(list(X_all.iloc[i]['segments_meta']))
        track_segs_2 = np.array(list(X_all.iloc[i]['segments_pitches']))
        track_segs_3 = np.array(list(X_all.iloc[i]['segments_timbre']))
        track_segs = np.hstack((track_segs_1, track_segs_2, track_segs_3))
        track_secs = np.array(list(X_all.iloc[i]['sections']))

        # We can add these to the data for each track, but we would need to consider weighing the features
        # Right now, we do not include these features (because they are the same for all segments). See 'Add here' below
        track_other = np.array(list(X_all.iloc[i][['acousticness',
                                                   'danceability',
                                                   'energy',
                                                   'instrumentalness',
                                                   'liveness',
                                                   'popularity',
                                                   'speechiness',
                                                   'valence']])).T

        t_segs = track_segs[:, 0]
        t_secs = track_secs[:, 0]

        t_segs_mask = [start <= t_segs[i] < start+playdur or
                       t_segs[i] < start < t_segs[i+1] for i in range(len(t_segs))]
        t_secs_mask = [start <= t_secs[i] < start+playdur or
                       t_secs[i] < start < t_secs[i+1] for i in range(len(t_secs))]

        track_segs = track_segs[t_segs_mask]
        track_secs = track_secs[t_secs_mask]

        newsegs = np.empty((0, track_segs.shape[1] + track_secs.shape[1] - 1))
        curseg = 0

        for cursec in range(len(track_secs) - 1):

            while curseg < len(track_segs) and track_segs[curseg][0] < track_secs[cursec + 1][0]:
                newsegs = np.vstack((newsegs,
                                     np.hstack((track_segs[curseg], track_secs[cursec][1:]))    # Add here ...
                                     ))
                curseg += 1

        while curseg < len(track_segs):
            newsegs = np.vstack((newsegs,
                                 np.hstack((track_segs[curseg], track_secs[-1][1:]))        # Add here ...
                                 ))
            curseg += 1

        trackData = dict()
        trackData['info'] = dict(X_all.iloc[i][[x for x in info]])
        trackData['genre'] = X_all.iloc[i]['genre']
        trackData['audio'] = X_all.iloc[i]['track_url']
        trackData['stFeatures'] = newsegs.T

        musicData.append(trackData)

    return musicData


def smalldata(limit, sp=sp, seedstart=0, maxseeds=None):

    seeds = sp.recommendation_genre_seeds()['genres'][seedstart:maxseeds]
    data = []
    cols_tr = ['genre',
               'title',
               'artists',
               'artists_ids',
               'is_playable',
               'explicit',
               'track_url',
               'popularity',
               'sample_30sec',
               'album']

    for ct, seed in enumerate(seeds):

        print(f'Genre: {seed}. That\'s {seeds.index(seed) + 1} / {len(seeds)} genres.')  # updates and debugging

        try:
            results = sp.recommendations(seed_genres=[seed], limit=limit, country='US')

        except spotipy.client.SpotifyException:
            sleep(61)
            sp = loadsp()[1]
            results = sp.recommendations(seed_genres=[seed], limit=limit, country='US')

        tracks = results['tracks']
        tracks = [track for track in tracks if isinstance(track['preview_url'], str)]

        limit = min(len(tracks), limit)

        for i in range(limit):

            track = tracks[i]

            data_tr = [seed,
                       track['name'],
                       [artist['name'] for artist in track['artists']],
                       [artist['id'] for artist in track['artists']],
                       track['is_playable'],
                       track['explicit'],
                       track['external_urls']['spotify'],
                       track['popularity'],
                       track['preview_url'],
                       track['album']['name']]

            data.append(data_tr)

    X_small = pd.DataFrame(data=data, columns=cols_tr)

    return X_small

# X_csv = pd.read_csv('/'.join([os.getcwd(), 'X_all.csv']))
# X_all = strtoarr(X_csv)
# tracks, X_info, X_all, X_meta, X_meta_np, X_bars, X_beats, X_tatums, X_secs, X_segs = trackdata(X_all)

# sess_leon = getuserauth()

# Set test_size to 0.01, so we need a user (or sim) to pre-label 0.01 * 1260 = 13 songs
# X_train, X_test, tracks_train, tracks_test = train_test_split(X_secs, tracks, test_size=0.01, random_state=42)