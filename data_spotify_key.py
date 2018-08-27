import requests
import json
import base64
import spotipy
import sys
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials


def getauthtoken(key, secret):
    '''
    This returns  \n
    :param key: Spotify App Key as *bytestring!*
    :param secret:  Spotify App Client Secret as *bytestring!*
    :return:
    '''

    preencode = b'%s:%s'
    toencode = preencode % (key, secret)

    code = base64.b64encode(toencode) # Convert <key>:<secret> to b'...' where ... is base64
    code = code.decode("utf-8")
    headers = {'Authorization': f'Basic {code}'}
    data = [('grant_type', 'client_credentials')]
    post = requests.post('https://accounts.spotify.com/api/token', headers=headers, data=data)

    return post.content.decode("utf-8")


def loadsp():
    '''
    This will load the spotipy session for my key and secret from my Spotify developer account.
    :return: Session
    '''

    # Get the authorization token for my key and secret (respectively)l
    tokenout = getauthtoken(b'XXXXXXXXXXXX', b'XXXXXXXXXXX')
    tokenout = json.loads(tokenout)
    token = tokenout['access_token']

    sp = spotipy.client.Spotify(auth=token)

    return tokenout, sp


def getuserauth(scope='playlist-modify-public', id='',
                secret='', use='cred'):

    # When you set up the spotify developer account, you MUST use the below uri as the redirect URI (in settings)
    uri = 'http://localhost/'

    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        print("Usage: %s username" % (sys.argv[0],))
        sys.exit()

    if use == 'oath':

        oauth = spotipy.oauth2.SpotifyOAuth(id, secret, uri, state=None, scope=None,
                                            cache_path=None, proxies=None)
        token = oauth.get_cached_token()

        return token, oauth, spotipy.Spotify(auth=token)

    elif use == 'user':
        token = util.prompt_for_user_token(username, scope, client_id=id,
                                       client_secret=secret, redirect_uri=uri)

        return token, spotipy.Spotify(auth=token)

    elif use == 'cred':

        cred = SpotifyClientCredentials(id, secret)
        token = cred.get_access_token()

        return token, cred, spotipy.Spotify(auth=token)

    else:
        print("Can't get token for ", username)
        return None

# results = sp.search()
# data = pd.read_json(results[...])

# Use sp.album('[albumid]') ... for album info

# Search Tags (separated by a +, no spaces):
# 'album:...'
# 'artist:...'
# 'track:...'
# 'year:...' -- this could be 'year:2010' or 'year:

# Types: 'track', 'album', 'artist', or 'playlist'

