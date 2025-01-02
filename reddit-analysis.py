import json
import os.path
import time

from convokit import Corpus, download
from lingua import Language, LanguageDetectorBuilder
from collections import defaultdict, Counter

from lingua.lingua import IsoCode639_1

l2_country_names = {
    'The Netherlands': 'nl',
    'Germany': 'de',
    'Greece': 'gr',
    'Estonia': 'ee',
    'Catalonia': 'es',
    'Belgium': 'be',
    'Malta': 'mt',
    'Croatia': 'cr',
    'Italy': 'it',
    'Sleswig-Holsteen': 'de',
    'Slovenia': 'sl',
    'Lombardy': 'it',
    'Bulgaria': 'bg',
    'Romania': 'ro',
    'Portugal': 'pt',
    'Austria': 'at',
    'Spain': 'es',
    'Denmark': 'dk',
    'Cyprus': 'cy',
    'Sweden': 'se',
    'Finland': 'fi',
    'Rumania': 'ro',
    'Latvia': 'lt',
    'Serbia': 'rs',
    'France': 'fr',
    'Switzerland': "ch"
}

l1_country_names = {
    'England',
    'European Union',
    'United States',
    'United States of America',
    'Scotland',
    'United Kingdom',
    'Ireland',
    'Canada'
}

def extract_context(utt, v):
    pos = utt.text.find(v)
    start = max(pos-20, 0)
    end = min(pos+len(v)+20, len(utt.text))
    return utt.text[start:end].replace('\n', ' ')

def load_conversation_data(subreddit_name):
    result = {}
    seen_unknown_countries = set()
    with open("/Users/yole/.convokit/downloads/subreddit-" + subreddit_name + "/conversations.json") as f:
        data = json.load(f)
        for k, v in data.items():
            flair_text = v['author_flair_text']
            if flair_text:
                if flair_text in l2_country_names:
                    result[k] = l2_country_names[flair_text]
                elif flair_text not in l1_country_names and flair_text not in seen_unknown_countries:
                    print("Unknown country " + flair_text)
                    seen_unknown_countries.add(flair_text)
    print("Found {0} L2 conversations".format(len(result)))
    return result

class RedditSpeaker:
    def __init__(self, id):
        self.id = id

class RedditUtterance:
    def __init__(self, data):
        self.id = data['id']
        self.text = data['text']
        self.speaker = RedditSpeaker(data['user'])
        self.meta = data['meta']

def create_obj(data):
    if 'text' in data:
        return RedditUtterance(data)
    return data

def analyze_subreddit(subreddit, language, variants, flair_based_l2=False):
    utterances = [[] for v in variants]
    if language:
        lg = Language.from_iso_code_639_1(getattr(IsoCode639_1, language.upper()))
        detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, lg).build()
    else:
        detector = None
        if flair_based_l2: language = 'eu'

    speakers = defaultdict(lambda: {'en': False, language: False})
    token_count = Counter()
    utterance_count = Counter()
    exclude_id = set()

    data_path = os.path.expanduser("~/.convokit/downloads/subreddit-" + subreddit)
    if os.path.exists(data_path):
        def iter_utterances():
            f = open(data_path + '/utterances.jsonl')
            return (json.loads(s, object_hook=create_obj) for s in f.readlines())

        progress_step = 5000
        total_count = -1
    else:
        corpus = Corpus(filename=download('subreddit-' + subreddit))
        all_utterances = corpus.utterances.values()
        iter_utterances = lambda: all_utterances
        print(f"Processing {len(all_utterances)} utterances")
        progress_step = len(all_utterances) // 20
        total_count = len(all_utterances)

    start_time = time.time()
    processed_count = 0
    for utt in iter_utterances():
        processed_count += 1
        if processed_count % progress_step == 0:
            current_time = time.time()
            percent =round(processed_count / (total_count // 100), 1) if total_count > 0 else 0
            print(f"{percent}% processed; elapsed time {round(current_time - start_time, 2)} sec; EN {utterance_count['en']}; local {utterance_count[language]}")

        if len(utt.text) < len(variants[0]) or utt.text == '[removed]' or utt.text == '[deleted]':
            continue

        speaker_id = utt.speaker.id
        speaker = speakers[speaker_id]

        if detector:
            lg = detector.detect_language_of(utt.text)
            if not lg:
                exclude_id.add(utt.id)
                print("No language for " + utt.text)
                continue
            lang = lg.iso_code_639_1.name.lower()
        elif flair_based_l2:
            flair = utt.meta['author_flair_text']
            if not flair:
                exclude_id.add(utt.id)
                continue
            if flair in l2_country_names:
                lang = 'eu'
            elif flair in l1_country_names:
                lang = 'en'
            else:
                exclude_id.add(utt.id)
                continue
        else:
            lang = 'en'
        speaker[lang] = True
        utterance_count[lang] += 1
        if lang != 'en' and not flair_based_l2:
            exclude_id.add(utt.id)

        for i, v in enumerate(variants):
            pos = utt.text.find(v)
            if pos >= 0 and (pos == 0 or not utt.text[pos-1].isalpha()) and \
                    (pos + len(v) == len(utt.text) or not utt.text[pos+len(v)].isalpha()):
                utterances[i].append(utt)

    en_only = [s for _, s in speakers.items() if s['en'] and not s[language]]
    foreign_only = [s for _, s in speakers.items() if s[language] and not s['en']]
    bilingual = [s for _, s in speakers.items() if s[language] and s['en']]

    for utt in iter_utterances():
        if utt.id in exclude_id: continue
        tokens = len(utt.text.split(' '))
        if (language and speakers[utt.speaker.id][language]) or (flair_based_l2 and speakers[utt.speaker.id]['eu']):
            token_count['bi'] += tokens
        else:
            token_count['en'] += tokens

    print(f"Speaker counts: mono-native {len(foreign_only)}, mono-EN {len(en_only)}, bilingual {len(bilingual)}")
    print(f"Mono-EN tokens: {token_count['en']}, bilingual tokens: {token_count['bi']}")

    for i, v in enumerate(variants):
        utterances_en = len([utt for utt in utterances[i] if not speakers[utt.speaker.id][language]])
        utterances_bi = len([utt for utt in utterances[i] if speakers[utt.speaker.id][language]])

        print(f"{v}: mono-EN {utterances_en} (pmw {utterances_en * 1_000_000 / token_count['en']})")
        if language:
              print(f"bilingual {utterances_bi} (pmw {utterances_bi * 1_000_000 / token_count['bi']})")

        # for utt in utterances[i]:
        #     print(extract_context(utt, v))

# load_conversation_data('europe')
analyze_subreddit('europe', None, ["try to", "try and"], flair_based_l2=True)
# analyze_subreddit('Netherlands', 'nl', ["didn't use to", "didn't used to", "usedn't to"])
# analyze_subreddit('Leiden', 'nl', ["used to", "try to", "try and"])
# analyze_subreddit('Leiden', 'nl', ["used to", "try to", "try and"])
# analyze_subreddit('paris', 'fr', ["try to", "try and"])
# analyze_subreddit('SriLanka', 'si', ["try to", "try and"])
# analyze_subreddit('Philippines', 'tl', ["try to", "try and"])
