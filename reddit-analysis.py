import time

from convokit import Corpus, download
from lingua import Language, LanguageDetectorBuilder
from collections import defaultdict, Counter

from lingua.lingua import IsoCode639_1


def extract_context(utt, v):
    pos = utt.text.find(v)
    start = max(pos-20, 0)
    end = min(pos+len(v)+20, len(utt.text))
    return utt.text[start:end].replace('\n', ' ')

def analyze_subreddit(subreddit, language, variants):
    utterances = [[] for v in variants]
    if language:
        lg = Language.from_iso_code_639_1(getattr(IsoCode639_1, language.upper()))
        detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, lg).build()

    speakers = defaultdict(lambda: {'en': False, language: False})
    token_count = Counter()
    utterance_count = Counter()
    exclude_id = set()

    corpus = Corpus(filename=download('subreddit-' + subreddit))
    all_utterances = corpus.utterances.values()
    print(f"Processing {len(all_utterances)} utterances")
    start_time = time.time()
    processed_count = 0
    for utt in all_utterances:
        processed_count += 1
        if processed_count % (len(all_utterances) // 20) == 0:
            current_time = time.time()
            percent = round(processed_count / (len(all_utterances) // 100), 1)
            print(f"{percent}% processed; elapsed time {round(current_time - start_time, 2)} sec; EN {utterance_count['en']}; local {utterance_count[language]}")

        if len(utt.text) < len(variants[0]) or utt.text == '[removed]' or utt.text == '[deleted]':
            continue

        speaker_id = utt.speaker.id
        speaker = speakers[speaker_id]

        if language:
            lg = detector.detect_language_of(utt.text)
            if not lg:
                print("No language for " + utt.text)
                continue
            lang = lg.iso_code_639_1.name.lower()
        else:
            lang = 'en'
        speaker[lang] = True
        utterance_count[lang] += 1
        if lang != 'en':
            exclude_id.add(utt.id)

        for i, v in enumerate(variants):
            pos = utt.text.find(v)
            if pos >= 0 and (pos == 0 or not utt.text[pos-1].isalpha()) and \
                    (pos + len(v) == len(utt.text) or not utt.text[pos+len(v)].isalpha()):
                utterances[i].append(utt)

    en_only = [s for _, s in speakers.items() if s['en'] and not s[language]]
    foreign_only = [s for _, s in speakers.items() if s[language] and not s['en']]
    bilingual = [s for _, s in speakers.items() if s[language] and s['en']]

    for utt in corpus.iter_utterances():
        if utt.id in exclude_id: continue
        tokens = len(utt.text.split(' '))
        if language and speakers[utt.speaker.id][language]:
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

# analyze_subreddit('Netherlands', 'nl', ["didn't use to", "didn't used to", "usedn't to"])
analyze_subreddit('paris', 'fr', ["try to", "try and"])
# analyze_subreddit('SriLanka', 'si', ["try to", "try and"])
