from convokit import Corpus, download
from langdetect import detect, LangDetectException
from collections import defaultdict, Counter

def analyze_subreddit(subreddit, language, variants):
    utterances = [[] for v in variants]

    speakers = defaultdict(lambda: {'en': False, language: False})
    token_count = Counter()
    exclude_id = set()

    corpus = Corpus(filename=download('subreddit-' + subreddit))
    for utt in corpus.iter_utterances():
        speaker_id = utt.speaker.id
        speaker = speakers[speaker_id]

        try:
            lang = detect(utt.text)
            speaker[lang] = True
            if lang != 'en':
                exclude_id.add(utt.id)
        except LangDetectException:
            pass

        for i, v in enumerate(variants):
            if v in utt.text:
                utterances[i].append(utt)

    en_only = [s for _, s in speakers.items() if s['en'] and not s[language]]
    foreign_only = [s for _, s in speakers.items() if s[language] and not s['en']]
    bilingual = [s for _, s in speakers.items() if s[language] and s['en']]

    for utt in corpus.iter_utterances():
        if utt.id in exclude_id: continue
        tokens = len(utt.text.split(' '))
        if speakers[utt.speaker.id][language]:
            token_count['bi'] += tokens
        else:
            token_count['en'] += tokens

    print(f"Speaker counts: mono-native {len(foreign_only)}, mono-EN {len(en_only)}, bilingual {len(bilingual)}")
    print(f"Mono-EN tokens: {token_count['en']}, bilingual tokens: {token_count['bi']}")

    for i, v in enumerate(variants):
        utterances_en = len([utt for utt in utterances[i] if not speakers[utt.speaker.id][language]])
        utterances_bi = len([utt for utt in utterances[i] if speakers[utt.speaker.id][language]])

        print(f"{v}: mono-EN {utterances_en}, bilingual {utterances_bi}")
    #
    # print("Utterance A: " + str(len(a_utterances)) + "; utterance B: " + str(len(b_utterances)))
    # for a_utterance in a_utterances:
    #     print(context(a_utterance, ))
    # for b_utterance in b_utterances:
    #     print(b_utterance.text)

analyze_subreddit('Amsterdam', 'nl', ['try to', 'try and'])
