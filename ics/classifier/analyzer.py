import re

__author__ = 'Andrea Esuli'

urlre = re.compile(r'https?://[^ ]+')
mentionre = re.compile(r'@\w+')
hashre = re.compile(r'#\w+')
emotre = re.compile(
    r'(:\w+:|<[/\\]?3|[()\\\D|*$][\-^]?[:;=]|[:;=B8][\-^]?[3DOPp@$*\\)(/|])(?=\s|[!.?]|$)')
featre = re.compile(
    r'(https?://[^ ]+|[\w\-]+|#\w+|@\w+|:\w+:|<[/\\]?3|[()\\\D|*$][\-^]?[:;=]|[:;=B8][\-^]?[3DOPp@$*\\)(/|])(?=\s|[;:,!.?]|$)')


def clean_html(html):
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    cleaned = re.sub(r"(?s)<!--(.*?)-->\n?", "", cleaned)
    cleaned = re.sub(r"(?s)<[/\w].*?>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    return cleaned.strip()


def ngrams(items, n, prefix):
    return [prefix + '_'.join(items[start:start + n]) for start in range(0, len(items) - n + 1)]


def custom_analyzer(doc, word_ngrams=None, char_ngrams=None, stopwords=None):
    if word_ngrams is None:
        word_ngrams = list()
    if char_ngrams is None:
        char_ngrams = list()
    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)

    doc = clean_html(doc)
    output = list()
    output.extend(featre.findall(doc))
    output = [x for x in output if len(x) > 1 and x not in stopwords]

    if word_ngrams is None:
        word_ngrams = list()
    ngm = list()
    for n in word_ngrams:
        ngm.extend(ngrams(output, n, '_W%iG_' % n))
    output.extend(ngm)

    ngm = list()
    for n in char_ngrams:
        ngm.extend(ngrams(doc, n, '_C%iG_' % n))
    output.extend(ngm)

    for alttag, regex in [('_URL', urlre), ('_MENTION', mentionre), ('_HASHTAG', hashre), ('_EMOTICON', emotre)]:
        output.extend([alttag for _ in regex.findall(doc)])
    return output
