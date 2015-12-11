import re

urlre = re.compile(r'http[s]{,1}://[^ ]+')
mentionre = re.compile(r'@[\w]+')
hashre = re.compile(r'#[\w]+')
emotre = re.compile(
    r'(:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')
featre = re.compile(
    r'(http[s]{,1}://[^ ]+|[\w\-]+|#\w+|@\w+|\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[;:,\!\.\?]|$)')


def clean_html(html):
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    cleaned = re.sub(r"(?s)<[/\w].*?>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    return cleaned.strip()


def ngrams(items, n, prefix):
    return [prefix + '_'.join(items[start:start + n]) for start in range(0, len(items) - n + 1)]


def rich_analyzer(doc):
    doc = clean_html(doc)
    output = list()
    output.extend(featre.findall(doc))
    output = [x for x in output if len(x) > 1]

    word_bigrams = ngrams(output, 2, '_WBG_')
    word_trigrams = ngrams(output, 3, '_WTG_')
    output.extend(word_bigrams)
    output.extend(word_trigrams)

    char_trigrams = ngrams(doc, 3, '_CTG_')
    output.extend(char_trigrams)

    for alttag, regex in [('_URL', urlre), ('_MENTION', mentionre), ('_HASHTAG', hashre), ('_EMOTICON', emotre)]:
        output.extend([alttag for _ in regex.findall(doc)])
    return output


if __name__ == '__main__':
    example = 'nel @me_2zzo y < x http://www.es.com/ <br /> <p> <a href=\'ff\'> ddd </a>  :del: https://wd.d cammin ;) e < r  r >= d f): di #nos22tr-fa, #vita.'
    output = rich_analyzer(example)
    print(output)
