# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# License: BSD 3 clause
"""
The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction._hashing_fast import transform
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.preprocessing import normalize


class LightweightRandomIndexingVectorizer(TransformerMixin, _VectorizerMixin,
                                          BaseEstimator):
    """Convert a collection of text documents to a matrix of lightweight
    random indexing-based vectors.

    Parameters
    ----------

    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.

    encoding : string, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : boolean, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer is not callable``.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.
        Only applies if ``analyzer is not callable``.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

        .. versionchanged:: 0.21

        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
        first read from the file and then passed to the given callable
        analyzer.

    n_features : integer, default=(2 ** 20)
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    transform_mode: 'once', 'twice', default='once'
        The current hash function return a single hash for a token.
        Lightweight random indexing requires two hashes.
        Without modifying the underlying cython implementation this can be
        done in a memory efficient way by using generators and running
        tokenization twice on input documents ('twice'), or in a time
        efficient way by saving a list of the tokenized document in memory
        and then computing hashed ('once', default).

    Examples
    --------
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = LightweightRandomIndexingVectorizer(n_features=2**4)
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(X.shape)
    (4, 16)

    See Also
    --------
    CountVectorizer, TfidfVectorizer

    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 norm='l2', dtype=np.float64, transform_mode="once"):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.norm = norm
        self.dtype = dtype
        self.transform_mode = transform_mode

    def partial_fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        return self

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        # triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._warn_for_unused_params()
        self._validate_params()

        return self

    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()

        analyzer = self.build_analyzer()

        # FIX Create a version of the self._hash cython function
        # (sklearn.feature_extraction._hashing_fast.transform) that computes
        # two hashes. This would allow run analyzer once (fast) and using
        # a generator (memory efficient), not forcing to choose only one
        # of the two with the "transform_mode" parameter.
        if self.transform_mode == 'once':
            tokens = [[(x, 1) for x in analyzer(doc)] for doc in X]
            X_0 = self._hash(tokens, seed=0)
            X_1 = self._hash(tokens, seed=1)
        elif self.transform_mode == 'twice':
            X_0 = self._hash((((x, 1) for x in analyzer(doc)) for doc in X),
                             seed=0)
            X_1 = self._hash((((x, 1) for x in analyzer(doc)) for doc in X),
                             seed=1)
        else:
            raise ValueError('Unknown transform_mode: ' + self.transform_mode)

        X = X_0 + X_1

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def _hash(self, raw_X, seed):
        indices, indptr, values = transform(raw_X, self.n_features,
                                            self.dtype, True, seed=seed)
        n_samples = indptr.shape[0] - 1
        if n_samples == 0:
            raise ValueError("Cannot vectorize empty sequence.")
        X = csr_matrix((values, indices, indptr), dtype=self.dtype,
                       shape=(n_samples, self.n_features))
        X.sum_duplicates()  # also sorts the indices
        return X

    def fit_transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {'X_types': ['string']}
