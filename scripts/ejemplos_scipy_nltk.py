import scipy.stats as stats
import nltk
from nltk.tokenize import word_tokenize

# Ejemplo con SciPy
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean = stats.tmean(data)
print(f'Media de los datos: {mean}')

# Ejemplo con Nltk
text = "Natural Language Processing with NLTK."
tokens = word_tokenize(text)
print(f'Tokens: {tokens}')