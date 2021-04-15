The Web Scraper is a search engine that contains, full HTML parsing, File/DB handling. 

The program scans the static corpus (37500 files) that mimics a webpage for all HTML using BeautifulSoup.

It extracts, lemmatizes, maps, and stores each word as tokens into an inverted index. Once all the files are scraped, the user can enter search queries that will scan the inverted index for the relevant word and display the top 20 most relevant URLS. 

The tokens are ranked by relevance using the cosine similarity score with tokens in certain HTML tags having a higher ranking than others.
