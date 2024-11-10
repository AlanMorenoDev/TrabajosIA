from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset_deporte/basquetbol'})
google_crawler.crawl(keyword='Básquetbol', max_num=100)
