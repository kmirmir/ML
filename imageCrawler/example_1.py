from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': 'image_dir_1'})

google_crawler.crawl(keyword='pop', max_num=10,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)
