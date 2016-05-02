
#import data
import requests 
from bs4 import BeautifulSoup 
import re

class Scrape:
    def parsed_posts(self, soup_p):
        titles = []

        for post in soup_p:
            titles.append(post.find(class_='hdrlnk').contents[0])
        return titles

    def run(self):
        
        #all data
        agg_posts = []


        current_post = 0

        baseURL='https://sfbay.craigslist.org/search/tia?'

        URL = baseURL + 's=0'
        print 'getting request...' + URL
        response = requests.get(URL)
        print 'OK'
        html = response.content
        soup = BeautifulSoup(html)
        soup_posts = soup.findAll('p')

        while len(soup_posts) > 0:
            agg_posts.extend(self.parsed_posts(soup_posts))
            current_post += 100
            URL = baseURL + '&s=' + str(current_post)
            print 'getting request...' + URL
            response = requests.get(URL)
            print 'OK'
            html = response.content
            soup = BeautifulSoup(html)
            soup_posts = soup.findAll('p')
            print 'page: ' + str(current_post)

        path = 'training.txt'
        with open(path,'w') as f:
            f.writelines([unicode(x.contents[0]+'\n').encode('utf8') for x in agg_posts])
        
        
if __name__ == '__main__':
    a = Scrape()
    a.run()
